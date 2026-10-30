"""Continuously generate Pikafish shards in parallel and merge them."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_teacher_labels import save_teacher_npz
from src.engine.uci import find_engine_binary
from src.game.game_state import NO_CAPTURE_DRAW_PLY_LIMIT
from src.game.rules import ACTION_ENCODING_VERSION, RULESET_VERSION
from src.training.teacher_label import SCORE_SCALE

DEFAULT_TRAIN = Path('data/teacher_labels/train.npz')
DEFAULT_VALIDATION = Path('data/teacher_labels/validation.npz')
DEFAULT_SHARDS_DIR = Path('data/teacher_labels/shards')
DEFAULT_LOG_DIR = Path('logs/pikafish_dataset')
DEFAULT_WORKERS = 2
DEFAULT_GAMES_PER_WORKER = 20
DEFAULT_DEPTH = 8
DEFAULT_OPENING_PLIES = 12
DEFAULT_MAX_GAME_PLIES = 160
DEFAULT_THREADS = 4
DEFAULT_HASH_MB = 256
DEFAULT_SEED = 0
DEFAULT_TIMEOUT = 60.0
METADATA_FIELDS = (
    'engine_depth', 'max_game_plies', 'no_capture_limit',
    'action_encoding_version', 'ruleset_version',
    'value_convention', 'score_scale',
)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Continuously generate and merge Pikafish dataset shards',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-output', type=Path, default=DEFAULT_TRAIN)
    parser.add_argument('--validation', type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument('--shards-dir', type=Path, default=DEFAULT_SHARDS_DIR)
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--workers', type=int, default=DEFAULT_WORKERS)
    parser.add_argument('--games-per-worker', type=int,
                        default=DEFAULT_GAMES_PER_WORKER)
    parser.add_argument('--batches', type=int, default=0,
                        help='number of batches; 0 means run continuously')
    parser.add_argument('--sleep-seconds', type=float, default=0.0)
    parser.add_argument('--depth', type=int, default=DEFAULT_DEPTH)
    parser.add_argument('--opening-plies', type=int,
                        default=DEFAULT_OPENING_PLIES)
    parser.add_argument('--max-game-plies', type=int,
                        default=DEFAULT_MAX_GAME_PLIES)
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS)
    parser.add_argument('--hash-mb', type=int, default=DEFAULT_HASH_MB)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--keep-shards', action='store_true')
    parser.add_argument('--keep-worker-logs', action='store_true')
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser,
                   args: argparse.Namespace) -> None:
    if args.workers <= 0 or args.games_per_worker <= 0:
        parser.error('--workers and --games-per-worker must be positive')
    if args.batches < 0 or args.sleep_seconds < 0:
        parser.error('--batches and --sleep-seconds must be nonnegative')
    if args.depth <= 0 or args.max_game_plies <= 0:
        parser.error('--depth and --max-game-plies must be positive')
    if args.opening_plies < 0:
        parser.error('--opening-plies must be nonnegative')
    if args.threads <= 0 or args.hash_mb <= 0:
        parser.error('--threads and --hash-mb must be positive')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')
    if args.train_output.resolve() == args.validation.resolve():
        parser.error('--train-output and --validation must differ')


def _archive_metadata(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as archive:
        missing = [field for field in METADATA_FIELDS
                   if field not in archive.files]
        if missing:
            raise ValueError(
                f'{path} is missing metadata: {", ".join(missing)}')
        if 'game_ids' not in archive.files:
            raise ValueError(f'{path} is missing game_ids')
        return {field: archive[field].item() for field in METADATA_FIELDS}


def _check_metadata(path: Path, depth: int, max_game_plies: int) -> None:
    metadata = _archive_metadata(path)
    expected = {
        'engine_depth': depth,
        'max_game_plies': max_game_plies,
        'no_capture_limit': NO_CAPTURE_DRAW_PLY_LIMIT,
        'action_encoding_version': ACTION_ENCODING_VERSION,
        'ruleset_version': RULESET_VERSION,
        'value_convention': 'red_centric',
        'score_scale': SCORE_SCALE,
    }
    mismatches = [
        f'{field}={metadata[field]!r} (expected {value!r})'
        for field, value in expected.items()
        if metadata[field] != value
    ]
    if mismatches:
        raise ValueError(f'{path} metadata mismatch: {"; ".join(mismatches)}')


def _game_ids(path: Path) -> set[int]:
    if not path.is_file():
        return set()
    with np.load(path, allow_pickle=False) as archive:
        if 'game_ids' not in archive.files:
            raise ValueError(f'{path} is missing game_ids')
        return {int(game_id) for game_id in archive['game_ids']}


def next_game_id(train_path: Path, validation_path: Path) -> int:
    """Return the first unused game ID across both archives."""
    ids = _game_ids(train_path) | _game_ids(validation_path)
    return max(ids, default=-1) + 1


def _load_arrays(path: Path) -> tuple[np.ndarray, np.ndarray,
                                     np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return (archive['states'], archive['policies'],
                archive['values'], archive['game_ids'])


def merge_shards(train_path: Path, shard_paths: list[Path], *,
                 depth: int, engine_path: Path,
                 max_game_plies: int) -> tuple[int, int]:
    """Merge shards into the train archive and return (positions, games)."""
    sources: list[Path] = []
    if train_path.is_file():
        _check_metadata(train_path, depth, max_game_plies)
        sources.append(train_path)
    for shard_path in shard_paths:
        _check_metadata(shard_path, depth, max_game_plies)
        sources.append(shard_path)
    if not sources:
        raise ValueError('no datasets to merge')
    arrays = [_load_arrays(path) for path in sources]
    source_ids = [{int(game_id) for game_id in item[3]} for item in arrays]
    all_ids = set().union(*source_ids)
    if sum(len(ids) for ids in source_ids) != len(all_ids):
        raise ValueError('game_ids overlap across train and shards')
    states = np.concatenate([item[0] for item in arrays], axis=0)
    policies = np.concatenate([item[1] for item in arrays], axis=0)
    values = np.concatenate([item[2] for item in arrays], axis=0)
    game_ids = np.concatenate([item[3] for item in arrays], axis=0)
    train_path.parent.mkdir(parents=True, exist_ok=True)
    partial = train_path.with_name(f'{train_path.stem}.partial.npz')
    save_teacher_npz(partial, states, policies, values, depth=depth,
                     engine_path=engine_path, game_ids=game_ids,
                     max_game_plies=max_game_plies,
                     no_capture_limit=NO_CAPTURE_DRAW_PLY_LIMIT)
    partial.replace(train_path)
    return len(states), len(all_ids)


def _worker_command(python: str, root: Path, shard: Path,
                    args: argparse.Namespace, seed: int,
                    game_id_start: int) -> list[str]:
    return [
        python, str(root / 'scripts' / 'generate_pikafish_shard.py'),
        '--output', str(shard),
        '--games', str(args.games_per_worker),
        '--game-id-start', str(game_id_start),
        '--seed', str(seed),
        '--depth', str(args.depth),
        '--opening-plies', str(args.opening_plies),
        '--max-game-plies', str(args.max_game_plies),
        '--threads', str(args.threads),
        '--hash-mb', str(args.hash_mb),
        '--timeout', str(args.timeout),
    ]


def _run_worker(command: list[str], log_path: Path, root: Path) -> int:
    with log_path.open('w', encoding='utf-8') as log:
        try:
            result = subprocess.run(
                command, cwd=str(root), stdout=log,
                stderr=subprocess.STDOUT, check=False)
        except OSError as exc:
            log.write(f'failed to start worker: {exc}\n')
            return 1
    return result.returncode


def _run_batch(python: str, root: Path, args: argparse.Namespace,
               batch_index: int, next_id: int) -> list[tuple[Path, Path]]:
    jobs = []
    for index in range(args.workers):
        shard = args.shards_dir / f'batch_{batch_index:06d}_worker_{index}.npz'
        log_path = args.log_dir / f'batch_{batch_index:06d}_worker_{index}.log'
        game_id_start = next_id + index * args.games_per_worker
        seed = args.seed + game_id_start
        command = _worker_command(python, root, shard, args, seed,
                                  game_id_start)
        print(f'  worker {index}: game_id_start={game_id_start} seed={seed}',
              flush=True)
        jobs.append((command, shard, log_path))
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [(executor.submit(_run_worker, command, log_path, root),
                    shard, log_path) for command, shard, log_path in jobs]
        completed = []
        for future, shard, log_path in futures:
            code = future.result()
            if code == 0 and shard.is_file():
                completed.append((shard, log_path))
            else:
                print(f'worker failed (exit {code}): {log_path}', flush=True)
    return completed


def _pending_shards(train_path: Path, shard_dir: Path) -> list[Path]:
    if not shard_dir.is_dir():
        return []
    existing = _game_ids(train_path)
    pending = []
    for shard in sorted(shard_dir.glob('*.npz')):
        ids = _game_ids(shard)
        if not ids:
            shard.unlink()
        elif ids <= existing:
            shard.unlink()
        elif ids & existing:
            raise ValueError(f'{shard} overlaps existing train game_ids')
        else:
            pending.append(shard)
    return pending


def _delete_shards(shards: list[Path], log_dir: Path,
                   keep_shards: bool, keep_logs: bool) -> None:
    if keep_shards:
        return
    for shard in shards:
        shard.unlink(missing_ok=True)
        if not keep_logs:
            (log_dir / f'{shard.stem}.log').unlink(missing_ok=True)


def _remove_partial_files(args: argparse.Namespace) -> None:
    train_partial = args.train_output.with_name(
        f'{args.train_output.stem}.partial.npz')
    train_partial.unlink(missing_ok=True)
    for partial in args.shards_dir.glob('*.partial.npz'):
        partial.unlink(missing_ok=True)


def _prepare(args: argparse.Namespace, root: Path) -> Path:
    if not args.validation.is_file():
        raise SystemExit(f'validation dataset not found: {args.validation}')
    _check_metadata(args.validation, args.depth, args.max_game_plies)
    if args.train_output.is_file():
        _check_metadata(args.train_output, args.depth, args.max_game_plies)
        overlap = _game_ids(args.train_output) & _game_ids(args.validation)
        if overlap:
            raise SystemExit(
                f'train/validation game_ids overlap: {sorted(overlap)[:5]}')
    engine_path = find_engine_binary()
    args.shards_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    _remove_partial_files(args)
    pending = _pending_shards(args.train_output, args.shards_dir)
    if pending:
        positions, games = merge_shards(
            args.train_output, pending, depth=args.depth,
            engine_path=engine_path, max_game_plies=args.max_game_plies)
        print(f'recovered {len(pending)} shard(s): {games} games, '
              f'{positions} positions', flush=True)
        _delete_shards(pending, args.log_dir, args.keep_shards,
                       args.keep_worker_logs)
    return engine_path


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    engine_path = _prepare(args, root)
    batch_index = 0
    try:
        while args.batches == 0 or batch_index < args.batches:
            next_id = next_game_id(args.train_output, args.validation)
            print(f'[batch {batch_index}] next_game_id={next_id}', flush=True)
            completed = _run_batch(
                args.python, root, args, batch_index, next_id)
            if completed:
                shards = [shard for shard, _ in completed]
                positions, games = merge_shards(
                    args.train_output, shards, depth=args.depth,
                    engine_path=engine_path,
                    max_game_plies=args.max_game_plies)
                print(f'[batch {batch_index}] merged {len(shards)} shard(s): '
                      f'{games} games, {positions} positions', flush=True)
                _delete_shards(shards, args.log_dir, args.keep_shards,
                               args.keep_worker_logs)
            else:
                print(f'[batch {batch_index}] no shard completed; retrying',
                      flush=True)
                time.sleep(max(5.0, args.sleep_seconds))
            batch_index += 1
            if args.sleep_seconds > 0 and completed:
                time.sleep(args.sleep_seconds)
    except KeyboardInterrupt:
        print('stopped by user', flush=True)


if __name__ == '__main__':
    main()
