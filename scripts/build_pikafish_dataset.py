"""Build game-split Pikafish supervised train/validation NPZ datasets."""
import argparse
from collections.abc import Callable
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_teacher_labels import save_teacher_npz
from src.engine.uci import PikafishEngine, find_engine_binary
from src.game.game_state import NO_CAPTURE_DRAW_PLY_LIMIT
from src.training.pikafish_dataset import (
    LabeledGame, flatten_games, generate_pikafish_games,
    split_train_validation,
)

DEFAULT_TRAIN_OUTPUT = Path(
    'data/teacher_labels/train.npz')
DEFAULT_VALIDATION_OUTPUT = Path(
    'data/teacher_labels/validation.npz')
DEFAULT_GAMES = 110
DEFAULT_VALIDATION_GAMES = 10
DEFAULT_DEPTH = 8
DEFAULT_OPENING_PLIES = 8
DEFAULT_MAX_GAME_PLIES = 160
DEFAULT_THREADS = 4
DEFAULT_HASH_MB = 256
DEFAULT_SEED = 0
DEFAULT_TIMEOUT = 60.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Build game-split Pikafish supervised datasets',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--train-output', type=Path,
                        default=DEFAULT_TRAIN_OUTPUT,
                        help='training dataset output path')
    parser.add_argument('--validation-output', type=Path,
                        default=DEFAULT_VALIDATION_OUTPUT,
                        help='validation dataset output path')
    parser.add_argument('--games', type=int, default=DEFAULT_GAMES,
                        help='total number of generated games')
    parser.add_argument('--validation-games', type=int,
                        default=DEFAULT_VALIDATION_GAMES,
                        help='number of trailing games reserved for validation')
    parser.add_argument('--depth', type=int, default=DEFAULT_DEPTH,
                        help='Pikafish search depth per labeled position')
    parser.add_argument('--opening-plies', type=int,
                        default=DEFAULT_OPENING_PLIES,
                        help='initial plies sampled from legal moves')
    parser.add_argument('--max-game-plies', type=int,
                        default=DEFAULT_MAX_GAME_PLIES,
                        help='maximum plies per generated game')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS,
                        help='Pikafish thread count')
    parser.add_argument('--hash-mb', type=int, default=DEFAULT_HASH_MB,
                        help='Pikafish transposition-table size in MB')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED,
                        help='random seed for opening diversity')
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT,
                        help='per-position search timeout in seconds')
    args = parser.parse_args(argv)
    _validate_args(parser, args)
    return args


def _validate_args(parser: argparse.ArgumentParser,
                   args: argparse.Namespace) -> None:
    """Fail before launching Pikafish when the requested split is invalid."""
    if args.train_output.resolve() == args.validation_output.resolve():
        parser.error('--train-output and --validation-output must differ')
    if args.games <= 0:
        parser.error('--games must be positive')
    if not 0 < args.validation_games < args.games:
        parser.error(
            '--validation-games must leave at least one train and validation game')
    if args.depth <= 0:
        parser.error('--depth must be positive')
    if args.opening_plies < 0:
        parser.error('--opening-plies must be nonnegative')
    if args.max_game_plies <= 0:
        parser.error('--max-game-plies must be positive')
    if args.threads <= 0 or args.hash_mb <= 0:
        parser.error('--threads and --hash-mb must be positive')
    if args.timeout <= 0:
        parser.error('--timeout must be positive')


def _save_games(path: Path, games: list[LabeledGame], depth: int,
                engine_path: Path,
                max_game_plies: int = DEFAULT_MAX_GAME_PLIES) -> int:
    """Flatten one game split, save it with state metadata, and return its size."""
    states, policies, values, game_ids = flatten_games(games)
    save_teacher_npz(path, states, policies, values, depth=depth,
                     engine_path=engine_path, game_ids=game_ids,
                     max_game_plies=max_game_plies,
                     no_capture_limit=NO_CAPTURE_DRAW_PLY_LIMIT)
    return len(states)


def _progress_reporter(
        total_games: int, started: float
) -> Callable[[int, LabeledGame], None]:
    """Print per-game progress with cumulative throughput and ETA."""
    completed_positions = 0

    def report(completed_games: int, game: LabeledGame) -> None:
        nonlocal completed_positions
        completed_positions += len(game.states)
        elapsed = time.perf_counter() - started
        rate = completed_positions / elapsed if elapsed > 0 else 0.0
        eta = elapsed / completed_games * (total_games - completed_games)
        print(f'[{completed_games}/{total_games}] game {game.game_id}: '
              f'{len(game.states)} plies | {completed_positions} positions | '
              f'{elapsed:.1f}s elapsed | {rate:.2f} pos/s | '
              f'ETA {eta:.0f}s', flush=True)

    return report


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    binary = find_engine_binary()
    started = time.perf_counter()
    with PikafishEngine(path=binary, threads=args.threads,
                        hash_mb=args.hash_mb) as engine:
        games = generate_pikafish_games(
            engine, args.games, args.depth, args.seed,
            opening_plies=args.opening_plies,
            max_game_plies=args.max_game_plies, timeout=args.timeout,
            progress=_progress_reporter(args.games, started))
    elapsed = time.perf_counter() - started

    train_games, validation_games = split_train_validation(
        games, args.validation_games)
    train_positions = _save_games(
        args.train_output, train_games, args.depth, binary, args.max_game_plies)
    validation_positions = _save_games(
        args.validation_output, validation_games, args.depth, binary,
        args.max_game_plies)
    total_positions = train_positions + validation_positions

    print(f'Generated {len(games)} games ({total_positions} positions) '
          f'in {elapsed:.1f}s')
    print(f'Train: {len(train_games)} games, {train_positions} positions '
          f'-> {args.train_output}')
    print(f'Validation: {len(validation_games)} games, '
          f'{validation_positions} positions -> {args.validation_output}')
    print(f'Throughput: {total_positions / elapsed:.2f} positions/s')


if __name__ == '__main__':
    main()
