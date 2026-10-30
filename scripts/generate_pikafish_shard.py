"""Generate one Pikafish self-play dataset shard."""
import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_teacher_labels import save_teacher_npz
from src.engine.uci import PikafishEngine, find_engine_binary
from src.game.game_state import NO_CAPTURE_DRAW_PLY_LIMIT
from src.training.pikafish_dataset import (
    LabeledGame, flatten_games, generate_pikafish_games,
)

DEFAULT_DEPTH = 8
DEFAULT_OPENING_PLIES = 12
DEFAULT_MAX_GAME_PLIES = 160
DEFAULT_THREADS = 4
DEFAULT_HASH_MB = 256
DEFAULT_TIMEOUT = 60.0


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='Generate one Pikafish self-play shard',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--games', type=int, required=True)
    parser.add_argument('--game-id-start', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--depth', type=int, default=DEFAULT_DEPTH)
    parser.add_argument('--opening-plies', type=int,
                        default=DEFAULT_OPENING_PLIES)
    parser.add_argument('--max-game-plies', type=int,
                        default=DEFAULT_MAX_GAME_PLIES)
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS)
    parser.add_argument('--hash-mb', type=int, default=DEFAULT_HASH_MB)
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT)
    args = parser.parse_args(argv)
    if args.games <= 0:
        parser.error('--games must be positive')
    if args.game_id_start < 0:
        parser.error('--game-id-start must be nonnegative')
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
    return args


def _report(completed_games: int, total_games: int, game: LabeledGame) -> None:
    print(f'[{completed_games}/{total_games}] game {game.game_id}: '
          f'{len(game.states)} plies', flush=True)


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
            game_id_start=args.game_id_start,
            progress=lambda completed, game: _report(
                completed, args.games, game))
    states, policies, values, game_ids = flatten_games(games)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    partial = args.output.with_name(f'{args.output.stem}.partial.npz')
    save_teacher_npz(partial, states, policies, values, depth=args.depth,
                     engine_path=binary, game_ids=game_ids,
                     max_game_plies=args.max_game_plies,
                     no_capture_limit=NO_CAPTURE_DRAW_PLY_LIMIT)
    partial.replace(args.output)
    elapsed = time.perf_counter() - started
    print(f'wrote {len(states)} positions from {len(games)} games '
          f'to {args.output} in {elapsed:.1f}s', flush=True)


if __name__ == '__main__':
    main()
