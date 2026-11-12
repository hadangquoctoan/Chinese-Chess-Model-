"""Merge per-depth Pikafish farm shards into one teacher-label archive.

The farm writes completed shards into ``<out-root>/depth<d>/`` and keeps
per-depth game-id progress in ``progress.json``. This script appends every
shard that is not already in the output archive for one depth. It never
deletes shards: removing downloaded shard files is the user's job, which
is safe because the game-id cursor in ``progress.json`` stops the farm
from reusing ids.
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.run_pikafish_dataset_loop import merge_shards
from scripts.run_pikafish_farm import DEFAULT_OUT_ROOT
from src.engine.uci import find_engine_binary
from src.training.shard_plan import (
    depth_dir, parse_shard_filename, read_game_ids,
)

DEFAULT_MAX_GAME_PLIES = 160
DEFAULT_OUTPUT_DIR = Path('data') / 'teacher_labels'


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse and validate the merge command line."""
    parser = argparse.ArgumentParser(
        description='Merge one depth of Pikafish farm shards into a dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT,
                        help='farm output root holding depth<d>/ shard dirs')
    parser.add_argument('--depth', type=int, required=True)
    parser.add_argument('--output', type=Path, default=None,
                        help='default: '
                             f'{DEFAULT_OUTPUT_DIR}/train_depth<depth>.npz')
    parser.add_argument('--max-game-plies', type=int,
                        default=DEFAULT_MAX_GAME_PLIES)
    args = parser.parse_args(argv)
    if args.depth <= 0 or args.max_game_plies <= 0:
        parser.error('--depth and --max-game-plies must be positive')
    if args.output is None:
        args.output = DEFAULT_OUTPUT_DIR / f'train_depth{args.depth}.npz'
    return args


def _shards_by_start(shard_dir: Path) -> list[Path]:
    """Return completed shard files sorted by their game-id start."""
    if not shard_dir.is_dir():
        return []
    shards: list[tuple[int, Path]] = []
    for path in shard_dir.glob('*.npz'):
        parsed = parse_shard_filename(path.name)
        if parsed is not None:
            shards.append((parsed[1], path))
    return [path for _, path in sorted(shards)]


def select_new_shards(output: Path, shard_dir: Path) -> list[Path]:
    """Return the unmerged shards of ``shard_dir`` by ``game_id_start``.

    A shard whose game ids are all present in ``output`` was already
    merged and is skipped. A shard sharing only part of its ids with the
    output means the archive or the shard set is inconsistent, so this
    raises ``ValueError`` instead of guessing.
    """
    existing = read_game_ids(output)
    selected: list[Path] = []
    for shard in _shards_by_start(shard_dir):
        ids = read_game_ids(shard)
        if ids <= existing:
            continue
        if ids & existing:
            raise ValueError(
                f'{shard} partially overlaps game_ids in {output}')
        selected.append(shard)
    return selected


def main(argv: list[str] | None = None) -> int:
    """Merge every new shard for one depth and return an exit code.

    Shards are left in place. Deleting them after downloading is the
    user's responsibility and is safe because the farm's ID cursor lives
    in each depth directory's ``progress.json``.
    """
    args = _parse_args(argv)
    shard_dir = depth_dir(args.out_root, args.depth)
    shards = select_new_shards(args.output, shard_dir)
    if not shards:
        print('nothing to merge', flush=True)
        return 0
    positions, games = merge_shards(
        args.output, shards, depth=args.depth,
        engine_path=find_engine_binary(),
        max_game_plies=args.max_game_plies)
    print(f'merged {len(shards)} shard(s): {games} games, '
          f'{positions} positions into {args.output}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
