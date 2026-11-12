"""Shard naming, durable game-id progress and depth target parsing.

This is the pure domain layer of the Pikafish data farm: it parses
``--target`` text into depth plans, names shard files, and keeps per-depth
game-id progress across restarts. It performs no engine work and has no
dependency on PyTorch or Pikafish.
"""

from dataclasses import dataclass
import json
import os
from pathlib import Path
import re
import tempfile

GAME_ID_STRIDE: int = 10_000_000

PROGRESS_FILENAME = 'progress.json'
PARTIAL_SUFFIX = '.partial.npz'

_SHARD_PATTERN = re.compile(
    r'd(?P<depth>\d+)_g(?P<start>\d+)_n(?P<games>\d+)\.npz')


@dataclass(frozen=True)
class DepthTarget:
    """One requested search depth and its optional game budget.

    ``games is None`` means the depth runs without a game limit.
    """

    depth: int
    games: int | None


@dataclass(frozen=True)
class DepthProgress:
    """Durable progress counters for one depth directory."""

    next_game_id: int
    games_done: int


def parse_targets(text: str) -> list[DepthTarget]:
    """Parse ``depth:games`` targets such as ``3:200,9:inf``.

    ``inf`` is valid only for the last target and means no game limit.
    Depths must be positive and unique. Raises ``ValueError`` on bad input.
    """
    parts = text.split(',')
    targets: list[DepthTarget] = []
    seen_depths: set[int] = set()
    for index, part in enumerate(parts):
        depth_text, separator, games_text = part.partition(':')
        if not separator or not depth_text or not games_text:
            raise ValueError(f'invalid target {part!r}: expected depth:games')
        depth = _parse_positive_int(depth_text, 'depth')
        if depth in seen_depths:
            raise ValueError(f'duplicate depth {depth} in targets {text!r}')
        seen_depths.add(depth)
        if games_text == 'inf':
            if index != len(parts) - 1:
                raise ValueError(
                    f"'inf' is only allowed on the last target: {text!r}")
            targets.append(DepthTarget(depth, None))
        else:
            games = _parse_positive_int(games_text, 'games')
            targets.append(DepthTarget(depth, games))
    return targets


def game_id_base(depth: int) -> int:
    """Return the first game id reserved for ``depth``."""
    return depth * GAME_ID_STRIDE


def depth_dir(out_root: Path, depth: int) -> Path:
    """Return the output directory holding all shards for ``depth``."""
    return out_root / f'depth{depth}'


def shard_filename(depth: int, game_id_start: int, games: int) -> str:
    """Return the canonical name of a completed shard of games."""
    if depth <= 0 or game_id_start < 0 or games <= 0:
        raise ValueError(
            f'invalid shard coordinates: depth={depth}, '
            f'game_id_start={game_id_start}, games={games}')
    return f'd{depth}_g{game_id_start}_n{games}.npz'


def parse_shard_filename(name: str) -> tuple[int, int, int] | None:
    """Return ``(depth, game_id_start, games)`` or ``None`` for non-shards.

    Partial files such as ``d3_g1_n1.partial.npz`` are never shards.
    """
    match = _SHARD_PATTERN.fullmatch(name)
    if match is None:
        return None
    return (int(match['depth']), int(match['start']), int(match['games']))


def load_progress(directory: Path, depth: int) -> DepthProgress:
    """Load the field-wise maximum progress for ``depth`` in ``directory``.

    Shard names track completed games; ``progress.json`` keeps progress
    durable after shards are merged and deleted. A missing or corrupt
    ``progress.json`` contributes nothing.
    """
    next_game_id = game_id_base(depth)
    games_done = 0
    for path in directory.glob('*.npz'):
        parsed = parse_shard_filename(path.name)
        if parsed is None or parsed[0] != depth:
            continue
        _, game_id_start, games = parsed
        next_game_id = max(next_game_id, game_id_start + games)
        games_done += games

    stored = _read_progress_json(directory / PROGRESS_FILENAME)
    if stored is not None:
        next_game_id = max(next_game_id, stored.next_game_id)
        games_done = max(games_done, stored.games_done)
    return DepthProgress(next_game_id, games_done)


def save_progress(directory: Path, progress: DepthProgress) -> None:
    """Atomically write ``progress.json`` inside ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    payload = {
        'next_game_id': progress.next_game_id,
        'games_done': progress.games_done,
    }
    descriptor, temp_name = tempfile.mkstemp(
        prefix='progress.', suffix='.tmp', dir=directory)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as handle:
            json.dump(payload, handle)
            handle.write('\n')
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, directory / PROGRESS_FILENAME)
    except OSError:
        temp_path.unlink(missing_ok=True)
        raise


def remove_partial_files(directory: Path) -> int:
    """Delete ``*.partial.npz`` leftovers and return how many were removed."""
    removed = 0
    for path in directory.glob(f'*{PARTIAL_SUFFIX}'):
        path.unlink(missing_ok=True)
        removed += 1
    return removed


def _parse_positive_int(text: str, field: str) -> int:
    """Convert ``text`` to a positive integer or raise ``ValueError``."""
    try:
        value = int(text)
    except ValueError as error:
        raise ValueError(
            f'invalid {field} {text!r}: expected an integer') from error
    if value <= 0:
        raise ValueError(
            f'invalid {field} {text!r}: expected a positive integer')
    return value


def _read_progress_json(path: Path) -> DepthProgress | None:
    """Read a progress file, treating missing or corrupt content as absent."""
    try:
        with path.open(encoding='utf-8') as handle:
            payload = json.load(handle)
        return DepthProgress(
            int(payload['next_game_id']), int(payload['games_done']))
    except (OSError, ValueError, KeyError, TypeError):
        return None
