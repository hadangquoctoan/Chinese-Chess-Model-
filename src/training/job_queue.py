"""Thread-safe shard job queue with a durable game-id cursor.

Workers claim :class:`ShardJob` instances from a :class:`JobQueue`. Every
claim reserves a disjoint game-id range and persists the cursor, so restarts
and retries never reuse game ids. A failed shard keeps its id range and gets
a higher attempt until ``max_attempts`` is exhausted; a dropped shard
releases its reservation so a fresh range can fill the target.
"""

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
import threading

from .shard_plan import (
    DepthProgress,
    DepthTarget,
    depth_dir,
    game_id_base,
    load_progress,
    remove_partial_files,
    save_progress,
    shard_filename,
)

DEFAULT_GAMES_PER_SHARD: dict[int, int] = {
    3: 20, 4: 20, 5: 16, 6: 12, 7: 10, 8: 8, 9: 6}
FALLBACK_GAMES_PER_SHARD: int = 8


@dataclass(frozen=True)
class ShardJob:
    """One claimable shard of games at a fixed search depth."""

    depth: int
    game_id_start: int
    games: int
    path: Path
    attempt: int = 0


@dataclass
class _DepthState:
    """Mutable bookkeeping for one depth in the queue."""

    target_games: int | None
    shard_size: int
    next_game_id: int
    games_done: int
    outstanding: int = 0


class JobQueue:
    """Thread-safe producer of shard jobs with a durable game-id cursor."""

    def __init__(
            self,
            targets: list[DepthTarget],
            out_root: Path,
            games_per_shard: int | Mapping[int, int] | None = None,
            max_attempts: int = 3,
    ) -> None:
        if max_attempts < 1:
            raise ValueError(
                f'max_attempts must be positive, got {max_attempts}')
        self._lock = threading.Lock()
        self._max_attempts = max_attempts
        self._order: list[int] = []
        self._states: dict[int, _DepthState] = {}
        self._directories: dict[int, Path] = {}
        self._retries: deque[ShardJob] = deque()
        for target in targets:
            self._add_target(target, out_root, games_per_shard)

    def claim(self) -> ShardJob | None:
        """Return the next unclaimed job, or ``None`` when targets are met.

        Retried jobs are handed out before fresh ranges. The durable cursor
        is saved before the job is returned.
        """
        with self._lock:
            job = self._retries.popleft() if self._retries else None
            if job is None:
                job = self._fresh_job()
            if job is None:
                return None
            self._persist(job.depth, self._states[job.depth])
            return job

    def complete(self, job: ShardJob) -> None:
        """Record a finished shard and persist the updated progress."""
        with self._lock:
            state = self._states[job.depth]
            state.outstanding -= job.games
            state.games_done += job.games
            self._persist(job.depth, state)

    def fail(self, job: ShardJob) -> bool:
        """Requeue ``job`` with ``attempt + 1`` or drop it when exhausted.

        Returns ``True`` when the job was requeued and ``False`` when it was
        dropped. A dropped job releases its outstanding reservation so a
        fresh id range can fill the target.
        """
        with self._lock:
            if job.attempt + 1 >= self._max_attempts:
                self._states[job.depth].outstanding -= job.games
                return False
            self._retries.append(replace(job, attempt=job.attempt + 1))
            return True

    def progress(self) -> dict[int, DepthProgress]:
        """Return a thread-safe snapshot of per-depth progress counters."""
        with self._lock:
            return {
                depth: DepthProgress(state.next_game_id, state.games_done)
                for depth, state in self._states.items()
            }

    def _add_target(
            self,
            target: DepthTarget,
            out_root: Path,
            games_per_shard: int | Mapping[int, int] | None,
    ) -> None:
        """Prepare one depth directory and load its durable progress."""
        directory = depth_dir(out_root, target.depth)
        directory.mkdir(parents=True, exist_ok=True)
        remove_partial_files(directory)
        stored = load_progress(directory, target.depth)
        self._order.append(target.depth)
        self._directories[target.depth] = directory
        self._states[target.depth] = _DepthState(
            target_games=target.games,
            shard_size=_games_per_shard(games_per_shard, target.depth),
            next_game_id=max(stored.next_game_id, game_id_base(target.depth)),
            games_done=stored.games_done,
        )

    def _fresh_job(self) -> ShardJob | None:
        """Reserve the next under-target id range in target order."""
        for depth in self._order:
            state = self._states[depth]
            remaining = _remaining_games(state)
            if remaining is not None and remaining <= 0:
                continue
            games = state.shard_size if remaining is None else min(
                state.shard_size, remaining)
            start = max(state.next_game_id, game_id_base(depth))
            state.next_game_id = start + games
            state.outstanding += games
            path = self._directories[depth] / shard_filename(
                depth, start, games)
            return ShardJob(depth, start, games, path)
        return None

    def _persist(self, depth: int, state: _DepthState) -> None:
        """Write the durable cursor for one depth (caller holds the lock)."""
        save_progress(
            self._directories[depth],
            DepthProgress(state.next_game_id, state.games_done))


def _games_per_shard(
        override: int | Mapping[int, int] | None, depth: int) -> int:
    """Resolve the shard size for ``depth`` from an override or defaults."""
    if override is None:
        size = DEFAULT_GAMES_PER_SHARD.get(depth, FALLBACK_GAMES_PER_SHARD)
    elif isinstance(override, Mapping):
        size = override.get(depth, FALLBACK_GAMES_PER_SHARD)
    else:
        size = override
    if size <= 0:
        raise ValueError(f'games_per_shard must be positive, got {size}')
    return size


def _remaining_games(state: _DepthState) -> int | None:
    """Return games still to issue, or ``None`` for an unlimited target."""
    if state.target_games is None:
        return None
    return state.target_games - state.games_done - state.outstanding
