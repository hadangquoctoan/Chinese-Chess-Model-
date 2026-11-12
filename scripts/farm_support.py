"""Status reporting and stop signalling for the Pikafish shard farm.

Kept separate from the CLI so the entry point stays small: this module owns
the thread-safe per-depth counters, the plan summary, the periodic status
lines, the final result report, and the monitor that turns the ``--hours``
deadline or a STOP file into a cooperative stop signal.
"""

from pathlib import Path
import threading
import time

from src.training.job_queue import JobQueue
from src.training.shard_farm import FarmResult
from src.training.shard_plan import DepthTarget

STOP_POLL_SECONDS: float = 2.0


class FarmStats:
    """Thread-safe per-depth position and running-worker counters."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._positions: dict[int, int] = {}
        self._running: dict[int, int] = {}

    def begin(self, depth: int) -> None:
        """Count one worker as running at ``depth``."""
        with self._lock:
            self._running[depth] = self._running.get(depth, 0) + 1

    def end(self, depth: int) -> None:
        """Count one worker at ``depth`` as no longer running."""
        with self._lock:
            self._running[depth] = max(0, self._running.get(depth, 0) - 1)

    def record(self, depth: int, positions: int) -> None:
        """Add ``positions`` from one completed shard at ``depth``."""
        with self._lock:
            self._positions[depth] = self._positions.get(depth, 0) + positions

    def snapshot(self) -> tuple[dict[int, int], dict[int, int]]:
        """Return copies of the positions and running-worker counters."""
        with self._lock:
            return dict(self._positions), dict(self._running)


def print_plan(queue: JobQueue, targets: list[DepthTarget]) -> None:
    """Print games already completed versus the target for every depth."""
    progress = queue.progress()
    for target in targets:
        print(f'depth {target.depth}: {progress[target.depth].games_done}'
              f'/{_target_games(target)} games already done', flush=True)


def print_status(queue: JobQueue, stats: FarmStats,
                 targets: list[DepthTarget], started: float) -> None:
    """Print one status line per depth with games, positions and rate."""
    elapsed_hours = max(time.monotonic() - started, 1.0) / 3600.0
    progress = queue.progress()
    positions, running = stats.snapshot()
    for target in targets:
        count = positions.get(target.depth, 0)
        print(f'[status] depth {target.depth}: '
              f'{progress[target.depth].games_done}'
              f'/{_target_games(target)} games, {count} positions, '
              f'{count / elapsed_hours:.0f} positions/h, '
              f'{running.get(target.depth, 0)} worker(s) active', flush=True)


def status_loop(stop: threading.Event, interval: float, queue: JobQueue,
                stats: FarmStats, targets: list[DepthTarget],
                started: float) -> None:
    """Print status lines every ``interval`` seconds until ``stop`` is set."""
    while not stop.wait(interval):
        print_status(queue, stats, targets, started)


def wait_for_farm(farmer: threading.Thread, stop: threading.Event,
                  deadline: float | None, stop_file: Path) -> str:
    """Join the farm thread, stopping it on deadline expiry or a stop file.

    Returns the stop reason (``'done'``, ``'deadline'`` or ``'stop-file'``)
    and always joins the farm thread so in-flight shards can finish.
    """
    reason = 'done'
    while farmer.is_alive():
        farmer.join(STOP_POLL_SECONDS)
        if not farmer.is_alive():
            break
        if deadline is not None and time.monotonic() >= deadline:
            reason = 'deadline'
            break
        if stop_file.exists():
            reason = 'stop-file'
            break
    if reason != 'done':
        stop.set()
    farmer.join()
    return reason


def _target_games(target: DepthTarget) -> str:
    """Render a target's game budget for status lines."""
    return 'inf' if target.games is None else str(target.games)


def latest_worker_log(log_dir: Path) -> Path | None:
    """Return the most recently modified worker log, if any."""
    logs = sorted(log_dir.glob('*.log'), key=lambda path: path.stat().st_mtime)
    return logs[-1] if logs else None


def report_result(result: FarmResult, log_dir: Path) -> int:
    """Print the farm outcome and map it to a process exit code."""
    if not result.aborted:
        print(f'farm finished: {result.completed} shard(s) written, '
              f'{result.failed} failed attempt(s)', flush=True)
        return 0
    latest = latest_worker_log(log_dir)
    location = f'check log: {latest}' if latest else 'no worker log found'
    print(f'farm aborted after {result.failed} failed attempt(s); '
          f'{location}', flush=True)
    return 1
