"""Calibration math and runner for the Pikafish shard farm.

A calibration run plays one game per worker at each requested depth under
full worker load in a throwaway temporary workspace, then extrapolates the
measured game cost to an expected positions-per-hour rate. The CLI injects
the shard worker, its monitor and the archive counter, so this module never
imports ``scripts``.
"""

from argparse import Namespace
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, median
import tempfile
import threading
import time

from .job_queue import JobQueue, ShardJob
from .shard_farm import FarmResult
from .shard_plan import DepthTarget

SECONDS_PER_HOUR: float = 3600.0
CALIBRATION_HEADER = (
    'depth  median_s/game  mean_positions/game  est_positions/hour')

RunJob = Callable[[ShardJob, Namespace], bool]
CountPositions = Callable[[Path], int]
RunToCompletion = Callable[
    [JobQueue, Callable[[ShardJob], bool], Namespace, threading.Event,
     float | None],
    FarmResult,
]


@dataclass(frozen=True)
class CalibrationRow:
    """Measured game cost and projected throughput for one search depth."""

    depth: int
    median_seconds_per_game: float
    mean_positions_per_game: float
    est_positions_per_hour: float


@dataclass(frozen=True)
class CalibrationTools:
    """Farm collaborators the CLI injects to keep this module script-free."""

    run_job: RunJob
    run_to_completion: RunToCompletion
    count_positions: CountPositions


def summarize_calibration(
        depth: int,
        seconds: list[float],
        positions: list[int],
        workers: int,
) -> CalibrationRow:
    """Summarize one depth's calibration games into a throughput row.

    ``seconds`` holds one wall-clock duration per measured game and
    ``positions`` the labeled positions produced by each game. The estimate
    assumes every worker keeps producing games back to back, so it scales
    the median game duration by ``workers``.
    """
    _validate_inputs(seconds, positions, workers)
    median_seconds = median(seconds)
    mean_positions = fmean(positions)
    return CalibrationRow(
        depth=depth,
        median_seconds_per_game=median_seconds,
        mean_positions_per_game=mean_positions,
        est_positions_per_hour=(
            workers * SECONDS_PER_HOUR / median_seconds * mean_positions),
    )


def run_calibration(options: Namespace, tools: CalibrationTools) -> int:
    """Benchmark every requested depth in a throwaway temporary workspace.

    Prints one aligned table row per depth in target order and returns a
    process exit code. Nothing is written to ``options.out_root``: the
    queue, worker logs and stop file are redirected into the workspace,
    which is deleted before returning.
    """
    print(CALIBRATION_HEADER, flush=True)
    with tempfile.TemporaryDirectory(
            prefix='pikafish_calibration_') as temp:
        workspace = _workspace_options(options, Path(temp))
        for target in options.targets:
            row = _summarize_depth(target.depth, workspace, tools)
            if row is None:
                return 1
            _print_row(row)
    return 0


def _workspace_options(options: Namespace, workspace: Path) -> Namespace:
    """Return a copy of ``options`` that writes only inside ``workspace``."""
    calibration = Namespace(**vars(options))
    calibration.out_root = workspace / 'out'
    calibration.log_dir = workspace / 'logs'
    calibration.stop_file = workspace / 'STOP'
    calibration.log_dir.mkdir(parents=True, exist_ok=True)
    return calibration


def _summarize_depth(
        depth: int, options: Namespace, tools: CalibrationTools,
) -> CalibrationRow | None:
    """Play ``workers`` one-game shards at ``depth`` and summarize them."""
    queue = JobQueue([DepthTarget(depth, options.workers)], options.out_root,
                     games_per_shard=1)
    measurements: list[tuple[float, int]] = []

    def run_timed(job: ShardJob) -> bool:
        started = time.perf_counter()
        if not tools.run_job(job, options):
            return False
        measurements.append((
            time.perf_counter() - started, tools.count_positions(job.path)))
        return True

    result = tools.run_to_completion(
        queue, run_timed, options, threading.Event(), deadline=None)
    if result.aborted or len(measurements) != options.workers:
        print(f'depth {depth}: calibration failed '
              f'({result.failed} failed attempt(s))', flush=True)
        return None
    return summarize_calibration(
        depth,
        [elapsed for elapsed, _ in measurements],
        [count for _, count in measurements],
        options.workers,
    )


def _print_row(row: CalibrationRow) -> None:
    """Print one aligned calibration table row."""
    print(f'{row.depth:>5}  {row.median_seconds_per_game:>13.1f}  '
          f'{row.mean_positions_per_game:>19.1f}  '
          f'{row.est_positions_per_hour:>18,.0f}', flush=True)


def _validate_inputs(
        seconds: list[float], positions: list[int], workers: int) -> None:
    """Reject inputs that would make the throughput estimate meaningless."""
    if workers <= 0:
        raise ValueError(f'workers must be positive, got {workers}')
    if not seconds:
        raise ValueError('calibration needs at least one measured game')
    if len(seconds) != len(positions):
        raise ValueError(
            'seconds and positions differ in length: '
            f'{len(seconds)} != {len(positions)}')
    if any(value <= 0 for value in seconds):
        raise ValueError('every calibrated game must take positive seconds')
