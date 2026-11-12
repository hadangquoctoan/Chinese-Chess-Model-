"""Run a resumable multi-depth Pikafish self-play shard farm.

Workers claim shards from a durable ``JobQueue``, launch
``scripts/generate_pikafish_shard.py`` in a separate process, and stop
cleanly on Ctrl+C, an expired ``--hours`` budget, or a STOP file. Completed
shards are never regenerated: per-depth game-id progress survives restarts.
``--calibrate`` measures one game per worker per depth in a temporary
workspace and never writes to ``--out-root``.
"""
import argparse
from collections.abc import Callable
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.farm_support import (
    FarmStats, print_plan, status_loop, wait_for_farm,
)
from src.engine.uci import find_engine_binary
from src.training.farm_calibration import CalibrationRow, summarize_calibration
from src.training.job_queue import (
    DEFAULT_GAMES_PER_SHARD, JobQueue, ShardJob,
)
from src.training.shard_farm import FarmResult, run_farm
from src.training.shard_plan import DepthTarget, parse_targets

DEFAULT_OUT_ROOT = Path('data/teacher_labels/farm')
DEFAULT_LOG_DIR = Path('logs/pikafish_farm')
DEFAULT_THREADS = 1
DEFAULT_HASH_MB = 64
DEFAULT_OPENING_PLIES = 12
DEFAULT_MAX_GAME_PLIES = 160
DEFAULT_TIMEOUT = 60.0
DEFAULT_SEED = 0
DEFAULT_STATUS_SECONDS = 60.0
STOP_FILENAME = 'STOP'
CALIBRATION_HEADER = (
    'depth  median_s/game  mean_positions/game  est_positions/hour')


def default_workers(cpu_count: int | None) -> int:
    """Return a worker count that leaves two logical CPUs for the OS."""
    if cpu_count is None:
        return 1
    return max(1, cpu_count - 2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse and validate the farm command line."""
    parser = argparse.ArgumentParser(
        description='Run a multi-depth Pikafish self-play shard farm',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--target', required=True,
                        help='depth:games list, e.g. 3:200,9:inf')
    parser.add_argument('--calibrate', action='store_true',
                        help='measure throughput with one game per worker '
                             'at every --target depth; target game counts '
                             'are ignored')
    parser.add_argument('--workers', type=int,
                        default=default_workers(os.cpu_count()))
    parser.add_argument('--out-root', type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument('--games-per-shard', type=int, default=None,
                        help='games per shard for every depth '
                             f'(default: per-depth {DEFAULT_GAMES_PER_SHARD})')
    parser.add_argument('--hours', type=float, default=None,
                        help='stop claiming new shards after this many hours')
    parser.add_argument('--threads', type=int, default=DEFAULT_THREADS)
    parser.add_argument('--hash-mb', type=int, default=DEFAULT_HASH_MB)
    parser.add_argument('--opening-plies', type=int,
                        default=DEFAULT_OPENING_PLIES)
    parser.add_argument('--max-game-plies', type=int,
                        default=DEFAULT_MAX_GAME_PLIES)
    parser.add_argument('--timeout', type=float, default=DEFAULT_TIMEOUT)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--status-seconds', type=float,
                        default=DEFAULT_STATUS_SECONDS)
    parser.add_argument('--stop-file', type=Path, default=None,
                        help='default: <out-root>/STOP')
    parser.add_argument('--log-dir', type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument('--keep-logs', action='store_true')
    parser.add_argument('--python', default=sys.executable)
    options = parser.parse_args(argv)
    _validate_options(parser, options)
    return options


def _validate_options(parser: argparse.ArgumentParser,
                      options: argparse.Namespace) -> None:
    """Reject bad values through argparse instead of raising tracebacks."""
    try:
        options.targets = parse_targets(options.target)
    except ValueError as error:
        parser.error(str(error))
    if options.workers <= 0:
        parser.error('--workers must be positive')
    if options.games_per_shard is not None and options.games_per_shard <= 0:
        parser.error('--games-per-shard must be positive')
    if options.hours is not None and options.hours <= 0:
        parser.error('--hours must be positive')
    if options.threads <= 0 or options.hash_mb <= 0:
        parser.error('--threads and --hash-mb must be positive')
    if options.opening_plies < 0:
        parser.error('--opening-plies must be nonnegative')
    if options.max_game_plies <= 0:
        parser.error('--max-game-plies must be positive')
    if options.timeout <= 0:
        parser.error('--timeout must be positive')
    if options.status_seconds <= 0:
        parser.error('--status-seconds must be positive')
    # Absolute paths keep the parent and the worker subprocess (which runs
    # with cwd=root) writing to the same directories from any caller cwd.
    options.out_root = options.out_root.resolve()
    options.log_dir = options.log_dir.resolve()
    if options.stop_file is None:
        options.stop_file = options.out_root / STOP_FILENAME
    else:
        options.stop_file = options.stop_file.resolve()


def build_shard_command(
        job: ShardJob, options: argparse.Namespace, root: Path) -> list[str]:
    """Return the worker command line that generates one shard."""
    return [
        options.python,
        str(root / 'scripts' / 'generate_pikafish_shard.py'),
        '--output', str(job.path),
        '--games', str(job.games),
        '--game-id-start', str(job.game_id_start),
        '--seed', str(options.seed + job.game_id_start),
        '--depth', str(job.depth),
        '--opening-plies', str(options.opening_plies),
        '--max-game-plies', str(options.max_game_plies),
        '--threads', str(options.threads),
        '--hash-mb', str(options.hash_mb),
        '--timeout', str(options.timeout),
    ]


def prepare_stop_file(path: Path) -> bool:
    """Remove a stale stop file and report whether one existed."""
    try:
        path.unlink()
    except FileNotFoundError:
        return False
    return True


def _process_group_kwargs() -> dict[str, int | bool]:
    """Isolate worker subprocesses from console interrupts and signals."""
    if sys.platform == 'win32':
        return {'creationflags': subprocess.CREATE_NEW_PROCESS_GROUP}
    return {'start_new_session': True}


def _count_positions(path: Path) -> int:
    """Return how many labeled positions one shard contains."""
    with np.load(path, allow_pickle=False) as archive:
        return int(archive['game_ids'].shape[0])


def run_job(job: ShardJob, options: argparse.Namespace, root: Path,
            stats: FarmStats) -> bool:
    """Generate one shard through the worker script and report success."""
    log_path = options.log_dir / f'{job.path.stem}.log'
    stats.begin(job.depth)
    try:
        with log_path.open('w', encoding='utf-8') as log:
            try:
                completed = subprocess.run(
                    build_shard_command(job, options, root),
                    cwd=str(root), stdout=log, stderr=subprocess.STDOUT,
                    check=False, **_process_group_kwargs())
            except OSError as error:
                log.write(f'failed to start shard worker: {error}\n')
                return False
        if completed.returncode != 0 or not job.path.is_file():
            return False
        stats.record(job.depth, _count_positions(job.path))
        if not options.keep_logs:
            log_path.unlink(missing_ok=True)
        return True
    finally:
        stats.end(job.depth)


def _wait_uninterrupted(farmer: threading.Thread, stop: threading.Event,
                        deadline: float | None, stop_file: Path) -> str:
    """Wait for the farm; on Ctrl+C set ``stop`` and drain in-flight shards."""
    try:
        return wait_for_farm(farmer, stop, deadline, stop_file)
    except KeyboardInterrupt:
        print('interrupted; waiting for running shards to finish', flush=True)
        stop.set()
        farmer.join()
        return 'interrupt'


def _run_to_completion(
        queue: JobQueue, execute: Callable[[ShardJob], bool],
        options: argparse.Namespace, stop: threading.Event,
        deadline: float | None) -> FarmResult:
    """Run the farm on its own thread while the main thread monitors stops."""
    result: FarmResult | None = None
    failure: Exception | None = None

    def farm_target() -> None:
        nonlocal result, failure
        try:
            result = run_farm(
                queue, execute, workers=options.workers, stop=stop,
                deadline=deadline)
        except Exception as error:  # Deliberate: re-raised on main thread.
            failure = error
        finally:
            stop.set()

    farmer = threading.Thread(
        target=farm_target, name='farm-runner', daemon=True)
    farmer.start()
    reason = _wait_uninterrupted(farmer, stop, deadline, options.stop_file)
    if reason != 'done':
        print(f'farm stopping: {reason}', flush=True)
    if failure is not None:
        raise failure
    if result is None:
        raise RuntimeError('farm thread finished without a result')
    return result


def _ensure_engine() -> None:
    """Fail fast when the bundled Pikafish engine is missing."""
    try:
        find_engine_binary()
    except FileNotFoundError as error:
        raise SystemExit(f'Pikafish engine not found: {error}') from error


def _preflight(options: argparse.Namespace) -> None:
    """Fail fast on a missing engine and clear stale stop state."""
    _ensure_engine()
    if prepare_stop_file(options.stop_file):
        print(f'removed stale stop file: {options.stop_file}', flush=True)
    options.out_root.mkdir(parents=True, exist_ok=True)
    options.log_dir.mkdir(parents=True, exist_ok=True)


def _calibration_options(
        options: argparse.Namespace, workspace: Path) -> argparse.Namespace:
    """Return a copy of ``options`` that writes only inside ``workspace``."""
    calibration = argparse.Namespace(**vars(options))
    calibration.out_root = workspace / 'out'
    calibration.log_dir = workspace / 'logs'
    calibration.stop_file = workspace / 'STOP'
    calibration.log_dir.mkdir(parents=True, exist_ok=True)
    return calibration


def _summarize_depth(
        depth: int, options: argparse.Namespace, root: Path,
) -> CalibrationRow | None:
    """Play ``workers`` one-game shards at ``depth`` and summarize them."""
    queue = JobQueue([DepthTarget(depth, options.workers)], options.out_root,
                     games_per_shard=1)
    stats = FarmStats()
    measurements: list[tuple[float, int]] = []

    def run_timed(job: ShardJob) -> bool:
        started = time.perf_counter()
        if not run_job(job, options, root, stats):
            return False
        measurements.append(
            (time.perf_counter() - started, _count_positions(job.path)))
        return True

    result = _run_to_completion(
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


def _run_calibration(options: argparse.Namespace, root: Path) -> int:
    """Benchmark every requested depth in a throwaway temporary workspace."""
    print(CALIBRATION_HEADER, flush=True)
    with tempfile.TemporaryDirectory(prefix='pikafish_calibration_') as temp:
        calibration = _calibration_options(options, Path(temp))
        for target in options.targets:
            row = _summarize_depth(target.depth, calibration, root)
            if row is None:
                return 1
            print(f'{row.depth:>5}  {row.median_seconds_per_game:>13.1f}  '
                  f'{row.mean_positions_per_game:>19.1f}  '
                  f'{row.est_positions_per_hour:>18,.0f}', flush=True)
    return 0


def _latest_log(log_dir: Path) -> Path | None:
    """Return the most recently modified worker log, if any."""
    logs = sorted(log_dir.glob('*.log'), key=lambda path: path.stat().st_mtime)
    return logs[-1] if logs else None


def _report_result(result: FarmResult,
                   options: argparse.Namespace) -> int:
    """Print the farm outcome and map it to a process exit code."""
    if not result.aborted:
        print(f'farm finished: {result.completed} shard(s) written, '
              f'{result.failed} failed attempt(s)', flush=True)
        return 0
    latest = _latest_log(options.log_dir)
    location = f'check log: {latest}' if latest else 'no worker log found'
    print(f'farm aborted after {result.failed} failed attempt(s); '
          f'{location}', flush=True)
    return 1


def main(argv: list[str] | None = None) -> int:
    """Run the shard farm or a calibration run and return an exit code."""
    options = _parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    if options.calibrate:
        _ensure_engine()
        return _run_calibration(options, root)
    _preflight(options)
    queue = JobQueue(options.targets, options.out_root,
                     games_per_shard=options.games_per_shard)
    print_plan(queue, options.targets)
    stats = FarmStats()
    stop = threading.Event()
    deadline = (None if options.hours is None
                else time.monotonic() + options.hours * 3600.0)
    started = time.monotonic()
    threading.Thread(
        target=status_loop,
        args=(stop, options.status_seconds, queue, stats, options.targets,
              started),
        name='farm-status', daemon=True).start()

    def execute(job: ShardJob) -> bool:
        return run_job(job, options, root, stats)

    result = _run_to_completion(queue, execute, options, stop, deadline)
    return _report_result(result, options)


if __name__ == '__main__':
    raise SystemExit(main())
