"""Multi-worker shard farm runner.

The farm drives a pool of plain :class:`threading.Thread` workers that claim
:class:`ShardJob` instances from a :class:`JobQueue`, execute them through a
caller-supplied ``run_job`` callback, and stop when the queue drains, the
shared stop event is set, the monotonic deadline passes, or too many shards
fail in a row. Plain threads (not a ``ThreadPoolExecutor``) are used so a
stop signal never has to wait for already queued work.
"""

from collections.abc import Callable
from dataclasses import dataclass
import threading
import time

from .job_queue import JobQueue, ShardJob

DEFAULT_MAX_CONSECUTIVE_FAILURES: int = 10


@dataclass(frozen=True)
class FarmResult:
    """Outcome of one farm run.

    ``completed`` counts written shards, ``failed`` counts failed attempts
    including retries, and ``aborted`` reports a stop forced by the
    consecutive-failure budget.
    """

    completed: int
    failed: int
    aborted: bool


class _FarmState:
    """Thread-safe counters and the first unexpected worker exception."""

    def __init__(self, max_consecutive_failures: int) -> None:
        self._lock = threading.Lock()
        self._max_consecutive_failures = max_consecutive_failures
        self._completed = 0
        self._failed = 0
        self._consecutive_failures = 0
        self._aborted = False
        self.error: Exception | None = None

    def record_success(self) -> None:
        """Count one written shard and clear the consecutive-failure streak."""
        with self._lock:
            self._completed += 1
            self._consecutive_failures = 0

    def record_failure(self) -> bool:
        """Count one failed attempt; return True when the budget is spent."""
        with self._lock:
            self._failed += 1
            self._consecutive_failures += 1
            if self._consecutive_failures < self._max_consecutive_failures:
                return False
            self._aborted = True
            return True

    def record_error(self, error: Exception) -> None:
        """Keep the first unexpected worker exception for the main thread."""
        with self._lock:
            if self.error is None:
                self.error = error

    def snapshot(self) -> FarmResult:
        """Return an immutable view of the current counters."""
        with self._lock:
            return FarmResult(self._completed, self._failed, self._aborted)


def run_farm(
        queue: JobQueue,
        run_job: Callable[[ShardJob], bool],
        *,
        workers: int,
        stop: threading.Event,
        deadline: float | None = None,
        max_consecutive_failures: int = DEFAULT_MAX_CONSECUTIVE_FAILURES,
) -> FarmResult:
    """Run ``workers`` threads that claim and execute shards until done.

    ``run_job`` returns ``True`` once its shard is written. A ``False``
    return or an ``OSError`` counts as a failed attempt and the job goes back
    to the queue for a retry. Any other exception stops every worker and is
    re-raised on the calling thread after all workers have joined.
    ``deadline`` is a ``time.monotonic()`` instant after which no new shard
    is claimed.
    """
    _validate_limits(workers, max_consecutive_failures)
    state = _FarmState(max_consecutive_failures)
    # Daemon threads keep an interrupted main thread from waiting on a long
    # shard; every worker is still joined before this function returns.
    threads = [
        threading.Thread(
            target=_worker_loop,
            args=(queue, run_job, stop, deadline, state),
            name=f'shard-farm-{index}',
            daemon=True,
        )
        for index in range(workers)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    if state.error is not None:
        raise state.error
    return state.snapshot()


def _validate_limits(workers: int, max_consecutive_failures: int) -> None:
    """Reject worker and failure budgets that would never make progress."""
    if workers < 1:
        raise ValueError(f'workers must be positive, got {workers}')
    if max_consecutive_failures < 1:
        raise ValueError(
            'max_consecutive_failures must be positive, got '
            f'{max_consecutive_failures}')


def _worker_loop(
        queue: JobQueue,
        run_job: Callable[[ShardJob], bool],
        stop: threading.Event,
        deadline: float | None,
        state: _FarmState,
) -> None:
    """Claim shards until the queue drains or a stop signal arrives."""
    try:
        while not stop.is_set():
            if _deadline_reached(deadline):
                return
            job = queue.claim()
            if job is None:
                return
            _process_job(queue, job, run_job, stop, state)
    except Exception as error:  # Deliberate: re-raised on the main thread.
        state.record_error(error)
        stop.set()


def _process_job(
        queue: JobQueue,
        job: ShardJob,
        run_job: Callable[[ShardJob], bool],
        stop: threading.Event,
        state: _FarmState,
) -> None:
    """Run one claimed shard and report its outcome to the queue."""
    if _run_shard(run_job, job):
        queue.complete(job)
        state.record_success()
        return
    queue.fail(job)
    if state.record_failure():
        stop.set()


def _run_shard(run_job: Callable[[ShardJob], bool], job: ShardJob) -> bool:
    """Return whether ``run_job`` wrote the shard; ``OSError`` means no."""
    try:
        return bool(run_job(job))
    except OSError:
        return False


def _deadline_reached(deadline: float | None) -> bool:
    """Return whether the ``time.monotonic()`` deadline has passed."""
    return deadline is not None and time.monotonic() >= deadline
