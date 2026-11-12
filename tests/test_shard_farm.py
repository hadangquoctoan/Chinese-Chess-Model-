"""Tests for the multi-worker shard farm runner."""

from pathlib import Path
import threading
import time

import pytest

from src.training.job_queue import JobQueue
from src.training.shard_farm import FarmResult, run_farm
from src.training.shard_plan import parse_targets


def _make_queue(
        out_root: Path,
        target: str,
        games_per_shard: int | None = None,
) -> JobQueue:
    """Build a JobQueue from a ``depth:games`` target string."""
    return JobQueue(
        parse_targets(target), out_root, games_per_shard=games_per_shard)


def test_runs_all_jobs(tmp_path):
    queue = _make_queue(tmp_path, '3:6', games_per_shard=2)

    result = run_farm(
        queue, lambda job: True, workers=3, stop=threading.Event())

    assert result == FarmResult(completed=3, failed=0, aborted=False)
    assert queue.claim() is None


def test_stop_event_prevents_new_claims(tmp_path):
    queue = _make_queue(tmp_path, '3:600', games_per_shard=2)
    stop = threading.Event()
    workers = 3

    def run_job(job):
        stop.set()
        return True

    result = run_farm(queue, run_job, workers=workers, stop=stop)

    assert 1 <= result.completed <= workers
    assert result.failed == 0
    assert stop.is_set()


def test_expired_deadline_claims_nothing(tmp_path):
    queue = _make_queue(tmp_path, '3:6', games_per_shard=2)

    result = run_farm(
        queue,
        lambda job: True,
        workers=3,
        stop=threading.Event(),
        deadline=time.monotonic() - 1,
    )

    assert result.completed == 0
    assert result.failed == 0
    assert queue.progress()[3].games_done == 0


def test_failed_job_is_retried(tmp_path):
    queue = _make_queue(tmp_path, '3:6', games_per_shard=2)

    result = run_farm(
        queue,
        lambda job: job.attempt > 0,
        workers=2,
        stop=threading.Event(),
    )

    assert result.completed == 3
    assert result.failed == 3
    assert result.aborted is False


def test_abort_after_consecutive_failures(tmp_path):
    queue = _make_queue(tmp_path, '9:inf')
    stop = threading.Event()
    started = time.monotonic()

    result = run_farm(
        queue,
        lambda job: False,
        workers=3,
        stop=stop,
        max_consecutive_failures=5,
    )

    assert result.aborted is True
    assert result.completed == 0
    assert result.failed >= 5
    assert stop.is_set()
    assert time.monotonic() - started < 5


def test_oserror_counts_as_failure(tmp_path):
    queue = _make_queue(tmp_path, '3:4', games_per_shard=2)

    def run_job(job):
        if job.attempt == 0:
            raise OSError('disk full')
        return True

    result = run_farm(
        queue, run_job, workers=2, stop=threading.Event())

    assert result.completed == 2
    assert result.failed == 2
    assert result.aborted is False


def test_unexpected_exception_propagates(tmp_path):
    queue = _make_queue(tmp_path, '9:inf')
    stop = threading.Event()

    def run_job(job):
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        run_farm(queue, run_job, workers=3, stop=stop)

    assert stop.is_set()
