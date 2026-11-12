"""Tests for the thread-safe shard job queue."""

from pathlib import Path
import threading

from src.training.job_queue import DEFAULT_GAMES_PER_SHARD, JobQueue, ShardJob
from src.training.shard_plan import (
    DepthProgress,
    DepthTarget,
    depth_dir,
    game_id_base,
    load_progress,
    shard_filename,
)


def _make_shards(directory: Path, depth: int, sizes: list[int]) -> None:
    """Create completed shard files of the given game counts in sequence."""
    start = game_id_base(depth)
    for games in sizes:
        (directory / shard_filename(depth, start, games)).touch()
        start += games


def test_claim_order_and_ids(tmp_path):
    queue = JobQueue(
        [DepthTarget(3, 40), DepthTarget(4, 20)],
        tmp_path,
        games_per_shard={3: 20, 4: 20},
    )

    assert queue.claim() == ShardJob(
        3, 30_000_000, 20, tmp_path / 'depth3' / 'd3_g30000000_n20.npz')
    assert queue.claim() == ShardJob(
        3, 30_000_020, 20, tmp_path / 'depth3' / 'd3_g30000020_n20.npz')
    assert queue.claim() == ShardJob(
        4, 40_000_000, 20, tmp_path / 'depth4' / 'd4_g40000000_n20.npz')
    assert queue.claim() is None


def test_last_shard_shrinks(tmp_path):
    queue = JobQueue([DepthTarget(3, 30)], tmp_path, games_per_shard=20)

    first = queue.claim()
    assert (first.game_id_start, first.games) == (30_000_000, 20)
    last = queue.claim()
    assert (last.game_id_start, last.games) == (30_000_020, 10)
    assert last.path == tmp_path / 'depth3' / 'd3_g30000020_n10.npz'
    assert queue.claim() is None


def test_resume_skips_finished_depth(tmp_path):
    directory = depth_dir(tmp_path, 3)
    directory.mkdir(parents=True)
    _make_shards(directory, 3, [20, 20])

    queue = JobQueue(
        [DepthTarget(3, 40), DepthTarget(4, 20)],
        tmp_path,
        games_per_shard={3: 20, 4: 20},
    )

    assert queue.progress()[3] == DepthProgress(30_000_040, 40)
    job = queue.claim()
    assert job.depth == 4
    assert (job.game_id_start, job.games) == (40_000_000, 20)
    assert queue.claim() is None


def test_target_below_done_gives_no_jobs(tmp_path):
    directory = depth_dir(tmp_path, 3)
    directory.mkdir(parents=True)
    _make_shards(directory, 3, [20, 20])

    queue = JobQueue([DepthTarget(3, 25)], tmp_path, games_per_shard=20)

    assert queue.claim() is None


def test_inf_target_never_exhausts(tmp_path):
    queue = JobQueue([DepthTarget(9, None)], tmp_path)

    starts = []
    for _ in range(1000):
        job = queue.claim()
        assert job is not None
        assert job.games == 6
        starts.append(job.game_id_start)

    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts)
    assert DEFAULT_GAMES_PER_SHARD[9] == 6


def test_fail_requeues_then_drops(tmp_path):
    queue = JobQueue([DepthTarget(3, 40)], tmp_path, games_per_shard=20)
    job = queue.claim()

    assert queue.fail(job) is True
    retry = queue.claim()
    assert retry.path == job.path
    assert (retry.game_id_start, retry.games, retry.attempt) == (
        30_000_000, 20, 1)

    assert queue.fail(retry) is True
    retry_again = queue.claim()
    assert (retry_again.game_id_start, retry_again.attempt) == (
        30_000_000, 2)

    assert queue.fail(retry_again) is False
    following = queue.claim()
    assert following is not None
    assert following.game_id_start == 30_000_020
    assert following.attempt == 0


def test_ids_survive_changed_shard_size(tmp_path):
    first_queue = JobQueue(
        [DepthTarget(3, 100)], tmp_path, games_per_shard=20)
    first = first_queue.claim()
    second = first_queue.claim()
    assert (first.game_id_start, second.game_id_start) == (
        30_000_000, 30_000_020)

    fresh_queue = JobQueue(
        [DepthTarget(3, 100)], tmp_path, games_per_shard=8)
    job = fresh_queue.claim()
    assert job.game_id_start == 30_000_040
    assert job.games == 8


def test_concurrent_claims_disjoint(tmp_path):
    queue = JobQueue([DepthTarget(9, None)], tmp_path, games_per_shard=5)
    claimed: list[ShardJob] = []
    errors: list[Exception] = []
    claimed_lock = threading.Lock()
    start_barrier = threading.Barrier(8)

    def worker():
        try:
            start_barrier.wait()
            for _ in range(25):
                job = queue.claim()
                if job is None:
                    raise AssertionError('infinite target ran out of jobs')
                with claimed_lock:
                    claimed.append(job)
        except Exception as error:  # Forwarded to the main thread assertion.
            errors.append(error)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors, errors
    assert len(claimed) == 200
    intervals = sorted(
        (job.game_id_start, job.game_id_start + job.games)
        for job in claimed)
    assert intervals[0][0] == 90_000_000
    assert intervals[-1][1] == 90_001_000
    for (_, previous_end), (current_start, _) in zip(
            intervals, intervals[1:]):
        assert previous_end == current_start


def test_progress_persisted_on_claim_and_complete(tmp_path):
    queue = JobQueue([DepthTarget(3, 100)], tmp_path, games_per_shard=20)
    job = queue.claim()
    directory = depth_dir(tmp_path, 3)

    assert load_progress(directory, 3).next_game_id == (
        job.game_id_start + job.games)
    assert queue.progress()[3] == DepthProgress(30_000_020, 0)

    job.path.touch()
    queue.complete(job)
    assert load_progress(directory, 3).games_done == job.games
    assert queue.progress()[3] == DepthProgress(30_000_020, 20)

    job.path.unlink()
    fresh_queue = JobQueue(
        [DepthTarget(3, 100)], tmp_path, games_per_shard=20)
    assert fresh_queue.progress()[3] == DepthProgress(30_000_020, 20)
    assert fresh_queue.claim().game_id_start == 30_000_020
