"""Tests for Pikafish farm calibration summaries and runner."""

import argparse
from collections.abc import Callable
from pathlib import Path

import pytest

from src.training.farm_calibration import (
    CalibrationRow,
    CalibrationTools,
    run_calibration,
    summarize_calibration,
)
from src.training.job_queue import JobQueue, ShardJob
from src.training.shard_farm import FarmResult, run_farm
from src.training.shard_plan import DepthTarget


def test_summarize_calibration():
    assert summarize_calibration(
        7, [60.0, 120.0, 90.0], [100, 140, 120], 54) == CalibrationRow(
            7, 90.0, 120.0, 259_200.0)


@pytest.mark.parametrize('seconds, positions, workers', [
    ([], [], 1),
    ([10.0], [], 1),
    ([10.0], [5], 0),
    ([0.0], [5], 1),
])
def test_summarize_calibration_rejects_bad_input(
        seconds: list[float], positions: list[int], workers: int):
    with pytest.raises(ValueError):
        summarize_calibration(3, seconds, positions, workers)


def _options(tmp_path: Path, workers: int = 2) -> argparse.Namespace:
    """Return CLI-style options for a calibration run in ``tmp_path``."""
    return argparse.Namespace(
        targets=[DepthTarget(3, 5), DepthTarget(4, None)],
        workers=workers,
        out_root=tmp_path / 'farm',
        log_dir=tmp_path / 'logs',
        stop_file=tmp_path / 'farm' / 'STOP',
    )


def _tools(
        seen: list[tuple[int, ShardJob, argparse.Namespace]],
        succeed: bool = True,
) -> CalibrationTools:
    """Return fake collaborators that record every claimed shard."""

    def run_job(job: ShardJob, options: argparse.Namespace) -> bool:
        seen.append((job.depth, job, options))
        if not succeed:
            return False
        job.path.parent.mkdir(parents=True, exist_ok=True)
        job.path.write_bytes(b'shard')
        return True

    def run_to_completion(
            queue: JobQueue, execute: Callable[[ShardJob], bool],
            options: argparse.Namespace, stop,
            deadline: float | None) -> FarmResult:
        return run_farm(
            queue, execute, workers=options.workers, stop=stop,
            deadline=deadline)

    return CalibrationTools(
        run_job=run_job,
        run_to_completion=run_to_completion,
        count_positions=lambda path: 10,
    )


def test_run_calibration_plays_workers_games_per_depth(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    options = _options(tmp_path)
    seen: list[tuple[int, ShardJob, argparse.Namespace]] = []

    assert run_calibration(options, _tools(seen)) == 0

    assert [depth for depth, _, _ in seen] == [3, 3, 4, 4]
    assert all(job.games == 1 for _, job, _ in seen)
    workspaces = {job.path.parents[2] for _, job, _ in seen}
    assert len(workspaces) == 1
    workspace = workspaces.pop()
    assert workspace.name.startswith('pikafish_calibration_')
    assert not workspace.exists()
    for _, job, job_options in seen:
        assert workspace in job.path.parents
        assert job_options.out_root == workspace / 'out'
        assert job_options.log_dir == workspace / 'logs'
        assert job_options.stop_file == workspace / 'STOP'
    assert not options.out_root.exists()
    assert not options.log_dir.exists()

    lines = capsys.readouterr().out.splitlines()
    assert lines[0] == (
        'depth  median_s/game  mean_positions/game  est_positions/hour')
    assert [line.split()[0] for line in lines[1:]] == ['3', '4']


def test_run_calibration_stops_when_depth_cannot_be_measured(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    options = _options(tmp_path)
    seen: list[tuple[int, ShardJob, argparse.Namespace]] = []

    assert run_calibration(options, _tools(seen, succeed=False)) == 1

    assert {depth for depth, _, _ in seen} == {3}
    assert 'depth 3: calibration failed' in capsys.readouterr().out
    assert not options.out_root.exists()
