"""Tests for Pikafish farm calibration summaries."""

import pytest

from src.training.farm_calibration import CalibrationRow, summarize_calibration


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
