"""Calibration math for the Pikafish shard farm.

A calibration run plays one game per worker at each requested depth under
full worker load, then extrapolates the measured game cost to an expected
positions-per-hour rate.
"""

from dataclasses import dataclass
from statistics import fmean, median

SECONDS_PER_HOUR: float = 3600.0


@dataclass(frozen=True)
class CalibrationRow:
    """Measured game cost and projected throughput for one search depth."""

    depth: int
    median_seconds_per_game: float
    mean_positions_per_game: float
    est_positions_per_hour: float


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
