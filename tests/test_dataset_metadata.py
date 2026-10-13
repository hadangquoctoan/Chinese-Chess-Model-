"""Reject corrupted game boundaries and contradictory winner targets."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.utils.dataset_saver import DatasetSaver


def one_state():
    return np.zeros((19, 10, 9), dtype=np.float32)


def one_policy():
    return np.full(1800, 1 / 1800, dtype=np.float32)


def test_unsigned_descending_offsets_are_rejected(tmp_path):
    saver = DatasetSaver(str(tmp_path))
    saver.save_games([([one_state()] * 3, [one_policy()] * 3, 1.0)], "games.npz")
    with np.load(tmp_path / "games.npz", allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["offsets"] = np.array([0, 2, 1, 3], dtype=np.uint64)
    payload["winners"] = np.array([1, 1, 1], dtype=np.float32)
    np.savez(tmp_path / "bad.npz", **payload)
    with pytest.raises(ValueError, match="boundaries"):
        saver.load_games("bad.npz")


def test_winner_and_saved_value_mismatch_rejected(tmp_path):
    saver = DatasetSaver(str(tmp_path))
    saver.save_games([([one_state()], [one_policy()], 1.0)], "games.npz")
    with np.load(tmp_path / "games.npz", allow_pickle=False) as source:
        payload = {key: source[key] for key in source.files}
    payload["winners"] = np.array([-1], dtype=np.float32)
    np.savez(tmp_path / "bad.npz", **payload)
    with pytest.raises(ValueError, match="winner"):
        saver.load_games("bad.npz")
