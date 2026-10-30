import numpy as np
import pytest

from scripts.build_teacher_labels import (
    MAX_TACTICAL_PROBES, _append_probe, _capture_value, load_probe_npz,
    save_probes_npz, save_teacher_npz,
)
from src.game.game_state import GameState
from src.training.teacher_label import build_teacher_label


def test_capture_value_detects_high_value_target():
    game = GameState()
    assert _capture_value(game.board, (0, 0, 9, 0)) == 9.0
    assert _capture_value(game.board, (3, 0, 4, 0)) == 0.0


def test_probe_list_is_capped():
    probes = []
    probe = (None, (), (0, 0, 0, 0), 4.0)
    for _ in range(MAX_TACTICAL_PROBES + 5):
        _append_probe(probes, probe)
    assert len(probes) == MAX_TACTICAL_PROBES


def test_probe_npz_round_trip(tmp_path):
    state = np.zeros((19, 10, 9), dtype=np.float32)
    probe = (state, ((3, 0, 4, 0),), (0, 0, 9, 0), 9.0)
    path = tmp_path / 'probes.npz'
    save_probes_npz(path, [probe])
    loaded = load_probe_npz(path)
    assert loaded['states'].shape == (1, 19, 10, 9)
    assert loaded['histories'].tolist() == [[3, 0, 4, 0]]
    assert loaded['moves'].tolist() == [[0, 0, 9, 0]]
    assert loaded['captured'].tolist() == [9.0]
    assert loaded['history_groups'] == [[(3, 0, 4, 0)]]


def test_teacher_npz_round_trips_game_ids(tmp_path):
    from scripts.build_teacher_labels import load_teacher_npz

    game = GameState()
    policy, value = build_teacher_label(game.board, True, "h2e2", 87)
    path = tmp_path / "teacher_game_ids.npz"
    save_teacher_npz(path, [game.to_tensor()], [policy], [value],
                     depth=2, engine_path=path, game_ids=[17])
    loaded = load_teacher_npz(path)
    assert loaded['game_ids'].tolist() == [17]


def test_teacher_npz_round_trips_state_limit_metadata(tmp_path):
    from scripts.build_teacher_labels import load_teacher_npz

    game = GameState()
    policy, value = build_teacher_label(game.board, True, "h2e2", 87)
    path = tmp_path / "teacher_limits.npz"
    save_teacher_npz(path, [game.to_tensor()], [policy], [value],
                     depth=8, engine_path=path, game_ids=[3],
                     max_game_plies=24, no_capture_limit=100)
    loaded = load_teacher_npz(path)
    assert loaded['max_game_plies'] == 24
    assert loaded['no_capture_limit'] == 100


def test_teacher_npz_omits_state_limit_metadata_by_default(tmp_path):
    from scripts.build_teacher_labels import load_teacher_npz

    game = GameState()
    policy, value = build_teacher_label(game.board, True, "h2e2", 87)
    path = tmp_path / "teacher_legacy.npz"
    save_teacher_npz(path, [game.to_tensor()], [policy], [value],
                     depth=2, engine_path=path)
    loaded = load_teacher_npz(path)
    assert 'max_game_plies' not in loaded
    assert 'no_capture_limit' not in loaded
    assert 'game_ids' not in loaded


def test_teacher_npz_rejects_mismatched_game_ids(tmp_path):
    game = GameState()
    policy, value = build_teacher_label(game.board, True, "h2e2", 87)
    path = tmp_path / "bad_game_ids.npz"
    with pytest.raises(ValueError, match="game_ids"):
        save_teacher_npz(path, [game.to_tensor()], [policy], [value],
                         depth=2, engine_path=path, game_ids=[1, 2])
