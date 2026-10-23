import numpy as np

from scripts.build_teacher_labels import (
    MAX_TACTICAL_PROBES, _append_probe, _capture_value, load_probe_npz,
    save_probes_npz,
)
from src.game.game_state import GameState


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
