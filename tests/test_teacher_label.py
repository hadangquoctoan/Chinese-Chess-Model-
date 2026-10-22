"""Tests for src.training.teacher_label.build_teacher_label."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest

from src.engine.coordinates import move_to_uci
from src.game.game_state import GameState
from src.game.rules import Rules
from src.training.teacher_label import SCORE_SCALE, build_teacher_label


def test_opening_cannon_red_value():
    state = GameState()
    policy, value = build_teacher_label(state.board, True, "h2e2", 87)
    assert policy.shape == (1800,)
    assert policy.dtype.name == "float32"
    expected_index = Rules.move_to_action_index(2, 7, 2, 4, state.board)
    assert policy[expected_index] == 1.0
    assert policy.sum() == 1.0
    assert value == pytest.approx(math.tanh(87 / SCORE_SCALE))


def test_black_to_move_flips_sign():
    state = GameState()
    _, value_red = build_teacher_label(state.board, True, "h2e2", 100)
    _, value_black = build_teacher_label(state.board, False, "h2e2", 100)
    assert value_black == pytest.approx(-value_red)


def test_sample_tokens_produce_valid_index():
    state = GameState()
    tokens = [move_to_uci(move) for move in state.get_legal_moves()]
    assert len(tokens) >= 30
    assert any(ch.isdigit() for token in tokens[:30] for ch in token)
    for token in tokens[:30]:
        policy, _ = build_teacher_label(state.board, True, token, 50)
        index = int(policy.argmax())
        assert 0 <= index < 1800
        assert policy.sum() == 1.0
    state.make_move(2, 7, 2, 4)
    policy, _ = build_teacher_label(state.board, False, "a9a8", 50)
    assert 0 <= int(policy.argmax()) < 1800


def test_value_range_extremes():
    state = GameState()
    for score in (100000, -100000, 0):
        _, value = build_teacher_label(state.board, True, "h2e2", score)
        assert -1.0 <= value <= 1.0
    _, zero = build_teacher_label(state.board, True, "h2e2", 0)
    assert zero == pytest.approx(0.0)


def test_pretrain_maps_red_centric_value_to_side_to_move():
    from scripts.pretrain_supervised import side_to_move_value

    red_state = GameState()
    red_tensor = red_state.to_tensor()
    assert red_tensor[16, 0, 0] > 0
    assert side_to_move_value(red_tensor, 0.4) == pytest.approx(0.4)
    assert red_state.make_move(2, 7, 2, 4)
    black_tensor = red_state.to_tensor()
    assert black_tensor[16, 0, 0] < 0
    assert side_to_move_value(black_tensor, 0.4) == pytest.approx(-0.4)


def test_teacher_npz_round_trip(tmp_path):
    from scripts.build_teacher_labels import load_teacher_npz, save_teacher_npz

    state = GameState()
    policy, value = build_teacher_label(state.board, True, "h2e2", 87)
    path = tmp_path / "teacher.npz"
    save_teacher_npz(path, [state.to_tensor()], [policy], [value],
                     depth=2, engine_path=path)
    loaded = load_teacher_npz(path)
    assert loaded["states"].shape == (1, 19, 10, 9)
    assert loaded["policies"].shape == (1, 1800)
    assert loaded["values"].shape == (1,)
    assert loaded["engine_depth"] == 2
    assert loaded["score_scale"] == SCORE_SCALE
    assert loaded["value_convention"] == "red_centric"
    assert np.isclose(loaded["values"][0], value)


def test_invalid_token_raises():
    state = GameState()
    with pytest.raises(ValueError):
        build_teacher_label(state.board, True, "zzz", 50)


if __name__ == "__main__":
    test_opening_cannon_red_value()
    test_black_to_move_flips_sign()
    test_sample_tokens_produce_valid_index()
    test_value_range_extremes()
    test_invalid_token_raises()
    print("teacher_label tests passed")
