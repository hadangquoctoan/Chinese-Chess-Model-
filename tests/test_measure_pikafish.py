"""Unit tests for Pikafish match scoring without a live engine."""
import numpy as np
import pytest

from scripts.measure_vs_pikafish import match_score, play_measured_game


def test_match_score_half_draws():
    assert match_score(15, 5, 20) == 0.875
    assert match_score(10, 0, 20) == 0.5
    assert match_score(0, 20, 20) == 0.5


def test_play_measured_game_with_fakes(monkeypatch):
    class FakeOpponent:
        def choose_move(self, state):
            return state.get_legal_moves()[0]

    class FakeMCTS:
        def __init__(self, model, **kwargs):
            pass

        def get_action_probs(self, game_state, temperature=0.0):
            return game_state.get_legal_moves()[0], np.zeros(1800, dtype=np.float32)

    monkeypatch.setattr("scripts.measure_vs_pikafish.MCTS", FakeMCTS)
    winner, _, plies = play_measured_game(
        object(), FakeOpponent(), model_is_red=True, mcts_simulations=1,
        max_game_plies=4)
    assert winner in (-1.0, 0.0, 1.0)
    assert plies == 4
