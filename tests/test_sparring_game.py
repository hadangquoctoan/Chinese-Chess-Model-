"""Sparring game keeps every ply so ReplayBuffer parity stays red-first."""
import numpy as np
import pytest

pytest.importorskip("torch")

from src.game.rules import ACTION_SPACE_SIZE, Rules
from src.training.replay_buffer import ReplayBuffer
from src.training.sparring_game import SparringWorker, play_sparring_game


class FakeOpponent:
    def choose_move(self, state):
        legal = state.get_legal_moves()
        return None if not legal else legal[0]

    def close(self) -> None:
        return None


class FakeMCTS:
    def __init__(self, model, **kwargs):
        self.model = model

    def get_action_probs(self, game_state, temperature=1.0):
        legal = game_state.get_legal_moves()
        if not legal:
            return None, np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        move = legal[0]
        policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
        policy[Rules.move_to_action_index(*move, game_state.board)] = 1.0
        return move, policy


def _play(monkeypatch, opponent_is_red: bool):
    monkeypatch.setattr("src.training.sparring_game.MCTS", FakeMCTS)
    return play_sparring_game(
        object(), FakeOpponent(), opponent_is_red=opponent_is_red,
        mcts_simulations=1, max_game_plies=4, no_capture_limit=8)


def test_fake_engine_game_shapes_and_parity(monkeypatch):
    states, policies, winner = _play(monkeypatch, opponent_is_red=False)
    assert winner in (-1.0, 0.0, 1.0)
    assert len(states) == len(policies) >= 1
    for state, policy in zip(states, policies):
        assert state.shape == (19, 10, 9)
        assert policy.shape == (1800,)
        assert np.isclose(policy.sum(), 1.0, atol=1e-5)
    buffer = ReplayBuffer(32)
    buffer.add_game(states, policies, winner)
    for i, (_, _, value) in enumerate(buffer.buffer):
        assert value == (winner if i % 2 == 0 else -winner)


def test_opponent_red_ply_zero_value_matches_winner(monkeypatch):
    monkeypatch.setattr("src.training.sparring_game.MCTS", FakeMCTS)
    worker = SparringWorker(
        object(), FakeOpponent(), opponent_is_red=True,
        mcts_simulations=1, max_game_plies=4, no_capture_limit=8)
    states, policies, winner = worker.play_game()
    buffer = ReplayBuffer(32)
    buffer.add_game(states, policies, winner)
    assert buffer.buffer[0][2] == winner
    assert np.isclose(policies[0].sum(), 1.0, atol=1e-5)
    assert np.count_nonzero(policies[0]) == 1
