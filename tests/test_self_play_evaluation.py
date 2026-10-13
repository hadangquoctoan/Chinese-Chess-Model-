"""Actual short-game checks for exploration and evaluation contracts."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet
from src.training.evaluator import Evaluator
from src.training.self_play import SelfPlayWorker


def tiny_model():
    return ChineseChessNet(ModelConfig(num_channels=4, num_residual_blocks=0,
                                      policy_channels=1, value_channels=1))


def test_evaluation_is_deterministic_and_preserves_rng_and_modes():
    model = tiny_model()
    evaluator = Evaluator(mcts_simulations=2, max_game_plies=2)
    np.random.seed(512)
    before = np.random.get_state()
    evaluator.play_game(model, model)
    after = np.random.get_state()
    assert np.array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert model.training


def test_draws_count_half_a_point_and_evaluate_restores_mode(monkeypatch):
    model = tiny_model()
    evaluator = Evaluator(mcts_simulations=1)
    monkeypatch.setattr(evaluator, "play_game", lambda *a, **k: 0.0)
    score, wins, losses, draws = evaluator.evaluate(model, model, 2, verbose=False)
    assert (score, wins, losses, draws) == (0.5, 0, 0, 2)
    assert model.training


def test_evaluate_restores_mode_after_failure(monkeypatch):
    model1, model2 = tiny_model(), tiny_model()
    model2.eval()
    evaluator = Evaluator(mcts_simulations=1)
    def fail(*args, **kwargs):
        raise RuntimeError("failed evaluation")
    monkeypatch.setattr(evaluator, "play_game", fail)
    with pytest.raises(RuntimeError, match="failed evaluation"):
        evaluator.evaluate(model1, model2, 2, verbose=False)
    assert model1.training
    assert not model2.training


def test_self_play_explicit_move_limit_generates_normalized_training_data():
    worker = SelfPlayWorker(tiny_model(), mcts_simulations=2,
                            max_game_plies=2, no_capture_limit=100,
                            dirichlet_alpha=0.2, dirichlet_epsilon=0.0)
    states, policies, winner = worker.play_game(verbose=True)
    assert len(states) == len(policies) == 2
    assert winner == 0.0
    assert worker.last_termination_reason == "Draw - Maximum game plies reached"
    assert all(state.shape == (19, 10, 9) for state in states)
    assert all(np.isclose(policy.sum(), 1.0) for policy in policies)


def test_evaluator_uses_same_explicit_move_limit():
    model = tiny_model()
    evaluator = Evaluator(mcts_simulations=2, max_game_plies=2, no_capture_limit=100)
    assert evaluator.play_game(model, model, verbose=True) == 0.0
    assert evaluator.last_termination_reason == "Draw - Maximum game plies reached"


def test_evaluation_rejects_odd_color_biased_budget():
    model = tiny_model()
    evaluator = Evaluator(mcts_simulations=1)
    with pytest.raises(ValueError, match="even"):
        evaluator.evaluate(model, model, num_games=3, verbose=False)


def test_model_predict_rejects_nonfinite_input():
    with pytest.raises(ValueError, match="finite"):
        tiny_model().predict(torch.full((19, 10, 9), float("nan")))


def test_evaluation_uses_same_opening_with_swapped_colors(monkeypatch):
    model1, model2 = tiny_model(), tiny_model()
    evaluator = Evaluator(mcts_simulations=1, max_game_plies=8,
                          opening_plies=2, opening_seed=93)
    played = []
    def record(red, black, verbose=False, opening_moves=()):
        played.append((red, black, opening_moves))
        return 0.0
    monkeypatch.setattr(evaluator, "play_game", record)
    evaluator.evaluate(model1, model2, num_games=4, verbose=False)
    assert len(played) == 4
    for first, second in ((played[0], played[1]), (played[2], played[3])):
        assert first[0] is second[1] and first[1] is second[0]
        assert len(first[2]) == 2
        assert first[2] == second[2]

