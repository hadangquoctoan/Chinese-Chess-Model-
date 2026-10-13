"""Regression tests for legal policy masking and search boundaries."""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.game.game_state import GameState
from src.game.rules import Rules
from src.game.pieces import PieceType, create_piece
from src.model.mcts import MCTS, MCTSNode
from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet


class FixedNet(ChineseChessNet):
    """A deterministic real module with configurable logits."""

    def __init__(self):
        super().__init__(ModelConfig(num_channels=4, num_residual_blocks=0,
                                     policy_channels=1, value_channels=1))
        self.calls = 0
        self.logits = torch.zeros(1800)

    def forward(self, x):
        self.calls += 1
        return self.logits.to(x.device).unsqueeze(0), x.new_zeros((1, 1))


def test_terminal_root_does_not_infer_or_return_a_move():
    game = GameState()
    for _ in range(2):
        for move in ((0, 0, 1, 0), (9, 0, 8, 0),
                     (1, 0, 0, 0), (8, 0, 9, 0)):
            assert game.make_move(*move)
    assert game.is_terminal()
    model = FixedNet()
    move, policy = MCTS(model, num_simulations=2).search(game)
    assert move is None
    assert np.count_nonzero(policy) == 0
    assert model.calls == 0
    assert model.training


def test_legal_logits_are_normalized_before_underflow():
    game = GameState()
    moves = game.get_legal_moves()
    model = FixedNet()
    model.logits.fill_(0.0)
    for move in moves:
        model.logits[Rules.move_to_action_index(*move, board=game.board)] = -1000.0
    first, second = moves[:2]
    model.logits[Rules.move_to_action_index(*first, board=game.board)] = -110.0
    model.logits[Rules.move_to_action_index(*second, board=game.board)] = -111.0
    root = MCTSNode(game)
    MCTS(model, num_simulations=1)._expand_node(root)
    assert root.children[first].prior == pytest.approx(0.73105858)
    assert root.children[second].prior == pytest.approx(0.26894142)


def test_search_restores_model_mode_when_inference_fails():
    class FailingNet(FixedNet):
        def forward(self, x):
            raise RuntimeError("inference failure")

    model = FailingNet()
    with pytest.raises(RuntimeError, match="inference failure"):
        MCTS(model, num_simulations=1).search(GameState())
    assert model.training


@pytest.mark.parametrize("temperature", [-1.0, float("nan"), float("inf")])
def test_temporary_temperature_rejects_invalid_values(temperature):
    search = MCTS(FixedNet(), num_simulations=1)
    with pytest.raises(ValueError, match="temperature"):
        search.get_action_probs(GameState(), temperature)
    assert search.temperature == 1.0


def test_small_temperature_keeps_relative_visit_weights_without_overflow():
    root = MCTSNode(GameState())
    for move, visits in (((3, 0, 4, 0), 10000), ((3, 2, 4, 2), 9999)):
        child = MCTSNode(GameState(), parent=root, move=move)
        child.visit_count = visits
        root.children[move] = child
    np.random.seed(123)
    selected = [root.select_action(temperature=0.001) for _ in range(100)]
    assert len(set(selected)) == 2


def test_predict_rejects_batch_in_single_position_api():
    model = FixedNet()
    with pytest.raises(ValueError, match="single"):
        model.predict(torch.zeros(2, 19, 10, 9))


def test_policy_target_keeps_formerly_aliased_visits_separate():
    game = GameState()
    root = MCTSNode(game)
    first, second = (2, 1, 2, 4), (2, 1, 9, 1)
    for move, visits in ((first, 90), (second, 10)):
        child = MCTSNode(game.copy(), parent=root, move=move)
        child.visit_count = visits
        root.children[move] = child
    policy = root.get_policy_target()
    assert policy[Rules.move_to_action_index(*first, board=game.board)] == pytest.approx(0.9)
    assert policy[Rules.move_to_action_index(*second, board=game.board)] == pytest.approx(0.1)


def test_real_search_selects_a_forced_win_and_keeps_input_unchanged():
    game = GameState()
    for row in range(10):
        for col in range(9):
            game.board.set_piece(row, col, create_piece(PieceType.EMPTY, True))
    for row, col, kind, red in ((0, 4, PieceType.GENERAL, True),
                                (9, 4, PieceType.GENERAL, False),
                                (5, 4, PieceType.SOLDIER, True),
                                (8, 0, PieceType.CHARIOT, True),
                                (7, 3, PieceType.CHARIOT, True),
                                (7, 5, PieceType.CHARIOT, True)):
        game.board.set_piece(row, col, create_piece(kind, red))
    game.reset_history()
    initial = game.to_tensor().copy()
    search = MCTS(FixedNet(), num_simulations=80, temperature=0,
                  c_puct=0.2, dirichlet_epsilon=0)
    move, policy = search.search(game)
    assert np.array_equal(game.to_tensor(), initial)
    assert game.move_count == 0
    assert policy.sum() == pytest.approx(1.0)
    assert game.make_move(*move)
    assert game.is_terminal()
    assert game.get_winner() is True
    assert game.get_game_result() == -1.0


@pytest.mark.parametrize("red", [True, False])
def test_search_backs_up_actual_checkmate_for_both_colors(red):
    game = GameState()
    for row in range(10):
        for col in range(9):
            game.board.set_piece(row, col, create_piece(PieceType.EMPTY, True))
    position = ((0, 4, PieceType.GENERAL, True),
                (9, 4, PieceType.GENERAL, False),
                (5, 4, PieceType.SOLDIER, True),
                (8, 0, PieceType.CHARIOT, True),
                (7, 3, PieceType.CHARIOT, True),
                (7, 5, PieceType.CHARIOT, True),
                (7, 4, PieceType.SOLDIER, True))
    for row, col, kind, color in position:
        game.board.set_piece(row if red else 9 - row, col,
                             create_piece(kind, color if red else not color))
    game.is_red_turn = red
    game.reset_history()
    move = (8, 0, 8, 4) if red else (1, 0, 1, 4)
    model = FixedNet()
    model.logits.fill_(-100.0)
    index = Rules.move_to_action_index(*move, board=game.board)
    model.logits[index] = 100.0
    before = game.to_tensor().copy()
    selected, policy = MCTS(model, num_simulations=2, temperature=0,
                            dirichlet_epsilon=0).search(game)
    assert selected == move
    np.testing.assert_array_equal(game.to_tensor(), before)
    assert game.make_move(*selected)
    assert Rules.is_in_check(game.board, game.is_red_turn)
    assert game.get_termination_reason().startswith("Checkmate")
    assert game.get_winner() is red
    assert game.get_game_result() == -1.0
    assert policy[index] == pytest.approx(1.0)


@pytest.mark.parametrize("options", [
    {"num_simulations": 1.2}, {"num_simulations": True},
    {"c_puct": float("nan")}, {"c_puct": float("inf")},
    {"dirichlet_alpha": float("nan")}, {"dirichlet_alpha": float("inf")},
])
def test_search_rejects_invalid_search_parameters(options):
    with pytest.raises(ValueError):
        MCTS(FixedNet(), **options)

