"""Regression checks for lazy search, cached rule data and inference lifetime."""
import numpy as np
import pytest
import torch

from src.game.game_state import GameState
from src.game.rules import Rules
from src.model.mcts import MCTS, MCTSNode
from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet


class ProbeNet(ChineseChessNet):
    """A real small network that records the execution context of each forward."""

    def __init__(self):
        super().__init__(ModelConfig(num_channels=4, num_residual_blocks=0,
                                     policy_channels=1, value_channels=1))
        self.contexts = []
        self.device_calls = 0

    def get_device(self):
        self.device_calls += 1
        return super().get_device()

    def forward(self, x):
        self.contexts.append((torch.is_inference_mode_enabled(),
                              torch.is_grad_enabled(), self.training))
        return super().forward(x)


class EagerMCTS(MCTS):
    """Reference child expansion using independent copies and validated moves."""

    def _create_child_nodes(self, parent, move_priors):
        for move, prior in move_priors:
            state = parent.game_state.copy()
            assert state.make_move(*move)
            parent.children[move] = MCTSNode(state, parent, move, prior)


def test_expansion_and_policy_do_not_copy_or_apply_unused_moves(monkeypatch):
    root = MCTSNode(GameState())
    calls = []
    original_copy = GameState.copy

    def counted_copy(state):
        calls.append(state)
        return original_copy(state)

    monkeypatch.setattr(GameState, 'copy', counted_copy)
    search = MCTS(ProbeNet(), num_simulations=1, dirichlet_epsilon=0)
    search._expand_node(root, policy=np.zeros(1800))
    assert len(root.children) == len(root.legal_moves)
    assert all(child._game_state is None for child in root.children.values())
    assert not calls
    selected = root.best_child(search.c_puct)
    selected.visit_count = 1
    assert root.get_policy_target().sum() == pytest.approx(1)
    assert root.select_action(0) == selected.move
    assert not calls
    state = selected.game_state
    assert selected.game_state is state
    assert len(calls) == 1
    assert state.move_count == 1 and root.game_state.move_count == 0
    assert sum(child._game_state is not None for child in root.children.values()) == 1


def test_lazy_sibling_states_and_captured_pieces_are_independent():
    game = GameState()
    root = MCTSNode(game)
    MCTS(ProbeNet(), num_simulations=1)._expand_node(root, policy=np.zeros(1800))
    capture = root.children[(2, 1, 9, 1)].game_state
    other = root.children[(3, 0, 4, 0)].game_state
    before = other.to_tensor().copy()
    record = capture._move_history.get_last_move()
    assert record.captured_piece is not game.board.get_piece(9, 1)
    record.captured_piece.is_red = True
    capture.board.get_piece(0, 0).is_red = False
    assert game.board.get_piece(0, 0).is_red
    assert not game.board.get_piece(9, 1).is_red
    np.testing.assert_array_equal(other.to_tensor(), before)


def test_search_copies_once_per_visited_leaf_and_analyzes_each_state_once(monkeypatch):
    copies, analyzed = [], []
    original_copy, original_info = GameState.copy, GameState.get_search_info

    def counted_copy(state):
        copies.append(state)
        return original_copy(state)

    def counted_info(state):
        analyzed.append(state)
        return original_info(state)

    def unexpected(*args, **kwargs):
        raise AssertionError('search repeated a terminal availability query')

    monkeypatch.setattr(GameState, 'copy', counted_copy)
    monkeypatch.setattr(GameState, 'get_search_info', counted_info)
    monkeypatch.setattr(GameState, 'is_terminal', unexpected)
    monkeypatch.setattr(Rules, 'has_legal_move', unexpected)
    simulations = 16
    model = ProbeNet()
    move, policy = MCTS(model, num_simulations=simulations,
                        dirichlet_epsilon=0).search(GameState())
    assert move is not None and policy.sum() == pytest.approx(1)
    assert len(copies) == simulations
    assert len(analyzed) == len(copies) + 1
    assert len({id(state) for state in analyzed}) == len(analyzed)
    assert model.device_calls == 1


def test_terminal_leaf_value_is_cached_without_inference(monkeypatch):
    game = GameState(max_plies=1)
    assert game.make_move(3, 0, 4, 0)
    root = MCTSNode(game)
    calls = []
    original_info = GameState.get_search_info

    def counted_info(state):
        calls.append(state)
        return original_info(state)

    monkeypatch.setattr(GameState, 'get_search_info', counted_info)
    model = ProbeNet()
    search = MCTS(model, num_simulations=1)
    for _ in range(5):
        assert search._expand_and_evaluate(root) == 0
        search._expand_node(root)
        assert search._select_path(root) == [root]
        assert root.legal_moves == []
    assert len(calls) == 1
    assert not model.contexts and not model.device_calls


@pytest.mark.parametrize('opening', [(), ((2, 1, 9, 1), (6, 0, 5, 0))])
@pytest.mark.parametrize('noise', [0, 0.25])
def test_lazy_search_matches_eager_move_policy_and_preserves_input(opening, noise):
    torch.manual_seed(19)
    model = ProbeNet()
    game = GameState()
    for move in opening:
        assert game.make_move(*move)
    before = game.to_tensor().copy()
    history = game.position_history
    options = dict(num_simulations=16, temperature=0, dirichlet_epsilon=noise)
    np.random.seed(123)
    expected_move, expected_policy = EagerMCTS(model, **options).search(game)
    np.random.seed(123)
    move, policy = MCTS(model, **options).search(game)
    assert move == expected_move
    np.testing.assert_array_equal(policy, expected_policy)
    np.testing.assert_array_equal(game.to_tensor(), before)
    assert game.position_history == history
    assert game.move_count == len(opening)


@pytest.mark.parametrize('training', [True, False])
def test_search_inference_context_is_scoped_and_model_remains_trainable(training):
    model = ProbeNet()
    model.train(training)
    assert not torch.is_inference_mode_enabled()
    search = MCTS(model, num_simulations=2, dirichlet_epsilon=0)
    search.search(GameState())
    assert model.contexts == [(True, False, False)] * 3
    assert model.training is training
    assert search._inference_device is None
    assert not torch.is_inference_mode_enabled() and torch.is_grad_enabled()
    model.train()
    logits, values = model(torch.zeros(2, 19, 10, 9))
    (logits.square().mean() + values.square().mean()).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_inference_and_temporary_temperature_restore_after_failure(monkeypatch):
    model = ProbeNet()
    search = MCTS(model, num_simulations=1)

    def fail(x):
        assert torch.is_inference_mode_enabled()
        raise RuntimeError('deliberate forward failure')

    with monkeypatch.context() as patch:
        patch.setattr(model, 'forward', fail)
        with pytest.raises(RuntimeError, match='deliberate'):
            search.get_action_probs(GameState(), temperature=0)
    assert model.training and search.temperature == 1
    assert search._inference_device is None
    assert not torch.is_inference_mode_enabled()
    move, policy = search.search(GameState())
    assert move is not None and policy.sum() == pytest.approx(1)


def test_direct_expansion_of_a_real_network_still_disables_autograd():
    model = ProbeNet()
    root = MCTSNode(GameState())
    MCTS(model, num_simulations=1)._expand_node(root)
    assert root.is_expanded and root.children
    assert model.contexts[0][:2] == (True, False)
    assert not torch.is_inference_mode_enabled()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA is unavailable')
def test_device_is_refreshed_between_searches_when_model_moves_to_cuda():
    model = ProbeNet()
    search = MCTS(model, num_simulations=2, temperature=0, dirichlet_epsilon=0)
    search.search(GameState())
    model.cuda()
    move, policy = search.search(GameState())
    assert move in GameState().get_legal_moves()
    assert policy.sum() == pytest.approx(1)
    assert model.device_calls == 2
    assert search._inference_device is None
