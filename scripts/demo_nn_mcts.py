"""Demonstrate Neural Network and MCTS public APIs without training or saving files."""
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.game.game_state import GameState
from src.model.mcts import MCTS, MCTSNode
from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet


MODEL_CONFIG = ModelConfig.from_size('large')
MCTS_CONFIG = {
    'num_simulations': 16,
    'c_puct': 1.5,
    'temperature': 0.0,
    'dirichlet_alpha': 0.3,
    'dirichlet_epsilon': 0.0,
}


def demo_network(model: ChineseChessNet, game: GameState) -> None:
    """Call forward, predict, get_device, count_parameters, and model display."""
    print('\n[1] Neural Network')
    print(model)
    print(f'get_device(): {model.get_device()}')
    print(f'count_parameters(): {model.count_parameters():,}')
    state = game.to_tensor()
    batch = torch.from_numpy(state).unsqueeze(0).to(model.get_device())
    with torch.no_grad():
        logits, value = model(batch)
    assert logits.shape == (1, 1800) and value.shape == (1, 1)
    assert torch.isfinite(logits).all() and torch.isfinite(value).all()
    assert -1.0 <= value.item() <= 1.0
    print(f'forward(): policy logits {tuple(logits.shape)}, value {value.item():.4f}')
    for name, board_tensor in [('numpy', state), ('torch', batch)]:
        policy, prediction = model.predict(board_tensor)
        assert policy.shape == (1800,) and np.isfinite(policy).all()
        assert (policy >= 0).all() and np.isclose(policy.sum(), 1.0)
        assert np.isfinite(prediction) and -1.0 <= prediction <= 1.0
        print(f'predict({name}): policy sum={policy.sum():.4f}, value={prediction:.4f}')
    print('forward() also calls every ResidualBlock, PolicyHead, and ValueHead.')
    print('predict() returns unmasked probabilities; MCTS masks illegal moves.')


def demo_mcts(model: ChineseChessNet, game: GameState) -> None:
    """Run both search interfaces and verify legal moves and policy targets."""
    print('\n[2] MCTS')
    search = MCTS(model, **MCTS_CONFIG)
    before = game.to_tensor().copy()
    legal_moves = game.get_legal_moves()
    move, policy = search.search(game)
    assert move in legal_moves
    assert policy.shape == (1800,) and np.isfinite(policy).all()
    assert (policy >= 0).all() and np.isclose(policy.sum(), 1.0)
    print(f'search(): move={move}, visit-policy sum={policy.sum():.4f}')
    move, policy = search.get_action_probs(game, temperature=1.0)
    assert move in legal_moves
    assert policy.shape == (1800,) and np.isfinite(policy).all()
    assert (policy >= 0).all() and np.isclose(policy.sum(), 1.0)
    assert search.temperature == MCTS_CONFIG['temperature']
    assert np.array_equal(game.to_tensor(), before)
    assert game.get_legal_moves() == legal_moves
    print(f'get_action_probs(temperature=1.0): move={move}, policy sum={policy.sum():.4f}')
    print('Both calls run a new search; the temperature override is temporary.')
    print('The policy target is normalized visits, not temperature-scaled sampling weights.')


def demo_nodes(game: GameState) -> None:
    """Call node APIs with illustrative statistics, separate from the real search."""
    print('\n[3] MCTSNode (illustrative statistics, not the searches above)')
    root = MCTSNode(game.copy())
    for move, prior, visits, total_value in zip(
        game.get_legal_moves()[:2], (0.6, 0.4), (3, 1), (-1.5, 0.25)
    ):
        state = game.copy()
        moved = state.make_move(*move)
        assert moved
        child = MCTSNode(state, parent=root, move=move, prior=prior)
        child.visit_count = visits
        child.total_value = total_value
        root.children[move] = child
    root.visit_count = sum(child.visit_count for child in root.children.values())
    root.is_expanded = True
    for move, child in root.children.items():
        q_value = child.Q()
        u_value = child.U(MCTS_CONFIG['c_puct'], root.visit_count)
        print(f'move={move}: Q={q_value:.4f}, U={u_value:.4f}, -Q+U={-q_value + u_value:.4f}')
    print(f'best_child(): {root.best_child(MCTS_CONFIG["c_puct"]).move}')
    print(f'select_action(temperature=0): {root.select_action(temperature=0.0)}')
    target = root.get_policy_target()
    assert target.shape == (1800,) and np.isclose(target.sum(), 1.0)
    print(f'get_policy_target(): shape={target.shape}, sum={target.sum():.4f}')


def main() -> None:
    """Demonstrate inference and search on the initial position with random weights."""
    np.random.seed(0)
    torch.manual_seed(0)
    MODEL_CONFIG.validate()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ChineseChessNet(MODEL_CONFIG).to(device)
    model.eval()
    print('NN + MCTS demo: random weights, no training, no files saved.')
    print(MODEL_CONFIG)
    print(f'get_parameter_count() estimate: {MODEL_CONFIG.get_parameter_count():,}')
    print(f'MCTS configuration: {MCTS_CONFIG}')
    game = GameState()
    demo_network(model, game)
    demo_mcts(model, game)
    demo_nodes(game)
    print('\n[PASS] NN/MCTS demo checks passed; this is not a full pipeline test.')


if __name__ == '__main__':
    main()
