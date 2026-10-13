"""Test neural network"""
import sys
from pathlib import Path
from unittest import SkipTest
sys.path.insert(0, str(Path(__file__).parent.parent))

def _torch_available():
    """Return whether the optional PyTorch dependency is installed."""
    try:
        import torch
        return True
    except ImportError:
        return False

def test_model_creation():
    """Test tạo model"""
    if not _torch_available():
        raise SkipTest("PyTorch not installed; model stack not verified")
    
    import torch
    from src.model.neural_network import ChineseChessNet
    from src.model.model_config import ModelConfig
    
    config = ModelConfig.from_size('small')
    model = ChineseChessNet(config)
    
    assert model is not None, "Model should be created"
    print(f"[PASS] Model created: {model.count_parameters():,} parameters")

def test_forward_pass():
    """Test forward pass"""
    if not _torch_available():
        raise SkipTest("PyTorch not installed; model stack not verified")
    
    import torch
    from src.model.neural_network import ChineseChessNet
    from src.model.model_config import ModelConfig
    
    config = ModelConfig.from_size('small')
    model = ChineseChessNet(config)
    model.eval()
    
    # Create dummy input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 19, 10, 9)
    
    with torch.no_grad():
        policy_logits, value = model(input_tensor)
    
    # Check output shapes
    assert policy_logits.shape == (batch_size, 1800), f"Policy shape wrong: {policy_logits.shape}"
    assert value.shape == (batch_size, 1), f"Value shape wrong: {value.shape}"
    
    # Check value range
    assert torch.all(value >= -1) and torch.all(value <= 1), "Value should be in [-1, 1]"
    
    print("[PASS] Forward pass test passed.")

def test_predict():
    """Test predict method"""
    if not _torch_available():
        raise SkipTest("PyTorch not installed; model stack not verified")
    
    import torch
    from src.model.neural_network import ChineseChessNet
    from src.model.model_config import ModelConfig
    
    config = ModelConfig.from_size('small')
    model = ChineseChessNet(config)
    
    # Create dummy board
    board_tensor = torch.randn(19, 10, 9)
    
    policy, value = model.predict(board_tensor)
    
    # Check output types
    assert len(policy) == 1800, f"Policy length wrong: {len(policy)}"
    assert isinstance(value, float), "Value should be float"
    
    # Check policy is probability distribution
    assert abs(policy.sum() - 1.0) < 0.01, "Policy should sum to 1"
    assert model.training, "predict should restore the original train/eval mode"
    
    print("[PASS] Predict method test passed.")


def test_predict_rejects_invalid_shape():
    """Test that predict rejects tensors that violate the input contract."""
    if not _torch_available():
        raise SkipTest("PyTorch not installed; model stack not verified")

    import torch
    from src.model.neural_network import ChineseChessNet
    from src.model.model_config import ModelConfig

    model = ChineseChessNet(ModelConfig.from_size('small'))
    invalid_tensor = torch.randn(19, 9, 9)

    try:
        model.predict(invalid_tensor)
    except ValueError:
        print("[PASS] Invalid prediction shape rejected.")
        return

    raise AssertionError("predict should reject invalid input shapes")

def test_model_sizes():
    """Test different model sizes"""
    if not _torch_available():
        raise SkipTest("PyTorch not installed; model stack not verified")
    
    from src.model.neural_network import ChineseChessNet
    from src.model.model_config import ModelConfig
    
    sizes = ['small', 'medium', 'large']
    
    for size in sizes:
        config = ModelConfig.from_size(size)
        model = ChineseChessNet(config)
        param_count = model.count_parameters()
        
        print(f"  {size:8s}: {param_count:10,} parameters")
    
    print("[PASS] Model sizes test passed.")


def test_mcts_puct_uses_parent_perspective():
    """A parent must prefer the child whose Q is worst for the opponent."""
    if not _torch_available():
        raise SkipTest("PyTorch not installed; model stack not verified")

    from src.game.game_state import GameState
    from src.model.mcts import MCTSNode

    root = MCTSNode(GameState())
    good_for_parent = MCTSNode(GameState(), parent=root, move=(3, 0, 4, 0), prior=0.5)
    bad_for_parent = MCTSNode(GameState(), parent=root, move=(3, 2, 4, 2), prior=0.5)
    good_for_parent.visit_count = 10
    good_for_parent.total_value = -10.0
    bad_for_parent.visit_count = 10
    bad_for_parent.total_value = 10.0
    root.visit_count = 20
    root.children = {
        good_for_parent.move: good_for_parent,
        bad_for_parent.move: bad_for_parent,
    }

    chosen = root.best_child(c_puct=0.0)
    assert chosen is good_for_parent, "PUCT must maximise -child.Q for the parent"
    print("[PASS] MCTS PUCT parent-perspective test passed.")


def test_mcts_search_returns_legal_move_and_policy():
    """A short search must return a legal move and a normalised policy target."""
    if not _torch_available():
        raise SkipTest("PyTorch not installed; model stack not verified")

    from src.game.game_state import GameState
    from src.game.rules import ACTION_SPACE_SIZE, Rules
    from src.model.mcts import MCTS
    from src.model.model_config import ModelConfig
    from src.model.neural_network import ChineseChessNet

    model = ChineseChessNet(ModelConfig.from_size('small'))
    model.eval()
    game = GameState()
    mcts = MCTS(model, num_simulations=8, temperature=0.0, dirichlet_epsilon=0.0)
    move, policy = mcts.search(game)

    assert move in game.get_legal_moves(), f"MCTS returned illegal move {move}"
    assert policy.shape == (ACTION_SPACE_SIZE,)
    assert abs(float(policy.sum()) - 1.0) < 1e-5
    assert Rules.move_to_action_index(*move, board=game.board) >= 0
    print("[PASS] MCTS search legality test passed.")


def test_mcts_backup_flips_value_along_the_path():
    """Backup must store +v at the leaf and -v at its parent."""
    if not _torch_available():
        raise SkipTest("PyTorch not installed; model stack not verified")

    from src.game.game_state import GameState
    from src.model.mcts import MCTS, MCTSNode
    from src.model.model_config import ModelConfig
    from src.model.neural_network import ChineseChessNet

    model = ChineseChessNet(ModelConfig.from_size('small'))
    mcts = MCTS(model, num_simulations=1)
    parent = MCTSNode(GameState())
    child = MCTSNode(GameState(), parent=parent, move=(3, 0, 4, 0), prior=1.0)
    mcts._backup_value([parent, child], 0.8)

    assert child.visit_count == 1
    assert parent.visit_count == 1
    assert abs(child.total_value - 0.8) < 1e-6
    assert abs(parent.total_value - (-0.8)) < 1e-6
    print("[PASS] MCTS backup sign-flip test passed.")


def test_mcts_evaluates_each_new_leaf_once():
    """Root expand plus N simulations should not double-evaluate every leaf."""
    if not _torch_available():
        raise SkipTest("PyTorch not installed; model stack not verified")

    from src.game.game_state import GameState
    from src.model.mcts import MCTS
    from src.model.model_config import ModelConfig
    from src.model.neural_network import ChineseChessNet

    class CountingNet(ChineseChessNet):
        def __init__(self, config):
            super().__init__(config)
            self.forwards = 0

        def forward(self, x):
            self.forwards += 1
            return super().forward(x)

    simulations = 8
    model = CountingNet(ModelConfig.from_size('small'))
    model.eval()
    mcts = MCTS(model, num_simulations=simulations, temperature=1.0, dirichlet_epsilon=0.0)
    mcts.search(GameState())

    # One forward to expand the root, then one per simulation at a new or reused leaf.
    assert 1 <= model.forwards <= simulations + 1, (
        f"expected at most {simulations + 1} forwards, got {model.forwards}"
    )
    print("[PASS] MCTS single-forward-per-leaf test passed.")


if __name__ == '__main__':
    print("Running neural network tests...\n")
    
    if not _torch_available():
        print("[INFO] PyTorch not installed. Install with:")
        print("  pip install torch")
        print("\n[SKIP] Model/MCTS tests were not executed; training readiness is unverified")
    else:
        test_model_creation()
        test_forward_pass()
        test_predict()
        test_predict_rejects_invalid_shape()
        test_model_sizes()
        test_mcts_puct_uses_parent_perspective()
        test_mcts_search_returns_legal_move_and_policy()
        test_mcts_backup_flips_value_along_the_path()
        test_mcts_evaluates_each_new_leaf_once()
        print("\n[SUCCESS] All tests passed!")

