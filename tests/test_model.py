"""Test neural network"""
import sys
from pathlib import Path
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
        print("[SKIP] PyTorch not installed, skipping model tests")
        return
    
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
        return
    
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
        return
    
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
        return

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
        return
    
    from src.model.neural_network import ChineseChessNet
    from src.model.model_config import ModelConfig
    
    sizes = ['small', 'medium', 'large']
    
    for size in sizes:
        config = ModelConfig.from_size(size)
        model = ChineseChessNet(config)
        param_count = model.count_parameters()
        
        print(f"  {size:8s}: {param_count:10,} parameters")
    
    print("[PASS] Model sizes test passed.")

if __name__ == '__main__':
    print("Running neural network tests...\n")
    
    if not _torch_available():
        print("[INFO] PyTorch not installed. Install with:")
        print("  pip install torch")
        print("\nSkipping neural network tests (not required for Colab training)")
        print("[SUCCESS] Tests completed (PyTorch tests skipped)")
    else:
        test_model_creation()
        test_forward_pass()
        test_predict()
        test_predict_rejects_invalid_shape()
        test_model_sizes()
        print("\n[SUCCESS] All tests passed!")

