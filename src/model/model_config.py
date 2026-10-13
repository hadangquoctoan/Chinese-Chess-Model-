"""
Cấu hình model
"""
import math
from dataclasses import dataclass
from typing import Literal

ModelSize = Literal['small', 'medium', 'large']

@dataclass
class ModelConfig:
    """Cấu hình cho Neural Network"""
    
    # Model size
    model_size: ModelSize = 'medium'
    
    # Architecture
    num_residual_blocks: int = 10
    num_channels: int = 256
    policy_channels: int = 32
    value_channels: int = 4
    
    # Input/Output
    input_channels: int = 19
    board_height: int = 10
    board_width: int = 9
    action_space: int = 1800
    action_encoding_version: str = "piece-slots-v2"
    ruleset_version: str = "xiangqi-training-v2"
    max_game_plies: int = 200
    no_capture_limit: int = 100

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 512
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
    def validate(self) -> None:
        """Reject incompatible tensor/action contracts and invalid budgets."""
        if (self.input_channels, self.board_height, self.board_width,
                self.action_space) != (19, 10, 9, 1800):
            raise ValueError("Model requires input (19, 10, 9) and 1800 actions")
        if type(self.use_amp) is not bool:
            raise ValueError("use_amp must be a YAML boolean, not a string or integer")
        if self.action_encoding_version != "piece-slots-v2":
            raise ValueError("Incompatible action encoding; retrain legacy checkpoints")
        if self.ruleset_version != "xiangqi-training-v2":
            raise ValueError("Incompatible training ruleset")
        for field in ("num_channels", "policy_channels", "value_channels", "batch_size",
                      "max_game_plies", "no_capture_limit"):
            value = getattr(self, field)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if type(self.num_residual_blocks) is not int or self.num_residual_blocks < 0:
            raise ValueError("num_residual_blocks must be a non-negative integer")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be finite and positive")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0:
            raise ValueError("weight_decay must be finite and non-negative")

    @classmethod
    def from_size(cls, size: ModelSize) -> 'ModelConfig':
        """Tạo config từ model size preset"""
        configs = {
            'small': cls(
                model_size='small',
                num_residual_blocks=5,
                num_channels=128,
                policy_channels=16,
                value_channels=2,
                batch_size=256
            ),
            'medium': cls(
                model_size='medium',
                num_residual_blocks=10,
                num_channels=256,
                policy_channels=32,
                value_channels=4,
                batch_size=512
            ),
            'large': cls(
                model_size='large',
                num_residual_blocks=20,
                num_channels=384,
                policy_channels=48,
                value_channels=8,
                batch_size=512
            )
        }
        try:
            return configs[size]
        except KeyError as error:
            raise ValueError(
                f"Unsupported model size: {size}. Expected one of {tuple(configs)}"
            ) from error
    
    def get_parameter_count(self) -> int:
        """Ước tính số lượng parameters"""
        # Initial conv
        params = self.input_channels * self.num_channels * 3 * 3
        
        # Residual blocks
        params += self.num_residual_blocks * (
            2 * self.num_channels * self.num_channels * 3 * 3
        )
        
        # Policy head
        policy_conv = self.num_channels * self.policy_channels * 3 * 3
        policy_fc = (self.policy_channels * self.board_height * self.board_width) * self.action_space
        params += policy_conv + policy_fc
        
        # Value head
        value_conv = self.num_channels * self.value_channels * 3 * 3
        value_fc1 = (self.value_channels * self.board_height * self.board_width) * 256
        value_fc2 = 256 * 1
        params += value_conv + value_fc1 + value_fc2
        
        return params
    
    def __str__(self):
        """Hiển thị thông tin config"""
        param_count = self.get_parameter_count()
        param_mb = param_count * 4 / (1024 * 1024)  # Float32
        
        return f"""ModelConfig({self.model_size}):
  Residual Blocks: {self.num_residual_blocks}
  Channels: {self.num_channels}
  Parameters: ~{param_count/1e6:.1f}M ({param_mb:.1f}MB)
  Batch Size: {self.batch_size}
  Learning Rate: {self.learning_rate}
  Mixed Precision: {self.use_amp}
"""
