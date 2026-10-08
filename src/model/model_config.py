"""
Cấu hình model
"""
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

    # Training
    learning_rate: float = 0.001
    weight_decay: float = 1e-4
    batch_size: int = 512
    
    # Mixed precision
    use_amp: bool = True  # Automatic Mixed Precision
    
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
