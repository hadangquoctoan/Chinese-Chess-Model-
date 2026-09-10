from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .model_config import ModelConfig

class ResidualBlock(nn.Module):
    """Residual block with 2 convolutional layers"""
    
    def __init__(self, num_channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class PolicyHead(nn.Module):
    """Policy head - predicts move probabilities"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.conv = nn.Conv2d(config.num_channels, config.policy_channels, 
                             kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(config.policy_channels)
        
        # Calculate flattened size
        flatten_size = config.policy_channels * config.board_height * config.board_width
        self.fc = nn.Linear(flatten_size, config.action_space)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        return x  # Return logits (not softmax)

class ValueHead(nn.Module):
    """Value head - predicts position evaluation"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.conv = nn.Conv2d(config.num_channels, config.value_channels,
                             kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(config.value_channels)
        
        # Calculate flattened size
        flatten_size = config.value_channels * config.board_height * config.board_width
        self.fc1 = nn.Linear(flatten_size, 256)
        self.fc2 = nn.Linear(256, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn(self.conv(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))  # Output in [-1, 1]
        return x

class ChineseChessNet(nn.Module):
    """
    Main neural network for Chinese Chess AI
    
    Architecture:
        Input (19, 10, 9) -> Conv Block -> Residual Blocks -> Policy + Value Heads
    """
    
    def __init__(self, config: ModelConfig = None):
        super().__init__()
        
        if config is None:
            config = ModelConfig()
        self.config = config
        
        # Initial convolutional block
        self.conv_input = nn.Conv2d(config.input_channels, config.num_channels,
                                    kernel_size=3, padding=1)
        self.bn_input = nn.BatchNorm2d(config.num_channels)
        
        # Residual tower
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(config.num_channels)
            for _ in range(config.num_residual_blocks)
        ])
        
        # Output heads
        self.policy_head = PolicyHead(config)
        self.value_head = ValueHead(config)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass
        
        Args:
            x: Input tensor shape (batch, 19, 10, 9)
        
        Returns:
            policy_logits: shape (batch, 1800)
            value: shape (batch, 1)
        """
        # Initial conv
        x = F.relu(self.bn_input(self.conv_input(x)))
        
        # Residual blocks
        for block in self.residual_blocks:
            x = block(x)
        
        # Policy and value outputs
        policy_logits = self.policy_head(x)
        value = self.value_head(x)
        
        return policy_logits, value
    
    def predict(self, board_tensor: Union[torch.Tensor, np.ndarray]) -> Tuple[np.ndarray, float]:
        """
        Prediction for single board (inference)
        
        Args:
            board_tensor: numpy array or tensor shape (19, 10, 9)
        
        Returns:
            policy: numpy array shape (1800,) - probabilities
            value: float - position evaluation
        """
        if not isinstance(board_tensor, torch.Tensor):
            board_tensor = torch.from_numpy(board_tensor)
        if board_tensor.dim() == 3:
            board_tensor = board_tensor.unsqueeze(0)
        if board_tensor.shape[1:] != (
            self.config.input_channels,
            self.config.board_height,
            self.config.board_width,
        ):
            raise ValueError(
                "board_tensor must have shape "
                f"(batch, {self.config.input_channels}, "
                f"{self.config.board_height}, {self.config.board_width})"
            )

        was_training = self.training
        self.eval()
        try:
            with torch.no_grad():
                parameter = next(self.parameters())
                board_tensor = board_tensor.to(
                    device=parameter.device,
                    dtype=parameter.dtype,
                )
                policy_logits, value = self.forward(board_tensor)
                policy = F.softmax(policy_logits, dim=1).cpu().numpy()[0]
                return policy, value.cpu().item()
        finally:
            self.train(was_training)
    
    def get_device(self) -> torch.device:
        """Get device where model is located"""
        return next(self.parameters()).device
    
    def count_parameters(self) -> int:
        """Count trainable parameters"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def __str__(self):
        param_count = self.count_parameters()
        return f"""ChineseChessNet:
  Config: {self.config.model_size}
  Residual Blocks: {self.config.num_residual_blocks}
  Channels: {self.config.num_channels}
  Parameters: {param_count:,} (~{param_count/1e6:.1f}M)
  Device: {self.get_device()}
"""
