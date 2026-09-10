"""
Replay Buffer - lưu trữ experience từ self-play
"""
from collections import deque
import random
from typing import List, Tuple

import numpy as np

class ReplayBuffer:
    """Circular buffer để lưu training data"""
    
    def __init__(self, max_size: int = 500000):
        """
        Args:
            max_size: Số lượng position tối đa trong buffer
        """
        if max_size <= 0:
            raise ValueError("max_size must be positive")

        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def add(self, state: np.ndarray, policy: np.ndarray, value: float):
        """
        Thêm một position vào buffer
        
        Args:
            state: Board tensor shape (19, 10, 9)
            policy: Target policy shape (1800,)
            value: Target value (-1, 0, 1)
        """
        self.buffer.append((state, policy, value))
    
    def add_game(self, states: List[np.ndarray], policies: List[np.ndarray], 
                 winner: float):
        """
        Thêm toàn bộ ván cờ vào buffer
        
        Args:
            states: List of board tensors
            policies: List of MCTS policies
            winner: Game result (1.0, -1.0, or 0.0)
        """
        if len(states) != len(policies):
            raise ValueError("states and policies must contain the same number of items")
        if winner not in (-1.0, 0.0, 1.0):
            raise ValueError("winner must be -1.0, 0.0, or 1.0")

        # Value từ góc nhìn của từng position
        for i, (state, policy) in enumerate(zip(states, policies)):
            # Flip value perspective cho mỗi nước đi
            value = winner if i % 2 == 0 else -winner
            self.add(state, policy, value)
    
    def sample(self, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Sample một batch ngẫu nhiên
        
        Args:
            batch_size: Kích thước batch
        
        Returns:
            states: shape (batch_size, 19, 10, 9)
            policies: shape (batch_size, 1800)
            values: shape (batch_size,)
        """
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.is_empty():
            raise ValueError("Cannot sample from an empty replay buffer")

        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        samples = random.sample(self.buffer, batch_size)
        
        states = np.array([s[0] for s in samples], dtype=np.float32)
        policies = np.array([s[1] for s in samples], dtype=np.float32)
        values = np.array([s[2] for s in samples], dtype=np.float32)
        
        return states, policies, values
    
    def __len__(self):
        """Số lượng positions trong buffer"""
        return len(self.buffer)
    
    def clear(self):
        """Xóa toàn bộ buffer"""
        self.buffer.clear()
    
    def is_empty(self):
        """Kiểm tra buffer có trống không"""
        return len(self.buffer) == 0
    
    def is_full(self):
        """Kiểm tra buffer đã đầy chưa"""
        return len(self.buffer) >= self.max_size
    
    def get_stats(self):
        """Lấy thống kê về buffer"""
        if self.is_empty():
            return {
                'size': 0,
                'capacity': self.max_size,
                'utilization': 0.0
            }
        
        values = [v for _, _, v in self.buffer]
        
        return {
            'size': len(self.buffer),
            'capacity': self.max_size,
            'utilization': len(self.buffer) / self.max_size,
            'value_mean': np.mean(values),
            'value_std': np.std(values),
            'wins': sum(1 for v in values if v > 0.5),
            'losses': sum(1 for v in values if v < -0.5),
            'draws': sum(1 for v in values if -0.5 <= v <= 0.5)
        }
