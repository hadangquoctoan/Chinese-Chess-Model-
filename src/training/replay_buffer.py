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
        if type(max_size) is not int or max_size <= 0:
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
        state = np.asarray(state, dtype=np.float32)
        policy = np.asarray(policy, dtype=np.float32)
        if state.shape != (19, 10, 9) or not np.isfinite(state).all():
            raise ValueError('state must be finite with shape (19, 10, 9)')
        if policy.shape != (1800,) or not np.isfinite(policy).all():
            raise ValueError('policy must be finite with shape (1800,)')
        if (policy < 0).any() or not np.isclose(policy.sum(), 1.0, atol=1e-5):
            raise ValueError('policy must be nonnegative and normalized')
        if np.ndim(value) != 0 or not np.isfinite(value) or not -1 <= value <= 1:
            raise ValueError('value must be a finite scalar in [-1, 1]')
        self.buffer.append((state.copy(), policy.copy(), float(value)))
    
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

        # Validate the whole game before touching the live buffer.
        pending = ReplayBuffer(max(1, len(states)))
        for i, (state, policy) in enumerate(zip(states, policies)):
            pending.add(state, policy, winner if i % 2 == 0 else -winner)
        self.buffer.extend(pending.buffer)
    
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
        if type(batch_size) is not int or batch_size <= 0:
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
    
    def state_dict(self) -> dict:
        """Return ordered CPU tensors and primitive metadata for safe checkpoints."""
        import torch

        from ..game.rules import ACTION_ENCODING_VERSION, RULESET_VERSION

        entries = list(self.buffer)
        states = np.asarray([s for s, _, _ in entries], dtype=np.float32).reshape(-1, 19, 10, 9)
        policies = np.asarray([p for _, p, _ in entries], dtype=np.float32).reshape(-1, 1800)
        values = np.asarray([v for _, _, v in entries], dtype=np.float32)
        return {
            'format_version': 1, 'max_size': self.max_size,
            'action_encoding_version': ACTION_ENCODING_VERSION,
            'ruleset_version': RULESET_VERSION,
            'states': torch.from_numpy(states),
            'policies': torch.from_numpy(policies),
            'values': torch.from_numpy(values),
        }

    def load_state_dict(self, payload: dict) -> None:
        """Validate an entire snapshot before replacing the live replay buffer."""
        from ..game.rules import ACTION_ENCODING_VERSION, RULESET_VERSION

        if not isinstance(payload, dict) or payload.get('format_version') != 1:
            raise ValueError('Legacy replay data; retrain with versioned data.')
        if payload.get('action_encoding_version') != ACTION_ENCODING_VERSION:
            raise ValueError('Incompatible replay action encoding; retrain.')
        if payload.get('ruleset_version') != RULESET_VERSION:
            raise ValueError('Incompatible replay ruleset; retrain.')
        required = {'states', 'policies', 'values', 'max_size'}
        if not required.issubset(payload):
            raise ValueError('Replay snapshot is missing arrays/capacity.')
        arrays = []
        for name in ('states', 'policies', 'values'):
            value = payload[name]
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            arrays.append(np.asarray(value, dtype=np.float32))
        states, policies, values = arrays
        if states.ndim != 4:
            raise ValueError('Replay states must be a four-dimensional batch.')
        count = len(states)
        if states.shape != (count, 19, 10, 9) or policies.shape != (count, 1800) or values.shape != (count,):
            raise ValueError('Replay snapshot array shapes/counts disagree.')
        capacity = payload['max_size']
        if type(capacity) is not int or capacity <= 0 or count > capacity:
            raise ValueError('Invalid replay capacity or snapshot count.')
        replacement = ReplayBuffer(capacity)
        for state, policy, value in zip(states, policies, values):
            replacement.add(state, policy, value)
        self.max_size, self.buffer = replacement.max_size, replacement.buffer

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
