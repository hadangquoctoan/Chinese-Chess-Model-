"""
Dataset Saver - Save/load training data to disk (optional utility)
"""
import pickle
import numpy as np
from pathlib import Path
from typing import Tuple, List


class DatasetSaver:
    """Utility to save/load replay buffer data to disk."""
    
    def __init__(self, data_dir: str = "data/training_data"):
        """
        Args:
            data_dir: Directory to save dataset files
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
    
    def save_buffer(self, replay_buffer, filename: str = "replay_buffer.pkl"):
        """
        Save replay buffer to disk.
        
        Args:
            replay_buffer: ReplayBuffer object
            filename: Output filename
        """
        filepath = self.data_dir / filename
        
        # Convert buffer to list for pickling
        buffer_data = list(replay_buffer.buffer)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'data': buffer_data,
                'max_size': replay_buffer.max_size,
                'size': len(replay_buffer)
            }, f)
        
        print(f"Buffer saved: {filepath} ({len(replay_buffer)} positions)")
        return str(filepath)
    
    def load_buffer(self, filename: str = "replay_buffer.pkl"):
        """
        Load replay buffer from disk.
        
        Args:
            filename: Input filename
            
        Returns:
            ReplayBuffer object
        """
        from ..training.replay_buffer import ReplayBuffer
        
        filepath = self.data_dir / filename
        
        with open(filepath, 'rb') as f:
            data_dict = pickle.load(f)
        
        # Create new buffer
        replay_buffer = ReplayBuffer(max_size=data_dict['max_size'])
        
        # Add data back
        for state, policy, value in data_dict['data']:
            replay_buffer.add(state, policy, value)
        
        print(f"Buffer loaded: {filepath} ({len(replay_buffer)} positions)")
        return replay_buffer
    
    def save_games(self, games: List[Tuple[List[np.ndarray], List[np.ndarray], float]], 
                   filename: str):
        """
        Save raw self-play games.
        
        Args:
            games: List of (states, policies, winner) tuples
            filename: Output filename
        """
        filepath = self.data_dir / filename
        
        with open(filepath, 'wb') as f:
            pickle.dump(games, f)
        
        print(f"Games saved: {filepath} ({len(games)} games)")
        return str(filepath)
    
    def load_games(self, filename: str) -> List[Tuple[List[np.ndarray], List[np.ndarray], float]]:
        """
        Load raw self-play games.
        
        Args:
            filename: Input filename
            
        Returns:
            List of (states, policies, winner) tuples
        """
        filepath = self.data_dir / filename
        
        with open(filepath, 'rb') as f:
            games = pickle.load(f)
        
        print(f"Games loaded: {filepath} ({len(games)} games)")
        return games
    
    def save_iteration_data(self, iteration: int, games: List, replay_buffer):
        """
        Save data from one training iteration.
        
        Args:
            iteration: Iteration number
            games: Self-play games from this iteration
            replay_buffer: Current replay buffer state
        """
        # Save games
        games_file = f"iter_{iteration:03d}_games.pkl"
        self.save_games(games, games_file)
        
        # Save buffer snapshot
        buffer_file = f"iter_{iteration:03d}_buffer.pkl"
        self.save_buffer(replay_buffer, buffer_file)
        
        print(f"Iteration {iteration} data saved.")
    
    def export_to_numpy(self, replay_buffer, filename: str = "dataset.npz"):
        """
        Export replay buffer to numpy format (.npz).
        
        Args:
            replay_buffer: ReplayBuffer object
            filename: Output filename
        """
        filepath = self.data_dir / filename
        
        # Extract arrays
        states = np.array([s[0] for s in replay_buffer.buffer], dtype=np.float32)
        policies = np.array([s[1] for s in replay_buffer.buffer], dtype=np.float32)
        values = np.array([s[2] for s in replay_buffer.buffer], dtype=np.float32)
        
        # Save to .npz
        np.savez_compressed(
            filepath,
            states=states,
            policies=policies,
            values=values
        )
        
        print(f"Dataset exported: {filepath}")
        print(f"  States: {states.shape}")
        print(f"  Policies: {policies.shape}")
        print(f"  Values: {values.shape}")
        return str(filepath)
    
    def import_from_numpy(self, filename: str = "dataset.npz"):
        """
        Import replay buffer from numpy format.
        
        Args:
            filename: Input filename
            
        Returns:
            ReplayBuffer object
        """
        from ..training.replay_buffer import ReplayBuffer
        
        filepath = self.data_dir / filename
        
        # Load .npz
        data = np.load(filepath)
        states = data['states']
        policies = data['policies']
        values = data['values']
        
        # Create buffer
        replay_buffer = ReplayBuffer(max_size=len(states))
        
        # Add data
        for state, policy, value in zip(states, policies, values):
            replay_buffer.add(state, policy, value)
        
        print(f"Dataset imported: {filepath} ({len(replay_buffer)} positions)")
        return replay_buffer
