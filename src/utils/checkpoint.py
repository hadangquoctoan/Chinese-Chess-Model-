from pathlib import Path
from typing import Optional

import torch

class CheckpointManager:
    """Quản lý checkpoints"""
    
    def __init__(self, checkpoint_dir: str = "models/checkpoints",
                 keep_best: int = 3):
        """
        Args:
            checkpoint_dir: Directory lưu checkpoints
            keep_best: Số lượng best checkpoints giữ lại
        """
        if keep_best <= 0:
            raise ValueError("keep_best must be positive")

        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.best_checkpoints = []  # List of (metric, filepath)
    
    def save(self, state_dict: dict, filename: str, metric: Optional[float] = None):
        """
        Lưu checkpoint
        
        Args:
            state_dict: Dict chứa model state
            filename: Tên file
            metric: Metric để track best models (e.g., win_rate)
        """
        filepath = self.checkpoint_dir / filename
        torch.save(state_dict, filepath)
        
        if metric is not None:
            self._update_best_checkpoints(metric, filepath)
    
    def load(self, filename: str, device: str = 'cpu'):
        """
        Load checkpoint
        
        Args:
            filename: Tên file
            device: Device to load to
        
        Returns:
            State dict
        """
        filepath = self.checkpoint_dir / filename
        return torch.load(filepath, map_location=device)
    
    def get_latest(self) -> Optional[Path]:
        """Lấy checkpoint mới nhất"""
        checkpoints = sorted(self.checkpoint_dir.glob("*.pt"), 
                           key=lambda p: p.stat().st_mtime)
        return checkpoints[-1] if checkpoints else None
    
    def get_best(self) -> Optional[Path]:
        """Lấy best checkpoint"""
        if not self.best_checkpoints:
            return None
        return self.best_checkpoints[0][1]
    
    def _update_best_checkpoints(self, metric: float, filepath: Path):
        """Update danh sách best checkpoints"""
        self.best_checkpoints.append((metric, filepath))
        self.best_checkpoints.sort(key=lambda x: x[0], reverse=True)
        
        # Xóa checkpoints thừa
        if len(self.best_checkpoints) > self.keep_best:
            to_remove = self.best_checkpoints[self.keep_best:]
            self.best_checkpoints = self.best_checkpoints[:self.keep_best]
            
            for _, old_filepath in to_remove:
                if old_filepath.exists():
                    old_filepath.unlink()
