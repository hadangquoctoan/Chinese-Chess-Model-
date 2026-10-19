"""Atomic checkpoint persistence and restart-safe best-model retention."""
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Optional

import torch

from .model_io import atomic_torch_save


class CheckpointManager:
    """Track highest finite metrics without duplicate filename deletion."""

    def __init__(self, checkpoint_dir: str = 'models/checkpoints', keep_best: int = 3):
        if not isinstance(keep_best, int) or keep_best <= 0:
            raise ValueError('keep_best must be positive')
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_best = keep_best
        self.metadata_path = self.checkpoint_dir / 'best_checkpoints.json'
        self.best_checkpoints: list[tuple[float, Path]] = []
        if self.metadata_path.exists():
            with self.metadata_path.open(encoding='utf-8') as stream:
                metadata = json.load(stream)
            for entry in metadata['best_checkpoints']:
                path = self._filepath(entry['filename'])
                metric = float(entry['metric'])
                if path.exists() and math.isfinite(metric):
                    self.best_checkpoints = [(m, p) for m, p in self.best_checkpoints if p != path]
                    self.best_checkpoints.append((metric, path))
            self.best_checkpoints.sort(key=lambda item: item[0], reverse=True)

    def _filepath(self, filename: str) -> Path:
        if not filename or Path(filename).name != filename or filename == self.metadata_path.name:
            raise ValueError('Checkpoint filename must be a basename inside checkpoint_dir')
        return self.checkpoint_dir / filename

    def save(self, state_dict: dict, filename: str, metric: Optional[float] = None) -> None:
        """Save atomically, optionally updating retained best files."""
        filepath = self._filepath(filename)
        if metric is not None and not math.isfinite(metric):
            raise ValueError('Checkpoint metric must be finite')
        atomic_torch_save(state_dict, filepath)
        # A replacement without a metric invalidates the old file's score.
        self.best_checkpoints = [(m, p) for m, p in self.best_checkpoints if p != filepath]
        if metric is not None:
            self.best_checkpoints.append((float(metric), filepath))
        self.best_checkpoints.sort(key=lambda item: item[0], reverse=True)
        removed = self.best_checkpoints[self.keep_best:]
        self.best_checkpoints = self.best_checkpoints[:self.keep_best]
        self._persist_metadata()
        for _, old_filepath in removed:
            old_filepath.unlink(missing_ok=True)

    def load(self, filename: str, device: str = 'cpu') -> dict:
        """Load with PyTorch's restricted tensor/primitive unpickler only."""
        return torch.load(self._filepath(filename), map_location=device, weights_only=True)

    def get_latest(self) -> Optional[Path]:
        """Return the most recently written .pt or .pth checkpoint."""
        checkpoints = list(self.checkpoint_dir.glob('*.pt')) + list(self.checkpoint_dir.glob('*.pth'))
        return max(checkpoints, key=lambda path: path.stat().st_mtime_ns) if checkpoints else None

    def get_best(self) -> Optional[Path]:
        """Return the retained best existing checkpoint, including after restart."""
        return next((path for _, path in self.best_checkpoints if path.exists()), None)

    def _persist_metadata(self) -> None:
        metadata = {'format_version': 1, 'best_checkpoints': [
            {'metric': metric, 'filename': path.name} for metric, path in self.best_checkpoints
        ]}
        descriptor, temporary = tempfile.mkstemp(dir=self.checkpoint_dir, prefix='.best.')
        try:
            with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
                json.dump(metadata, stream, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.metadata_path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
