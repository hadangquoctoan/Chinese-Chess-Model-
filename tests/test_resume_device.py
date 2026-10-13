"""Full-session checkpoints must keep large replay tensors off the GPU."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet
from src.training.trainer import Trainer


def test_checkpoint_load_maps_replay_to_cpu(monkeypatch, tmp_path):
    config = ModelConfig(num_channels=4, num_residual_blocks=0,
                         policy_channels=1, value_channels=1, use_amp=False)
    trainer = Trainer(ChineseChessNet(config), config, device="cpu",
                      checkpoint_dir=str(tmp_path / "checkpoints"),
                      log_dir=str(tmp_path / "logs"))
    observed = []
    def load(path, device):
        observed.append(str(device))
        return {}
    monkeypatch.setattr("src.training.trainer.safe_torch_load", load)
    trainer.device = "cuda"
    try:
        with pytest.raises(ValueError, match="Legacy"):
            trainer.load_checkpoint("irrelevant.pt")
        assert observed == ["cpu"]
    finally:
        trainer.close()
