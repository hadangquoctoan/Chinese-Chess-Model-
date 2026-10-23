from pathlib import Path

import torch
import yaml

from scripts.extract_learner import extract_learner
from scripts.train import build_model_config
from src.model.neural_network import ChineseChessNet
from src.utils.model_io import load_model_artifact


def _config():
    return {
        'model': {'size': 'small', 'num_residual_blocks': 1, 'num_channels': 4,
                  'policy_channels': 1, 'value_channels': 1},
        'training': {'learning_rate': .001, 'weight_decay': 0., 'batch_size': 2,
                     'use_amp': False},
        'self_play': {'max_game_plies': 4, 'no_capture_limit': 2},
    }


def test_extract_learner_writes_loadable_artifact(tmp_path):
    config_path = tmp_path / 'config.yaml'
    config_path.write_text(yaml.safe_dump(_config()), encoding='utf-8')
    model = ChineseChessNet(build_model_config(_config()))
    checkpoint = tmp_path / 'checkpoint.pt'
    torch.save({'model_state_dict': model.state_dict()}, checkpoint)
    output = tmp_path / 'learner.pt'

    assert extract_learner(checkpoint, config_path, output) == output
    loaded = load_model_artifact(output)
    assert loaded.config.max_game_plies == 4
