"""Validate model contracts before allocating network parameters."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet


@pytest.mark.parametrize("settings", [
    {"action_space": 1799}, {"input_channels": 18}, {"board_height": 9},
    {"num_channels": 0}, {"num_residual_blocks": -1},
    {"learning_rate": float("nan")}, {"batch_size": 0},
])
def test_model_rejects_invalid_contract(settings):
    with pytest.raises(ValueError):
        ChineseChessNet(ModelConfig(**settings))


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_use_amp_requires_boolean(value):
    with pytest.raises(ValueError, match="use_amp"):
        ModelConfig(use_amp=value).validate()


def test_model_config_marks_new_training_contract():
    config = ModelConfig()
    assert config.action_encoding_version == "piece-slots-v2"
    assert config.ruleset_version == "xiangqi-training-v2"


def test_model_config_keeps_concrete_rules_limits_in_artifacts():
    config = ModelConfig(max_game_plies=40, no_capture_limit=30)
    assert config.max_game_plies == 40
    assert config.no_capture_limit == 30
    config.validate()


@pytest.mark.parametrize("field", ["max_game_plies", "no_capture_limit"])
def test_model_config_rejects_invalid_rules_limits(field):
    with pytest.raises(ValueError):
        ModelConfig(**{field: 0}).validate()



def test_missing_torch_is_reported_as_skip_not_pass(monkeypatch):
    import importlib.util
    import unittest

    spec = importlib.util.spec_from_file_location(
        "model_tests", Path(__file__).with_name("test_model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "_torch_available", lambda: False)
    with pytest.raises(unittest.SkipTest):
        module.test_forward_pass()

