"""Spawn-based self-play workers must return well-formed games."""
import torch

from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet
from src.training.parallel_self_play import ParallelSelfPlayRunner


def tiny_config() -> ModelConfig:
    config = ModelConfig.from_size('small')
    config.num_residual_blocks = 0
    config.num_channels = 4
    config.policy_channels = 1
    config.value_channels = 1
    config.validate()
    return config


def test_spawn_pool_returns_valid_games_on_cpu():
    torch.set_num_threads(1)
    model = ChineseChessNet(tiny_config())
    options = {'mcts_simulations': 1, 'temperature': 0.,
               'temperature_threshold': 0, 'dirichlet_alpha': 0.3,
               'dirichlet_epsilon': 0., 'max_game_plies': 2,
               'no_capture_limit': 100}
    runner = ParallelSelfPlayRunner(model, options, workers=2)
    try:
        games = [runner.submit(seed).result() for seed in (1, 2)]
    finally:
        runner.close()

    assert len(games) == 2
    for states, policies, winner in games:
        assert 1 <= len(states) <= 2
        assert states[0].shape == (19, 10, 9)
        assert policies[0].shape == (1800,)
        assert winner in (-1., 0., 1.)
