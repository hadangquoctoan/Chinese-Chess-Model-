from pathlib import Path

import numpy as np
import pytest
import torch

from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet
from src.training.replay_buffer import ReplayBuffer
from src.training.supervised import evaluate_supervised, fit_supervised


def _replay():
    replay = ReplayBuffer(2)
    state = np.zeros((19, 10, 9), dtype=np.float32)
    policy = np.zeros(1800, dtype=np.float32)
    policy[0] = 1.0
    replay.add(state, policy, 0.0)
    return replay


def _tiny_model():
    config = ModelConfig(model_size='small', num_residual_blocks=0,
                         num_channels=2, policy_channels=1, value_channels=1,
                         batch_size=1, use_amp=False)
    return ChineseChessNet(config)


def test_validation_returns_metrics_without_mutating_weights():
    config = ModelConfig(model_size='small', num_residual_blocks=0,
                         num_channels=2, policy_channels=1, value_channels=1,
                         batch_size=1, use_amp=False)
    model = ChineseChessNet(config)
    model.train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    metrics = evaluate_supervised(model, _replay(), batch_size=1)
    assert set(metrics) == {'policy_loss', 'value_loss', 'total_loss', 'top1_accuracy'}
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics['top1_accuracy'] <= 1.0
    assert model.training
    assert all(torch.equal(before[key], value)
               for key, value in model.state_dict().items())


def test_evaluation_keeps_model_in_eval_mode():
    model = _tiny_model()
    model.eval()
    evaluate_supervised(model, _replay(), batch_size=1)
    assert not model.training


def test_validation_samples_all_entries_across_multiple_batches(monkeypatch):
    replay = ReplayBuffer(16)
    expected_ids = []
    for index in range(5):
        state = np.zeros((19, 10, 9), dtype=np.float32)
        state[0, 0, 0] = float(index)
        policy = np.zeros(1800, dtype=np.float32)
        policy[index] = 1.0
        replay.add(state, policy, (index - 2) / 2.0)
        expected_ids.append(float(index))

    sampled_ids = []
    original_sample = ReplayBuffer.sample

    def spy_sample(self, batch_size):
        states, policies, values = original_sample(self, batch_size)
        sampled_ids.append(states[:, 0, 0, 0].tolist())
        return states, policies, values

    monkeypatch.setattr(ReplayBuffer, 'sample', spy_sample)
    model = _tiny_model()
    batch_sizes = []
    original_forward = model.forward

    def counting_forward(batch_states):
        batch_sizes.append(len(batch_states))
        return original_forward(batch_states)

    monkeypatch.setattr(model, 'forward', counting_forward)
    metrics = evaluate_supervised(model, replay, batch_size=2)

    assert len(sampled_ids) == 1
    assert sorted(sampled_ids[0]) == expected_ids
    assert batch_sizes == [2, 2, 1]
    assert all(np.isfinite(value) for value in metrics.values())
    assert 0.0 <= metrics['top1_accuracy'] <= 1.0


@pytest.mark.parametrize('options', [
    {'batch_size': 0},
    {'batch_size': True},
    {'batch_size': 1, 'num_workers': -1},
])
def test_evaluation_rejects_invalid_loader_options(options):
    with pytest.raises(ValueError):
        evaluate_supervised(_tiny_model(), _replay(), **options)


def test_early_stopping_saves_only_validation_improvements(tmp_path, monkeypatch):
    from src.utils.model_io import (
        atomic_torch_save, build_model_artifact, load_model_artifact,
    )

    config = ModelConfig(model_size='small', num_residual_blocks=0,
                         num_channels=2, policy_channels=1, value_channels=1,
                         batch_size=3, use_amp=False)
    model = ChineseChessNet(config)

    class FakeTrainer:
        def __init__(self):
            self.epoch = 0
            self.saved = []
            self.model = model
            self.config = config
            self.num_workers = 2

        def train_epoch(self, replay, count):
            self.epoch += 1
            with torch.no_grad():
                for parameter in self.model.parameters():
                    parameter.fill_(float(self.epoch))
            return 0.1, 0.2, 0.3

        def save_model(self, path):
            self.saved.append(Path(path).name)
            atomic_torch_save(build_model_artifact(self.model), path)

    losses = iter([1.0, 0.8, 0.9])
    calls = []

    def fake_evaluate(model, replay, *, batch_size, num_workers):
        calls.append((model, replay, batch_size, num_workers))
        return {
            'policy_loss': 0.0, 'value_loss': 0.0,
            'total_loss': next(losses), 'top1_accuracy': 0.5}

    monkeypatch.setattr('src.training.supervised.evaluate_supervised',
                        fake_evaluate)
    trainer = FakeTrainer()
    validation_replay = _replay()
    output = tmp_path / 'best.pt'
    history = fit_supervised(trainer, _replay(), validation_replay,
                             output, epochs=5, patience=1)
    assert len(history) == 3
    assert trainer.saved == ['best.pt', 'best.pt']
    assert len(calls) == 3
    assert all(call[0] is trainer.model and call[1] is validation_replay
               for call in calls)
    assert [(batch_size, num_workers) for _, _, batch_size, num_workers in calls] == [
        (3, 2), (3, 2), (3, 2)]

    saved = load_model_artifact(output, device='cpu')
    assert all(torch.equal(parameter, torch.full_like(parameter, 2.0))
               for parameter in saved.parameters())


def test_fit_without_validation_saves_final_model(tmp_path):
    class FakeTrainer:
        def __init__(self):
            self.epoch = 0
            self.saved = []

        def train_epoch(self, replay, count):
            self.epoch += 1
            return 0.1, 0.2, 0.3

        def save_model(self, path):
            self.saved.append(Path(path).name)

    trainer = FakeTrainer()
    output = tmp_path / 'nested' / 'final.pt'
    history = fit_supervised(trainer, _replay(), None, output, epochs=2, patience=1)
    assert [record['epoch'] for record in history] == [1, 2]
    assert all('validation' not in record for record in history)
    assert trainer.saved == ['final.pt']
    assert output.parent.is_dir()


@pytest.mark.parametrize('options', [
    {'epochs': 0, 'patience': 1},
    {'epochs': 1, 'patience': 0},
    {'epochs': True, 'patience': 1},
])
def test_invalid_epochs_or_patience_rejected(tmp_path, options):
    class NoopTrainer:
        epoch = 0

    with pytest.raises(ValueError, match='positive integer'):
        fit_supervised(NoopTrainer(), _replay(), None, tmp_path / 'x.pt', **options)
