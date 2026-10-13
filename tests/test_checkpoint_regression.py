"""Regression coverage for safe, explicitly versioned model persistence."""
from dataclasses import asdict

import pytest
import torch

from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet


def tiny_config(**overrides):
    config = ModelConfig(num_residual_blocks=1, num_channels=4,
                         policy_channels=2, value_channels=1,
                         batch_size=2, use_amp=False)
    for name, value in overrides.items():
        setattr(config, name, value)
    return config


def test_versioned_model_roundtrip_rejects_legacy_and_mismatch(tmp_path):
    from src.utils.model_io import build_model_artifact, load_model_artifact, validate_model_artifact

    model = ChineseChessNet(tiny_config())
    artifact = build_model_artifact(model)
    assert isinstance(artifact['metadata']['config'], dict)
    assert artifact['metadata']['action_encoding_version'] == 'piece-slots-v2'
    path = tmp_path / 'model.pt'
    torch.save(artifact, path)
    restored = load_model_artifact(path)
    assert asdict(restored.config) == asdict(model.config)
    for name, parameter in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[name], parameter)
    torch.save(model.state_dict(), path)
    with pytest.raises(ValueError, match='(?i)legacy|retrain'):
        load_model_artifact(path)
    with pytest.raises(ValueError, match='(?i)config|architecture'):
        validate_model_artifact(artifact, tiny_config(num_channels=8))
    artifact['metadata']['action_encoding_version'] = 'legacy'
    with pytest.raises(ValueError, match='(?i)encoding|retrain'):
        validate_model_artifact(artifact)


def test_envelope_missing_config_versions_rejected():
    from src.utils.model_io import build_model_artifact, validate_model_artifact

    artifact = build_model_artifact(ChineseChessNet(tiny_config()))
    del artifact['metadata']['config']['action_encoding_version']
    with pytest.raises(ValueError, match='(?i)version|encoding|retrain'):
        validate_model_artifact(artifact)


def test_missing_architecture_metadata_is_not_silently_defaulted():
    from src.utils.model_io import build_model_artifact, validate_model_artifact

    artifact = build_model_artifact(ChineseChessNet(tiny_config()))
    del artifact['metadata']['config']['num_channels']
    with pytest.raises(ValueError, match='(?i)config|architecture'):
        validate_model_artifact(artifact)


def test_trainer_checkpoint_roundtrip_safe_plain_config_all_state(tmp_path):
    from src.training.trainer import Trainer
    from src.training.replay_buffer import ReplayBuffer
    import numpy as np

    config = tiny_config()
    trainer = Trainer(ChineseChessNet(config), config, device='cpu',
                      log_dir=str(tmp_path / 'logs'), checkpoint_dir=str(tmp_path / 'cp'))
    replay = ReplayBuffer(2)
    policy = np.zeros(1800, dtype=np.float32)
    policy[0] = 1
    replay.add(np.ones((19, 10, 9), dtype=np.float32), policy, 1.0)
    path = tmp_path / 'checkpoint.pt'
    try:
        trainer.train_epoch(replay, 1)
        trainer.save_checkpoint(path, {'pipeline': {'iteration': 4, 'replay': replay.state_dict()}})
        saved = torch.load(path, weights_only=True)
        assert saved['config'] == asdict(config)
        restored = Trainer(ChineseChessNet(tiny_config()), tiny_config(), device='cpu',
                           log_dir=str(tmp_path / 'otherlogs'), checkpoint_dir=str(tmp_path / 'othercp'))
        try:
            loaded = restored.load_checkpoint(path)
            assert loaded['pipeline']['iteration'] == 4
            assert restored.epoch == trainer.epoch == 1
            assert restored.global_step == trainer.global_step == 1
            assert restored.scheduler.state_dict() == trainer.scheduler.state_dict()
            assert restored.scaler.state_dict() == trainer.scaler.state_dict()
            for name, tensor in trainer.model.state_dict().items():
                torch.testing.assert_close(restored.model.state_dict()[name], tensor)
            for index, state in trainer.optimizer.state_dict()['state'].items():
                for key, value in state.items():
                    torch.testing.assert_close(restored.optimizer.state_dict()['state'][index][key], value)
            saved['config']['learning_rate'] *= 2
            torch.save(saved, path)
            with pytest.raises(ValueError, match='(?i)config'):
                restored.load_checkpoint(path)
            with pytest.raises(ValueError, match='(?i)reserved|overwrite'):
                trainer.save_checkpoint(path, {'epoch': 9})
            trainer.save_model(tmp_path / 'model.pt')
            restored.load_model(tmp_path / 'model.pt')
            torch.save(trainer.model.state_dict(), tmp_path / 'model.pt')
            with pytest.raises(ValueError, match='(?i)legacy|retrain'):
                restored.load_model(tmp_path / 'model.pt')
        finally:
            restored.close()
    finally:
        trainer.close()


def test_checkpoint_manager_reused_filename_dedupes_and_persists_best(tmp_path):
    from src.utils.checkpoint import CheckpointManager

    manager = CheckpointManager(str(tmp_path), keep_best=2)
    manager.save({'weight': torch.tensor([1])}, 'same.pt', metric=0.5)
    manager.save({'weight': torch.tensor([2])}, 'same.pt', metric=0.7)
    manager.save({'weight': torch.tensor([3])}, 'other.pt', metric=0.6)
    assert (tmp_path / 'same.pt').exists()
    assert len(manager.best_checkpoints) == 2
    restarted = CheckpointManager(str(tmp_path), keep_best=2)
    assert restarted.get_best() == tmp_path / 'same.pt'
    assert restarted.load('same.pt')['weight'].item() == 2
    restarted.save({'weight': torch.tensor([4])}, 'third.pt', metric=0.8)
    assert restarted.get_best() == tmp_path / 'third.pt'
    assert not (tmp_path / 'other.pt').exists()
    with pytest.raises(ValueError):
        restarted.save({}, '../escape.pt')
    with pytest.raises(ValueError):
        restarted.save({}, 'bad.pt', metric=float('nan'))


def test_safe_model_loader_rejects_pickled_config_without_executing(tmp_path):
    from src.utils.model_io import load_model_artifact

    path = tmp_path / 'legacy.pt'
    torch.save({'config': tiny_config()}, path)
    with pytest.raises(ValueError, match='(?i)legacy|unsafe|retrain'):
        load_model_artifact(path)


def test_checkpoint_invalid_metadata_and_options_rejected_before_restore(tmp_path):
    from src.training.trainer import Trainer

    config = tiny_config()
    trainer = Trainer(ChineseChessNet(config), config, device='cpu',
                      log_dir=str(tmp_path / 'logs'), checkpoint_dir=str(tmp_path / 'cp'))
    path = tmp_path / 'cp.pt'
    try:
        trainer.save_checkpoint(path)
        payload = torch.load(path, weights_only=True)
        payload['trainer_options']['gradient_accumulation_steps'] = 2
        torch.save(payload, path)
        with pytest.raises(ValueError, match='(?i)options|config'):
            trainer.load_checkpoint(path)
        payload['trainer_options']['gradient_accumulation_steps'] = 1
        del payload['scheduler_state_dict']
        torch.save(payload, path)
        with pytest.raises(ValueError, match='(?i)missing|incomplete'):
            trainer.load_checkpoint(path)
        torch.save([], path)
        with pytest.raises(ValueError, match='(?i)legacy|format|checkpoint'):
            trainer.load_checkpoint(path)
        with pytest.raises(ValueError, match='(?i)safe|unsupported'):
            trainer.save_checkpoint(path, {'custom': tiny_config()})
    finally:
        trainer.close()


def test_cpu_resume_retains_dormant_amp_scaler_state(tmp_path):
    from src.training.trainer import Trainer

    config = tiny_config()
    trainer = Trainer(ChineseChessNet(config), config, device='cpu',
                      log_dir=str(tmp_path / 'logs'), checkpoint_dir=str(tmp_path / 'cp'))
    first, second = tmp_path / 'first.pt', tmp_path / 'second.pt'
    try:
        trainer.scaler = torch.amp.GradScaler('cpu')
        expected = trainer.scaler.state_dict()
        assert expected
        trainer.save_checkpoint(first)
        trainer.scaler = torch.amp.GradScaler('cuda', enabled=False)
        trainer.load_checkpoint(first)
        trainer.save_checkpoint(second)
        assert torch.load(second, weights_only=True)['scaler_state_dict'] == expected
    finally:
        trainer.close()
