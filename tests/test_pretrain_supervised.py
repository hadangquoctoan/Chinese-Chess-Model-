from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.build_teacher_labels import save_teacher_npz
from scripts.pretrain_supervised import (
    check_init_architecture, main, validate_validation_contract,
)
from src.game.game_state import GameState
from src.model.model_config import ModelConfig
from src.training.replay_buffer import ReplayBuffer


def _fields():
    return {
        'num_residual_blocks': 1, 'num_channels': 4, 'policy_channels': 1,
        'value_channels': 1, 'input_channels': 19, 'board_height': 10,
        'board_width': 9, 'action_space': 1800,
        'action_encoding_version': 'piece-slots-v2',
        'ruleset_version': 'xiangqi-training-v2',
        'max_game_plies': 4, 'no_capture_limit': 2,
    }


def test_architecture_mismatch_is_rejected():
    config = SimpleNamespace(**_fields())
    loaded = SimpleNamespace(config=SimpleNamespace(**_fields()))
    check_init_architecture(loaded, config)
    loaded.config.max_game_plies = 200
    with pytest.raises(ValueError, match='max_game_plies'):
        check_init_architecture(loaded, config)


def _model_config():
    return ModelConfig(model_size='small', num_residual_blocks=0,
                       num_channels=2, policy_channels=1, value_channels=1,
                       batch_size=1, use_amp=False)


def _write_archive(path, game_ids, *, max_game_plies=200,
                   no_capture_limit=100, include_game_ids=True):
    """Write a real supervised archive matching the tiny test model config."""
    state = GameState().to_tensor()
    policy = np.full(1800, 1 / 1800, dtype=np.float32)
    count = len(game_ids)
    save_teacher_npz(
        path, [state] * count, [policy] * count, [0.0] * count,
        depth=2, engine_path=path,
        game_ids=game_ids if include_game_ids else None,
        max_game_plies=max_game_plies, no_capture_limit=no_capture_limit)
    return path


def _rewrite_archive(path, **updates):
    """Replace or remove one metadata field without touching the arrays."""
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    for name, value in updates.items():
        if value is None:
            payload.pop(name, None)
        else:
            payload[name] = np.asarray(value)
    np.savez_compressed(path, **payload)


def test_validation_contract_accepts_matching_archives(tmp_path):
    train = _write_archive(tmp_path / 'train.npz', [0, 1])
    validation = _write_archive(tmp_path / 'validation.npz', [2])
    validate_validation_contract(train, validation, _model_config())


def test_validation_contract_rejects_same_path(tmp_path):
    path = _write_archive(tmp_path / 'data.npz', [0, 1])
    with pytest.raises(ValueError, match='different files'):
        validate_validation_contract(path, path, _model_config())


def test_validation_contract_rejects_missing_game_ids(tmp_path):
    train = _write_archive(tmp_path / 'train.npz', [0, 1])
    validation = _write_archive(
        tmp_path / 'validation.npz', [2], include_game_ids=False)
    with pytest.raises(ValueError, match='missing game_ids'):
        validate_validation_contract(train, validation, _model_config())


def test_validation_contract_rejects_misaligned_game_ids(tmp_path):
    train = _write_archive(tmp_path / 'train.npz', [0, 1])
    validation = _write_archive(tmp_path / 'validation.npz', [2])
    _rewrite_archive(validation, game_ids=np.array([2, 3]))
    with pytest.raises(ValueError, match='one entry per state'):
        validate_validation_contract(train, validation, _model_config())


def test_validation_contract_rejects_overlapping_game_ids(tmp_path):
    train = _write_archive(tmp_path / 'train.npz', [0, 1])
    validation = _write_archive(tmp_path / 'validation.npz', [1, 2])
    with pytest.raises(ValueError, match='overlap: \\[1\\]'):
        validate_validation_contract(train, validation, _model_config())


def test_validation_contract_rejects_mismatched_state_limits(tmp_path):
    train = _write_archive(tmp_path / 'train.npz', [0])
    validation = _write_archive(
        tmp_path / 'validation.npz', [1], max_game_plies=24)
    with pytest.raises(ValueError, match='max_game_plies=24 does not match'):
        validate_validation_contract(train, validation, _model_config())


def test_validation_contract_requires_state_limit_metadata(tmp_path):
    train = _write_archive(tmp_path / 'train.npz', [0])
    validation = _write_archive(tmp_path / 'validation.npz', [1])
    _rewrite_archive(train, max_game_plies=None)
    with pytest.raises(ValueError, match='missing max_game_plies'):
        validate_validation_contract(train, validation, _model_config())


def test_validation_contract_rejects_encoding_mismatch(tmp_path):
    train = _write_archive(tmp_path / 'train.npz', [0])
    validation = _write_archive(tmp_path / 'validation.npz', [1])
    _rewrite_archive(validation, action_encoding_version='legacy-v1')
    with pytest.raises(ValueError, match='action_encoding_version'):
        validate_validation_contract(train, validation, _model_config())


def test_validation_contract_rejects_non_red_centric_values(tmp_path):
    train = _write_archive(tmp_path / 'train.npz', [0])
    validation = _write_archive(tmp_path / 'validation.npz', [1])
    _rewrite_archive(validation, value_convention='side_to_move')
    with pytest.raises(ValueError, match="must be 'red_centric'"):
        validate_validation_contract(train, validation, _model_config())


def _replay():
    replay = ReplayBuffer(2)
    state = np.zeros((19, 10, 9), dtype=np.float32)
    policy = np.zeros(1800, dtype=np.float32)
    policy[0] = 1.0
    replay.add(state, policy, 0.0)
    return replay


class RecordingTrainer:
    """Capture the options and artifacts a supervised CLI run produces."""

    instances = []

    def __init__(self, model, config, checkpoint_dir, log_dir, *, num_workers=0):
        self.model = model
        self.config = config
        self.num_workers = num_workers
        self.epoch = 0
        self.saved = []
        self.closed = False
        type(self).instances.append(self)

    def train_epoch(self, replay_buffer, num_samples):
        self.epoch += 1
        return 0.1, 0.2, 0.3

    def save_model(self, filepath):
        self.saved.append(Path(filepath).name)
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        Path(filepath).write_bytes(b'weights')

    def close(self):
        self.closed = True


def _setup_cli(monkeypatch, *, load_real_buffers=False):
    model_config = _model_config()
    converted = []
    monkeypatch.setattr('scripts.pretrain_supervised.load_config',
                        lambda path: {'optimization': {}})
    monkeypatch.setattr('scripts.pretrain_supervised.build_model_config',
                        lambda config: model_config)
    if not load_real_buffers:
        monkeypatch.setattr('scripts.pretrain_supervised.buffer_from_npz',
                            lambda path: converted.append(path) or _replay())
    monkeypatch.setattr('scripts.pretrain_supervised.Trainer', RecordingTrainer)
    RecordingTrainer.instances.clear()
    return converted


def test_cli_with_validation_data_loads_archives_and_saves(tmp_path, monkeypatch,
                                                           capsys):
    converted = _setup_cli(monkeypatch, load_real_buffers=True)
    train_path = _write_archive(tmp_path / 'train.npz', [0, 1])
    val_path = _write_archive(tmp_path / 'validation.npz', [2])
    output = tmp_path / 'pretrained.pt'
    main(['--data', str(train_path), '--val-data', str(val_path),
          '--output', str(output), '--epochs', '1', '--patience', '1'])
    trainer = RecordingTrainer.instances[-1]
    assert converted == []
    assert trainer.saved == ['pretrained.pt']
    assert trainer.epoch == 1
    assert trainer.closed
    assert output.read_bytes() == b'weights'
    printed = capsys.readouterr().out
    assert 'val_total=' in printed
    assert 'val_top1=' in printed


def test_cli_without_validation_still_saves_final_model(tmp_path, monkeypatch, capsys):
    converted = _setup_cli(monkeypatch)
    train_path = tmp_path / 'train.npz'
    output = tmp_path / 'pretrained.pt'
    main(['--data', str(train_path), '--output', str(output), '--epochs', '2'])
    trainer = RecordingTrainer.instances[-1]
    assert converted == [train_path]
    assert trainer.saved == ['pretrained.pt']
    assert trainer.epoch == 2
    assert trainer.closed
    assert 'val_top1=' not in capsys.readouterr().out


def test_cli_raises_when_no_artifact_was_written(tmp_path, monkeypatch, capsys):
    _setup_cli(monkeypatch, load_real_buffers=True)
    train_path = _write_archive(tmp_path / 'train.npz', [0])
    val_path = _write_archive(tmp_path / 'validation.npz', [1])
    monkeypatch.setattr(
        'src.training.supervised.evaluate_supervised',
        lambda *args, **kwargs: {
            'policy_loss': float('nan'), 'value_loss': float('nan'),
            'total_loss': float('nan'), 'top1_accuracy': 0.0})
    output = tmp_path / 'missing.pt'
    with pytest.raises(RuntimeError, match='No supervised artifact was written'):
        main(['--data', str(train_path), '--val-data', str(val_path),
              '--output', str(output), '--epochs', '1', '--patience', '1'])
    assert RecordingTrainer.instances[-1].closed
    assert 'Wrote' not in capsys.readouterr().out


@pytest.mark.parametrize('patience', ['0', '-1'])
def test_cli_rejects_nonpositive_patience(tmp_path, monkeypatch, capsys, patience):
    _setup_cli(monkeypatch)
    with pytest.raises(SystemExit):
        main(['--data', str(tmp_path / 'train.npz'),
              '--output', str(tmp_path / 'pretrained.pt'),
              '--patience', patience])
    assert 'patience must be positive' in capsys.readouterr().err
    assert RecordingTrainer.instances == []
