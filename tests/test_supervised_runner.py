"""Supervised config and smoke/full runner regression tests."""
from pathlib import Path
import sys

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import build_model_config

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / 'configs' / 'supervised.yaml'


def _load_supervised_config() -> dict:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding='utf-8'))


def _flag(command: list[str], name: str) -> str:
    """Return the value following ``name`` in a child-process command."""
    return command[command.index(name) + 1]


def test_supervised_config_needs_no_alphazero_sections():
    config = _load_supervised_config()
    assert 'self_play' not in config
    assert 'replay_buffer' not in config
    assert 'training_loop' not in config
    model = build_model_config(config)
    assert model.num_residual_blocks == 5
    assert model.num_channels == 128
    assert model.batch_size == 64
    assert model.use_amp is True


def test_supervised_config_pins_training_ply_normalization():
    config = _load_supervised_config()
    assert config['rules'] == {'max_game_plies': 160, 'no_capture_limit': 100}
    model = build_model_config(config)
    assert model.max_game_plies == 160
    assert model.no_capture_limit == 100


def _alphazero_config(**self_play) -> dict:
    return {
        'model': {'size': 'small', 'num_residual_blocks': 1, 'num_channels': 4,
                  'policy_channels': 1, 'value_channels': 1},
        'training': {'learning_rate': 0.001, 'weight_decay': 0.0,
                     'batch_size': 2, 'use_amp': False},
        'self_play': self_play,
    }


def test_build_model_config_without_rules_keeps_self_play_limits():
    model = build_model_config(
        _alphazero_config(max_game_plies=120, no_capture_limit=40))
    assert model.max_game_plies == 120
    assert model.no_capture_limit == 40


def test_build_model_config_without_rules_uses_alphazero_defaults():
    model = build_model_config(_alphazero_config())
    assert model.max_game_plies == 200
    assert model.no_capture_limit == 100


def test_runner_never_calls_alphazero_training():
    from scripts.run_supervised import supervised_steps

    steps = supervised_steps('python', Path('.'), smoke=True)
    flattened = [part for step in steps for part in step]
    assert all(Path(part).name != 'train.py' for part in flattened)
    assert '--depth' in flattened
    assert '8' in flattened
    assert any(part.endswith('pretrain_supervised.py') for part in flattened)
    assert any(part == 'raw' for part in flattened)


def test_smoke_steps_run_dataset_pretrain_then_raw_and_mcts():
    from scripts.run_supervised import supervised_steps

    steps = supervised_steps('python', Path('workspace'), smoke=True)
    assert [Path(step[1]).name for step in steps] == [
        'build_pikafish_dataset.py',
        'pretrain_supervised.py',
        'measure_vs_pikafish.py',
        'measure_vs_pikafish.py',
    ]
    assert [_flag(step, '--mode') for step in steps[2:]] == ['raw', 'mcts']


def test_full_steps_measure_raw_and_mcts_at_depths_one_and_two():
    from scripts.run_supervised import supervised_steps

    steps = supervised_steps('python', Path('workspace'), smoke=False)
    measurements = steps[2:]
    assert [(_flag(step, '--mode'), _flag(step, '--depth'))
            for step in measurements] == [
        ('raw', '1'), ('raw', '2'), ('mcts', '1'), ('mcts', '2')]
    assert all(_flag(step, '--mcts-sims') == '64'
               for step in measurements if _flag(step, '--mode') == 'mcts')


@pytest.mark.parametrize(('smoke', 'expected_plies'), [(True, '24'), (False, '160')])
def test_runner_pins_ply_limit_for_dataset_and_measurements(smoke, expected_plies):
    from scripts.run_supervised import supervised_steps

    steps = supervised_steps('python', Path('workspace'), smoke=smoke)
    dataset, _, *measurements = steps
    assert _flag(dataset, '--max-game-plies') == expected_plies
    assert [_flag(step, '--max-game-plies') for step in measurements] == [
        expected_plies] * len(measurements)


def test_dataset_step_uses_depth_eight_four_threads_and_256_mb():
    from scripts.run_supervised import supervised_steps

    dataset = supervised_steps('python', Path('workspace'), smoke=True)[0]
    assert _flag(dataset, '--depth') == '8'
    assert _flag(dataset, '--threads') == '4'
    assert _flag(dataset, '--hash-mb') == '256'


def test_dataset_step_pins_opening_plies_to_eight():
    from scripts.run_supervised import supervised_steps

    dataset = supervised_steps('python', Path('workspace'), smoke=True)[0]
    assert _flag(dataset, '--opening-plies') == '8'


@pytest.mark.parametrize('smoke', [True, False])
def test_measurement_steps_use_four_threads_and_256_mb(smoke):
    from scripts.run_supervised import supervised_steps

    measurements = supervised_steps('python', Path('workspace'), smoke=smoke)[2:]
    assert measurements
    assert [_flag(step, '--threads') for step in measurements] == [
        '4'] * len(measurements)
    assert [_flag(step, '--hash-mb') for step in measurements] == [
        '256'] * len(measurements)


@pytest.mark.parametrize(('smoke', 'stem', 'dataset_prefix'), [
    (True, 'supervised_smoke', 'supervised_smoke_'),
    (False, 'supervised', '')])
def test_profiles_use_isolated_artifacts(smoke, stem, dataset_prefix):
    from scripts.run_supervised import supervised_steps

    root = Path('workspace')
    dataset, pretrain, *measurements = supervised_steps(
        'python', root, smoke=smoke)
    labels = root / 'data' / 'teacher_labels'
    assert Path(_flag(dataset, '--train-output')) == labels / f'{dataset_prefix}train.npz'
    assert Path(_flag(dataset, '--validation-output')) == labels / f'{dataset_prefix}validation.npz'
    assert Path(_flag(pretrain, '--data')) == labels / f'{dataset_prefix}train.npz'
    assert Path(_flag(pretrain, '--val-data')) == labels / f'{dataset_prefix}validation.npz'
    assert Path(_flag(pretrain, '--output')) == root / 'models' / stem / 'best_model.pt'
    assert Path(_flag(pretrain, '--log-dir')) == root / 'logs' / stem / 'pretrain'
    assert Path(_flag(pretrain, '--checkpoint-dir')) == (
        root / 'models' / stem / 'pretrain_checkpoints')
    assert all(Path(_flag(step, '--model')) == root / 'models' / stem / 'best_model.pt'
               for step in measurements)


def test_pretrain_step_uses_profile_config_and_patience():
    from scripts.run_supervised import supervised_steps

    root = Path('workspace')
    smoke_pretrain = supervised_steps('python', root, smoke=True)[1]
    full_pretrain = supervised_steps('python', root, smoke=False)[1]
    assert Path(_flag(smoke_pretrain, '--config')) == (
        root / 'configs' / 'supervised_smoke.yaml')
    assert Path(_flag(full_pretrain, '--config')) == (
        root / 'configs' / 'supervised.yaml')
    assert _flag(smoke_pretrain, '--patience') == '2'
    assert _flag(full_pretrain, '--patience') == '2'


def test_smoke_config_matches_base_settings_with_24_ply_rules():
    smoke = yaml.safe_load(
        (ROOT / 'configs' / 'supervised_smoke.yaml').read_text(encoding='utf-8'))
    full = _load_supervised_config()
    assert smoke['model'] == full['model']
    assert smoke['training'] == full['training']
    assert smoke['optimization'] == full['optimization']
    assert smoke['rules'] == {'max_game_plies': 24, 'no_capture_limit': 100}
    assert smoke['paths']['checkpoint_dir'].startswith('models/supervised_smoke/')
    assert smoke['paths']['best_model_path'].startswith('models/supervised_smoke/')
    model = build_model_config(smoke)
    assert model.max_game_plies == 24
    assert model.no_capture_limit == 100


def test_runner_measurement_steps_forward_reproducible_openings():
    from scripts.run_supervised import supervised_steps

    measurements = supervised_steps('python', Path('workspace'), smoke=False)[2:]
    assert [_flag(step, '--opening-plies') for step in measurements] == [
        '4'] * len(measurements)
    assert [_flag(step, '--opening-seed') for step in measurements] == [
        '0'] * len(measurements)


@pytest.mark.parametrize('profile_name', ['SMOKE_PROFILE', 'FULL_PROFILE'])
def test_runner_ply_limit_matches_resolved_profile_config(profile_name):
    from scripts import run_supervised as module

    profile = getattr(module, profile_name)
    config = yaml.safe_load(
        (ROOT / profile.config_path).read_text(encoding='utf-8'))
    model = build_model_config(config)
    assert model.max_game_plies == profile.max_game_plies
    assert model.no_capture_limit == 100


def test_smoke_profile_uses_tiny_budgets():
    from scripts.run_supervised import supervised_steps

    dataset, pretrain, *measurements = supervised_steps(
        'python', Path('workspace'), smoke=True)
    assert _flag(dataset, '--games') == '3'
    assert _flag(dataset, '--validation-games') == '1'
    assert _flag(pretrain, '--epochs') == '1'
    assert [_flag(step, '--games') for step in measurements] == ['2', '2']


def test_full_profile_uses_configured_budgets():
    from scripts.run_supervised import supervised_steps

    dataset, pretrain, *measurements = supervised_steps(
        'python', Path('workspace'), smoke=False)
    assert _flag(dataset, '--games') == '110'
    assert _flag(dataset, '--validation-games') == '10'
    assert _flag(pretrain, '--epochs') == '12'
    assert [_flag(step, '--games') for step in measurements] == ['8'] * 4


def test_main_runs_every_step_with_check(monkeypatch, capsys):
    from scripts import run_supervised as module

    calls = []
    monkeypatch.setattr(module.subprocess, 'run',
                        lambda command, check: calls.append((command, check)))
    module.main(['--smoke'])

    expected = module.supervised_steps(sys.executable, ROOT, smoke=True)
    assert calls == [(command, True) for command in expected]
    assert capsys.readouterr().out.count('+ ') == len(expected)
