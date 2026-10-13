"""Small real-model regressions for transactional training-session persistence."""
from copy import deepcopy
from pathlib import Path

import pytest
import torch

from scripts.train import create_session
from src.training.pipeline import TrainingSession
from src.utils.model_io import build_model_artifact, safe_torch_load


def settings(root: Path) -> dict:
    """Use a tiny CPU network and scratch-only generated outputs."""
    return {
        'model': {'size': 'small', 'num_residual_blocks': 1, 'num_channels': 4,
                  'policy_channels': 1, 'value_channels': 1},
        'training': {'learning_rate': .001, 'weight_decay': .0001,
                     'batch_size': 2, 'use_amp': False},
        'self_play': {'num_games_per_iteration': 2, 'mcts_simulations': 1,
                      'max_game_plies': 2, 'no_capture_limit': 3},
        'replay_buffer': {'max_size': 8},
        'training_loop': {'num_iterations': 2, 'samples_per_iteration': 2,
                          'evaluation_games': 2, 'score_threshold': .55},
        'paths': {'best_model_path': str(root / 'best.pt'),
                  'checkpoint_dir': str(root / 'checkpoints'),
                  'log_dir': str(root / 'logs')},
    }


@pytest.fixture
def real_session(tmp_path):
    torch.set_num_threads(1)
    run = create_session(settings(tmp_path))
    assert run.trainer.device.type == 'cpu'
    try:
        run.start()
        yield run
    finally:
        run.trainer.close()


def snapshot(run: TrainingSession, path: Path) -> dict:
    """Save and restricted-load the complete real trainer checkpoint."""
    run.save_progress(str(path))
    return safe_torch_load(path)


@pytest.mark.parametrize('alias', ['checkpoint_progress.pt', 'checkpoint_iter_1.pt',
                                    'nested/best.pt', '.', '../checkpoints', '../logs/best.pt'])
def test_output_aliases_rejected_before_components(tmp_path, alias):
    config = settings(tmp_path)
    config['paths']['best_model_path'] = str(tmp_path / 'checkpoints' / alias)
    with pytest.raises(ValueError, match='overlap'):
        create_session(config)
    assert not (tmp_path / 'checkpoints').exists()
    assert not (tmp_path / 'logs').exists()


@pytest.mark.parametrize('relation', ['equal', 'nested'])
def test_checkpoint_log_directory_overlap_rejected(tmp_path, relation):
    config = settings(tmp_path)
    directory = Path(config['paths']['checkpoint_dir'])
    config['paths']['log_dir'] = str(directory if relation == 'equal' else directory / 'logs')
    with pytest.raises(ValueError, match='overlap'):
        create_session(config)
    assert not directory.exists()


def test_stale_checkpoint_cannot_overwrite_newer_accepted_best(real_session, tmp_path):
    run = real_session
    path = tmp_path / 'old.pt'
    snapshot(run, path)
    with torch.no_grad():
        next(run.trainer.model.parameters()).add_(1.)
    run.trainer.save_model(run.config['paths']['best_model_path'])
    current_best = Path(run.config['paths']['best_model_path']).read_bytes()
    learner = deepcopy(run.trainer.model.state_dict())
    with pytest.raises(ValueError, match='accepted.best.*new output paths'):
        run.start(resume=str(path))
    assert all(torch.equal(value, learner[key]) for key, value in run.trainer.model.state_dict().items())
    assert Path(run.config['paths']['best_model_path']).read_bytes() == current_best


def test_best_artifact_config_mismatch_is_transactional(real_session, tmp_path):
    run = real_session
    path = tmp_path / 'old.pt'
    snapshot(run, path)
    artifact = build_model_artifact(run.best_model)
    artifact['metadata']['config']['max_game_plies'] += 1
    torch.save(artifact, run.config['paths']['best_model_path'])
    with pytest.raises(ValueError, match='accepted.best.*new output paths'):
        run.start(resume=str(path))


def test_missing_best_requires_explicit_incumbent_publication(real_session, tmp_path):
    run = real_session
    path = tmp_path / 'old.pt'
    with torch.no_grad():
        next(run.trainer.model.parameters()).add_(1.)
    saved = snapshot(run, path)
    run.config['paths']['best_model_path'] = str(tmp_path / 'fork' / 'best.pt')
    with pytest.raises(ValueError, match='publish_missing_best'):
        run.start(resume=str(path))
    run.start(resume=str(path), publish_missing_best=True)
    published = safe_torch_load(run.config['paths']['best_model_path'])
    assert all(torch.equal(value, saved['training_session']['best_model'][key])
               for key, value in published['state_dict'].items())


@pytest.mark.parametrize('field,value', [
    ('phase', 'trian'), ('iteration', -1), ('iteration', 3),
    ('games_completed', -1), ('games_completed', 2), ('games_completed', True),
    ('stale_iterations', -1), ('best_score', float('nan')),
    ('pending_losses', [1., 2., 3.]), ('metrics', [{}]),
    ('version', 7), ('phase', 'evaluate'), ('phase', 'train'),
    ('iteration_settings', {}), ('rules_limits', {}),
])
def test_invalid_session_rejected_without_mutating_learner(real_session, tmp_path, field, value):
    run = real_session
    path = tmp_path / 'invalid.pt'
    saved = snapshot(run, path)
    saved['training_session'][field] = value
    torch.save(saved, path)
    with torch.no_grad():
        next(run.trainer.model.parameters()).add_(1.)
    learner = deepcopy(run.trainer.model.state_dict())
    run.trainer.epoch = 7
    run.trainer.global_step = 9
    with pytest.raises(ValueError):
        run.start(resume=str(path))
    assert run.trainer.epoch == 7
    assert run.trainer.global_step == 9
    assert all(torch.equal(value, learner[key]) for key, value in run.trainer.model.state_dict().items())
    assert len(run.replay_buffer) == 0


@pytest.mark.parametrize('corruption', ['replay', 'best_model', 'rng', 'nonfinite_best'])
def test_nested_session_payload_validation_precedes_learner_mutation(real_session, tmp_path, corruption):
    run = real_session
    path = tmp_path / 'invalid.pt'
    saved = snapshot(run, path)
    state = saved['training_session']
    if corruption == 'replay':
        state['replay']['policies'] = torch.zeros(1, 1800)
    elif corruption == 'best_model':
        key = next(iter(state['best_model']))
        state['best_model'][key] = torch.zeros(1)
    elif corruption == 'nonfinite_best':
        key = next(iter(state['best_model']))
        state['best_model'][key].fill_(float('inf'))
    else:
        state['rng']['torch'] = torch.zeros(1, dtype=torch.uint8)
    torch.save(saved, path)
    run.trainer.epoch = 7
    with pytest.raises(ValueError, match='training_session'):
        run.start(resume=str(path))
    assert run.trainer.epoch == 7


@pytest.mark.parametrize('capacity_source', ['config', 'snapshot'])
def test_resume_rejects_changed_replay_capacity(real_session, tmp_path, capacity_source):
    run = real_session
    path = tmp_path / 'capacity.pt'
    saved = snapshot(run, path)
    if capacity_source == 'config':
        run.config['replay_buffer']['max_size'] += 1
    else:
        saved['training_session']['replay']['max_size'] += 1
        torch.save(saved, path)
    run.trainer.epoch = 7
    with pytest.raises(ValueError, match='replay.*capacity'):
        run.start(resume=str(path))
    assert run.trainer.epoch == 7


@pytest.mark.parametrize('pending', [[], [1., 2.], [1., float('nan'), 3.], [1., 2., -3.]])
def test_evaluate_phase_requires_finite_loss_triple(real_session, tmp_path, pending):
    run = real_session
    path = tmp_path / 'losses.pt'
    saved = snapshot(run, path)
    saved['epoch'] = 1
    saved['training_session'].update(phase='evaluate', games_completed=2, pending_losses=pending)
    torch.save(saved, path)
    run.trainer.epoch = 7
    with pytest.raises(ValueError, match='training_session'):
        run.start(resume=str(path))
    assert run.trainer.epoch == 7


def test_cpu_resume_preserves_dormant_cuda_rng_metadata(real_session, tmp_path):
    run = real_session
    assert not torch.cuda.is_available()
    original = tmp_path / 'cuda-origin.pt'
    saved = snapshot(run, original)
    dormant = [torch.arange(32, dtype=torch.uint8), torch.arange(24, dtype=torch.uint8)]
    saved['training_session']['rng']['cuda'] = dormant
    torch.save(saved, original)
    run.start(resume=str(original))
    rewritten = snapshot(run, tmp_path / 'cpu-rewrite.pt')
    actual = rewritten['training_session']['rng']['cuda']
    assert len(actual) == len(dormant)
    assert all(torch.equal(value, expected) for value, expected in zip(actual, dormant))


@pytest.mark.parametrize('games', [1, 3])
def test_evaluation_gate_requires_even_positive_games(tmp_path, games):
    config = settings(tmp_path)
    config['training_loop']['evaluation_games'] = games
    with pytest.raises(ValueError, match='evaluation_games.*even'):
        create_session(config)
    assert not (tmp_path / 'checkpoints').exists()
    assert not (tmp_path / 'logs').exists()


@pytest.mark.parametrize('corruption', ['negative_wins', 'bad_losses', 'bad_epoch', 'bad_step'])
def test_invalid_completed_session_keeps_optimizer_and_rng(real_session, tmp_path, corruption):
    import random
    import numpy as np

    run = real_session
    run.config['training_loop']['evaluation_mcts_simulations'] = 1
    run.config['training_loop']['evaluation_opening_plies'] = 0
    run.run_iteration()
    path = tmp_path / 'completed.pt'
    saved = snapshot(run, path)
    if corruption == 'negative_wins':
        saved['training_session']['metrics'][0]['wins'] = -1
    elif corruption == 'bad_losses':
        saved['training_session']['metrics'][0]['training_losses'] = [1., float('inf'), 3.]
    elif corruption == 'bad_epoch':
        saved['epoch'] = 9
    else:
        saved['global_step'] = -1
    torch.save(saved, path)
    optimizer = deepcopy(run.trainer.optimizer.state_dict())
    scheduler = deepcopy(run.trainer.scheduler.state_dict())
    learner = deepcopy(run.trainer.model.state_dict())
    generators = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    with pytest.raises(ValueError, match='training_session'):
        run.start(resume=str(path))
    assert run.trainer.scheduler.state_dict() == scheduler
    actual = run.trainer.optimizer.state_dict()
    assert actual['param_groups'] == optimizer['param_groups']
    for parameter, state in optimizer['state'].items():
        for key, value in state.items():
            assert torch.equal(actual['state'][parameter][key], value)
    assert all(torch.equal(value, learner[key]) for key, value in run.trainer.model.state_dict().items())
    assert random.getstate() == generators[0]
    assert np.array_equal(np.random.get_state()[1], generators[1][1])
    assert torch.equal(torch.get_rng_state(), generators[2])
