"""Regression tests for the resumable training orchestration."""
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def settings(tmp_path):
    return {
        'self_play': {'num_games_per_iteration': 2, 'mcts_simulations': 3,
                      'temperature': .8, 'temperature_threshold': 4,
                      'dirichlet_alpha': .4, 'dirichlet_epsilon': .2},
        'replay_buffer': {'max_size': 20},
        'training_loop': {'num_iterations': 2, 'samples_per_iteration': 2,
                          'evaluation_games': 2, 'win_rate_threshold': .55,
                          'checkpoint_every': 1},
        'paths': {'best_model_path': str(tmp_path / 'best.pt'),
                  'checkpoint_dir': str(tmp_path / 'checkpoints'),
                  'log_dir': str(tmp_path / 'logs')},
    }


def test_fresh_run_refuses_existing_best_without_modifying_it(tmp_path):
    from src.training.pipeline import check_fresh_paths

    config = settings(tmp_path)
    best = Path(config['paths']['best_model_path'])
    best.write_bytes(b'legacy accepted model')
    with pytest.raises(FileExistsError, match='occupied'):
        check_fresh_paths(config)
    assert best.read_bytes() == b'legacy accepted model'


class TinyReplay:
    def __init__(self):
        self.games = []

    def add_game(self, *game):
        self.games.append(game)

    def state_dict(self):
        return {'games': self.games.copy()}

    def load_state_dict(self, state):
        self.games = state['games'].copy()


class TinyTrainer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1, bias=False)
        self.model.weight.data.fill_(1.)
        self.epoch = 0
        self.saved_models = 0
        self.trained = 0

    def train_epoch(self, replay_buffer, num_samples):
        self.model.weight.data.add_(1.)
        self.epoch += 1
        self.trained += 1
        return (1., 2., 3.)

    def save_model(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        self.saved_models += 1

    def save_checkpoint(self, path, additional_info):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({'model': self.model.state_dict(), 'epoch': self.epoch,
                    **additional_info}, path)

    def load_checkpoint(self, path):
        state = torch.load(path, weights_only=True)
        self.model.load_state_dict(state['model'])
        self.epoch = state['epoch']
        return state


class ControlledWorker:
    models = []
    options = []

    def __init__(self, model, **options):
        self.models.append(float(model.weight.item()) if hasattr(model, 'weight') else None)
        self.options.append(options)

    def play_game(self, verbose=False):
        return ([torch.zeros(19, 10, 9)], [torch.ones(1800) / 1800], 0.)


class RejectingEvaluator:
    def evaluate(self, **kwargs):
        return (.25, 0, 1, 1)


def session(config, **kwargs):
    from src.training.pipeline import TrainingSession

    return TrainingSession(TinyTrainer(), TinyReplay(), config,
                           worker_factory=ControlledWorker,
                           evaluator=RejectingEvaluator(), **kwargs)


def test_rejected_candidate_is_not_next_self_play_incumbent(tmp_path):
    ControlledWorker.models.clear()
    run = session(settings(tmp_path))
    run.start()
    run.run_iteration()
    run.run_iteration()
    assert ControlledWorker.models == [1., 1.]
    assert run.trainer.model.weight.item() == 3.  # Continuous learner, not rollback.
    assert run.best_model.weight.item() == 1.


def test_resume_restores_incumbent_replay_metrics_iteration_rng_without_save(tmp_path):
    import random
    import numpy as np

    config = settings(tmp_path)
    run = session(config)
    run.start()
    run.run_iteration()
    run.best_score = .6
    path = tmp_path / 'resume.pt'
    run.save_progress(str(path))
    expected_random = (random.random(), float(np.random.random()), torch.rand(1).item())
    best_bytes = Path(config['paths']['best_model_path']).read_bytes()
    restored = session(config)
    restored.start(resume=str(path))
    assert restored.iteration == 1
    assert restored.best_score == .6
    assert restored.stale_iterations == 1
    assert restored.metrics == run.metrics
    assert len(restored.replay_buffer.games) == 2
    assert restored.best_model.weight.item() == 1.
    assert restored.trainer.model.weight.item() == 2.
    assert restored.trainer.saved_models == 0
    assert Path(config['paths']['best_model_path']).read_bytes() == best_bytes
    assert (random.random(), float(np.random.random()), torch.rand(1).item()) == expected_random


def test_time_checkpoint_resumes_after_game_without_repeating_games(tmp_path):
    config = settings(tmp_path)
    config['training_loop']['auto_checkpoint_interval'] = 5
    clock = SimpleNamespace(now=0.)
    run = session(config, clock=lambda: clock.now)
    run.start()
    clock.now = 6.
    run.advance()
    path = Path(config['paths']['checkpoint_dir']) / 'checkpoint_progress.pt'
    state = torch.load(path, weights_only=True)['training_session']
    assert state['phase'] == 'self_play'
    assert state['games_completed'] == 1
    restored = session(config, clock=lambda: clock.now)
    restored.start(resume=str(path))
    restored.run_iteration()
    assert len(restored.replay_buffer.games) == 2
    assert restored.trainer.trained == 1


def test_resume_after_epoch_does_not_train_it_again(tmp_path):
    config = settings(tmp_path)
    run = session(config)
    run.start()
    run.advance()
    run.advance()
    run.advance()
    assert run.phase == 'evaluate'
    path = tmp_path / 'epoch.pt'
    run.save_progress(str(path))
    restored = session(config)
    restored.start(resume=str(path))
    restored.run_iteration()
    assert restored.trainer.trained == 0
    assert restored.metrics[0]['training_losses'] == [1., 2., 3.]


def test_checkpoint_every_and_early_stopping_stop_actual_iterations(tmp_path):
    config = settings(tmp_path)
    config['training_loop'].update(num_iterations=10, checkpoint_every=2,
                                   early_stopping_patience=3)
    run = session(config)
    run.start()
    run.run()
    assert run.iteration == 3
    paths = list(Path(config['paths']['checkpoint_dir']).glob('checkpoint_iter_*.pt'))
    assert [path.name for path in paths] == ['checkpoint_iter_2.pt']
    final = torch.load(Path(config['paths']['checkpoint_dir']) / 'checkpoint_progress.pt',
                       weights_only=True)['training_session']
    assert final['iteration'] == 3


@pytest.mark.parametrize('value', [0, -1, 17, True, '2', 2.5])
def test_invalid_parallel_workers_rejected_before_start(tmp_path, value):
    config = settings(tmp_path)
    config['self_play']['parallel_workers'] = value
    with pytest.raises(ValueError, match='parallel_workers'):
        session(config)


class FakeParallelRunner:
    closed = 0
    options = None

    def __init__(self, model, options, workers):
        self.options = options
        self.workers = workers

    def submit(self, seed):
        class Completed:
            def result(self):
                return ([torch.zeros(19, 10, 9)],
                        [torch.ones(1800) / 1800], 0.)
        return Completed()

    def close(self):
        FakeParallelRunner.closed += 1


def test_parallel_workers_play_a_batch_per_advance(tmp_path):
    config = settings(tmp_path)
    config['self_play'].update(num_games_per_iteration=4, parallel_workers=2)
    FakeParallelRunner.closed = 0
    run = session(config, runner_factory=FakeParallelRunner)
    run.start()
    run.advance()
    assert run.games_completed == 2
    assert len(run.replay_buffer.games) == 2
    assert run.phase == 'self_play'
    run.advance()
    assert run.games_completed == 4
    assert len(run.replay_buffer.games) == 4
    assert run.phase == 'train'
    assert run._self_play_runner is None
    assert FakeParallelRunner.closed == 1


def test_iteration_settings_record_parallel_workers(tmp_path):
    from src.training.pipeline import iteration_settings

    config = settings(tmp_path)
    config['self_play']['parallel_workers'] = 4
    assert iteration_settings(config)['self_play']['parallel_workers'] == 4


def test_options_propagate_to_real_trainer_worker_and_evaluator(tmp_path):
    from scripts.train import create_session

    config = settings(tmp_path)
    config['model'] = {'size': 'small', 'num_residual_blocks': 1,
                       'num_channels': 4, 'policy_channels': 1, 'value_channels': 1}
    config['training'] = {'learning_rate': .001, 'weight_decay': .0001,
                          'batch_size': 4, 'use_amp': False}
    config['optimization'] = {'num_workers': 0, 'gradient_accumulation_steps': 2,
                              'max_batch_size': 2}
    config['training_loop']['evaluation_mcts_simulations'] = 7
    config['training_loop']['evaluation_opening_plies'] = 1
    config['training_loop']['evaluation_opening_seed'] = 42
    config['self_play'].update(max_game_plies=2, no_capture_limit=3)
    run = create_session(config)
    try:
        assert run.trainer.gradient_accumulation_steps == 2
        assert run.trainer.max_batch_size == 2
        assert run.trainer.num_workers == 0
        assert run.evaluator.mcts_simulations == 7
        assert run.evaluator.opening_plies == 1
        assert run.evaluator.opening_seed == 42
        run.worker_factory = ControlledWorker
        run.start()
        run.advance()
        assert ControlledWorker.options[-1] == {
            'mcts_simulations': 3, 'temperature': .8, 'temperature_threshold': 4,
            'dirichlet_alpha': .4, 'dirichlet_epsilon': .2,
            'max_game_plies': 2, 'no_capture_limit': 3}
    finally:
        run.trainer.close()


@pytest.mark.parametrize('changed', ['max_game_plies', 'no_capture_limit'])
def test_resume_rejects_changed_rule_limits(tmp_path, changed):
    config = settings(tmp_path)
    run = session(config)
    run.start()
    path = tmp_path / 'rules.pt'
    run.save_progress(str(path))
    config['self_play'][changed] = 3
    restored = session(config)
    with pytest.raises(ValueError, match='rules limits'):
        restored.start(resume=str(path))


def test_resume_requires_full_orchestration_state(tmp_path):
    config = settings(tmp_path)
    trainer = TinyTrainer()
    path = tmp_path / 'legacy.pt'
    trainer.save_checkpoint(path, {})
    run = session(config)
    with pytest.raises(ValueError, match='training_session'):
        run.start(resume=str(path))


def test_short_real_session_round_trip(tmp_path):
    from scripts.train import create_session

    config = settings(tmp_path)
    config['model'] = {'size': 'small', 'num_residual_blocks': 1,
                       'num_channels': 4, 'policy_channels': 1, 'value_channels': 1}
    config['training'] = {'learning_rate': .001, 'weight_decay': .0001,
                          'batch_size': 2, 'use_amp': False}
    config['self_play'].update(num_games_per_iteration=1, mcts_simulations=1,
                               max_game_plies=2, no_capture_limit=3)
    config['training_loop'].update(evaluation_mcts_simulations=1, num_iterations=1)
    torch.set_num_threads(1)
    run = create_session(config)
    try:
        run.start()
        run.run()
        assert run.iteration == 1
        assert len(run.replay_buffer) == 2
        path = str(Path(config['paths']['checkpoint_dir']) / 'checkpoint_progress.pt')
        best_bytes = Path(config['paths']['best_model_path']).read_bytes()
        restored = create_session(config, resume=path)
        try:
            restored.start(resume=path)
            assert restored.iteration == 1
            assert restored.metrics == run.metrics
            assert len(restored.replay_buffer) == 2
            assert restored.trainer.epoch == 1
            assert restored.trainer.optimizer.state_dict()['state']
            assert Path(config['paths']['best_model_path']).read_bytes() == best_bytes
        finally:
            restored.trainer.close()
    finally:
        run.trainer.close()


def test_changed_mid_iteration_game_budget_rejected_on_resume(tmp_path):
    config = settings(tmp_path)
    run = session(config)
    run.start()
    run.advance()
    path = tmp_path / 'budget.pt'
    run.save_progress(str(path))
    config['self_play']['num_games_per_iteration'] = 1
    restored = session(config)
    with pytest.raises(ValueError, match='iteration settings'):
        restored.start(resume=str(path))


def test_explicit_score_threshold_promotes_candidate(tmp_path):
    config = settings(tmp_path)
    del config['training_loop']['win_rate_threshold']
    config['training_loop']['score_threshold'] = .2
    run = session(config)
    run.start()
    metric = run.run_iteration()
    assert metric['accepted']
    assert run.best_model.weight.item() == 2.
    assert run.trainer.saved_models == 2


def test_unknown_or_retired_configuration_options_are_rejected(tmp_path):
    from src.training.pipeline import validate_session_config

    config = settings(tmp_path)
    config['colab'] = {'auto_checkpoint_interval': 7200}
    with pytest.raises(ValueError, match='Unsupported'):
        validate_session_config(config)


@pytest.mark.parametrize('section,key,value', [
    ('self_play', 'num_games_per_iteration', 1.5),
    ('training_loop', 'checkpoint_every', 1.5),
    ('training_loop', 'early_stopping_patience', 1.5),
    ('training_loop', 'auto_checkpoint_interval', float('nan')),
    ('self_play', 'max_game_plies', 0),
    ('self_play', 'mcts_simulations', 0),
    ('self_play', 'dirichlet_epsilon', 1.5),
    ('self_play', 'dirichlet_alpha', float('nan')),
    ('self_play', 'temperature', -1),
    ('training_loop', 'evaluation_mcts_simulations', 0),
    ('training_loop', 'evaluation_opening_plies', -1),
    ('training_loop', 'evaluation_opening_seed', 1.5),
])
def test_invalid_numeric_configuration_is_rejected(tmp_path, section, key, value):
    from src.training.pipeline import validate_session_config

    config = settings(tmp_path)
    config[section][key] = value
    with pytest.raises(ValueError):
        validate_session_config(config)


@pytest.mark.parametrize('occupied', ['checkpoint_dir', 'log_dir'])
def test_occupied_output_directories_refused_before_component_creation(tmp_path, occupied):
    from scripts.train import create_session

    config = settings(tmp_path)
    directory = Path(config['paths'][occupied])
    directory.mkdir()
    (directory / 'keep.bin').write_bytes(b'keep')
    with pytest.raises(FileExistsError):
        create_session(config)
    assert not Path(config['paths']['best_model_path']).exists()
    assert (directory / 'keep.bin').read_bytes() == b'keep'


def test_prechecked_fresh_start_still_refuses_existing_best(tmp_path):
    from src.training.pipeline import TrainingSession

    config = settings(tmp_path)
    run = TrainingSession(TinyTrainer(), TinyReplay(), config,
                          evaluator=RejectingEvaluator(), fresh_paths_checked=True)
    best = Path(config['paths']['best_model_path'])
    best.write_bytes(b'do not replace')
    with pytest.raises(FileExistsError):
        run.start()
    assert best.read_bytes() == b'do not replace'


def test_should_stop_halts_at_game_boundary(tmp_path):
    config = settings(tmp_path)
    run = session(config)
    run.start()
    run.advance()
    run.run(should_stop=lambda: True)
    assert run.games_completed == 1
    assert run.phase == 'self_play'
    assert run.iteration == 0


def test_sparring_unknown_key_rejected_before_start(tmp_path):
    config = settings(tmp_path)
    config['sparring'] = {'enabled': False, 'movetime': 100}
    with pytest.raises(ValueError, match='Unsupported'):
        session(config)


def test_sparring_enabled_without_engine_rejected_before_start(tmp_path):
    config = settings(tmp_path)
    config['sparring'] = {'enabled': True, 'ratio': 0.3, 'depth': 1,
                          'threads': 1, 'hash_mb': 16, 'engine_path': 'missing.bin'}
    with pytest.raises(ValueError, match='engine_path'):
        session(config)


def test_iteration_settings_record_sparring_ratio_and_depth(tmp_path):
    from src.training.pipeline import iteration_settings

    config = settings(tmp_path)
    config['sparring'] = {'enabled': False, 'ratio': 0.25, 'depth': 2}
    assert iteration_settings(config)['sparring'] == {'ratio': 0.25, 'depth': 2}


def test_sparring_worker_alternates_color_when_selected(tmp_path, monkeypatch):
    config = settings(tmp_path)
    binary = tmp_path / 'engine.bin'
    binary.write_bytes(b'x')
    config['sparring'] = {'enabled': True, 'ratio': 1.0, 'depth': 1,
                          'threads': 1, 'hash_mb': 16, 'engine_path': str(binary)}
    colors = []

    class FakePlayer:
        depth = 1

        def close(self):
            return None

    class FakeSparring:
        def __init__(self, **kwargs):
            colors.append(kwargs['opponent_is_red'])

        def play_game(self, verbose=False):
            return ([torch.zeros(19, 10, 9)], [torch.ones(1800) / 1800], 0.)

    monkeypatch.setattr('src.engine.player.load_pikafish_player',
                        lambda **kwargs: FakePlayer())
    monkeypatch.setattr('src.training.sparring_game.SparringWorker', FakeSparring)
    run = session(config)
    run.start()
    run.advance()
    run.advance()
    assert colors == [False, True]


def test_direct_session_default_evaluator_uses_configured_budget(tmp_path):
    from src.training.pipeline import TrainingSession

    config = settings(tmp_path)
    config['training_loop'].update(evaluation_mcts_simulations=7,
                                   evaluation_opening_plies=1, evaluation_opening_seed=42)
    config['self_play'].update(max_game_plies=2, no_capture_limit=3)
    run = TrainingSession(TinyTrainer(), TinyReplay(), config)
    assert run.evaluator.mcts_simulations == 7
    assert run.evaluator.max_game_plies == 2
    assert run.evaluator.no_capture_limit == 3
    assert run.evaluator.opening_plies == 1
    assert run.evaluator.opening_seed == 42
