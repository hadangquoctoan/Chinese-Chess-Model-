"""Resumable AlphaZero orchestration, independent of command-line parsing."""
from copy import deepcopy
import logging
import math
from pathlib import Path
import random
import time
from typing import Any, Callable

import numpy as np
import torch

from ..utils.model_io import (
    atomic_torch_save, build_model_artifact, safe_torch_load, validate_model_artifact,
)
from .evaluator import Evaluator
from .self_play import SelfPlayWorker
from .sparring_config import validate_sparring_config
from .sparring_depth import choose_sparring_depth
from .worker_select import choose_worker_for_game

_LOGGER = logging.getLogger('ChineseChessAI')
MAX_PARALLEL_WORKERS = 16


def _empty_sparring_record() -> dict[int, dict[str, int]]:
    return {1: {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0},
            2: {'wins': 0, 'losses': 0, 'draws': 0, 'games': 0}}


def _copy_sparring_record(payload: Any) -> dict[int, dict[str, int]]:
    record = _empty_sparring_record()
    if not isinstance(payload, dict):
        return record
    for depth in (1, 2):
        entry = payload.get(depth, payload.get(str(depth), {}))
        if not isinstance(entry, dict):
            continue
        for key in record[depth]:
            value = entry.get(key, 0)
            if type(value) is int and value >= 0:
                record[depth][key] = value
    return record


def _finite_number(value: Any, minimum: float = 0., maximum: float = math.inf) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and minimum <= value <= maximum


def _loss_triple(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and len(value) == 3 and all(_finite_number(item) for item in value)


def _validate_metrics(state: dict, config: dict) -> None:
    metrics = state['metrics']
    if not isinstance(metrics, list) or len(metrics) != state['iteration']:
        raise ValueError('metrics must contain one entry per completed iteration')
    budget = config['training_loop']['evaluation_games']
    stale = 0
    for iteration, metric in enumerate(metrics, 1):
        if not isinstance(metric, dict) or metric.get('iteration') != iteration:
            raise ValueError('metrics iteration sequence is invalid')
        counters = [metric.get(key) for key in ('wins', 'losses', 'draws')]
        if any(type(value) is not int or value < 0 for value in counters) or sum(counters) != budget:
            raise ValueError('metrics evaluation counters disagree with budget')
        if not _finite_number(metric.get('score'), maximum=1.):
            raise ValueError('metrics score must be finite in [0, 1]')
        if type(metric.get('accepted')) is not bool or not _loss_triple(metric.get('training_losses')):
            raise ValueError('metrics acceptance/loss fields are invalid')
        stale = 0 if metric['accepted'] else stale + 1
    if state['stale_iterations'] != stale:
        raise ValueError('stale_iterations disagrees with metrics')


def _validate_rng(state: dict) -> None:
    random.Random().setstate(state['python'])
    name, keys, position, gaussian, cached = state['numpy']
    np.random.RandomState().set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.Generator(device='cpu').set_state(state['torch'])
    cuda = state['cuda']
    if not isinstance(cuda, list) or any(not isinstance(value, torch.Tensor)
                                       or value.dtype != torch.uint8 or value.ndim != 1
                                       or not value.numel() for value in cuda):
        raise ValueError('CUDA RNG metadata must be a list of byte tensors')
    if cuda and torch.cuda.is_available():
        if len(cuda) != torch.cuda.device_count():
            raise ValueError('CUDA RNG device count differs from checkpoint')
        for index, value in enumerate(cuda):
            torch.Generator(device=f'cuda:{index}').set_state(value)


def _validate_progress(state: dict, checkpoint: dict, config: dict) -> None:
    for key in ('iteration', 'games_completed', 'stale_iterations'):
        if type(state[key]) is not int or state[key] < 0:
            raise ValueError(f'{key} must be a nonnegative integer')
    for key in ('epoch', 'global_step'):
        if key in checkpoint and (type(checkpoint[key]) is not int or checkpoint[key] < 0):
            raise ValueError(f'{key} must be a nonnegative integer')
    phase = state['phase']
    if phase not in ('self_play', 'train', 'evaluate'):
        raise ValueError('phase must be self_play, train, or evaluate')
    budget = config['self_play']['num_games_per_iteration']
    games = state['games_completed']
    if (phase == 'self_play' and games >= budget) or (phase != 'self_play' and games != budget):
        raise ValueError('phase game counters disagree with self-play budget')
    losses = state['pending_losses']
    if (phase == 'evaluate' and not _loss_triple(losses)) or (phase != 'evaluate' and losses != []):
        raise ValueError('pending_losses must be a finite loss triple only in evaluate phase')
    iteration = state['iteration']
    total = config['training_loop']['num_iterations']
    if iteration > total or (iteration == total and (phase != 'self_play' or games != 0)):
        raise ValueError('iteration/phase exceeds configured budget')
    if checkpoint.get('epoch') != iteration + (phase == 'evaluate'):
        raise ValueError('epoch disagrees with completed iterations and phase')
    if not _finite_number(state['best_score'], maximum=1.):
        raise ValueError('best_score must be finite in [0, 1]')
    _validate_metrics(state, config)


def _validate_session_snapshot(checkpoint: dict, config: dict, model: Any,
                               replay: Any) -> tuple[dict, Any, Any]:
    """Validate temporary objects so rejected resumes never alter live learner state."""
    state = checkpoint.get('training_session') if isinstance(checkpoint, dict) else None
    required = {'best_model', 'replay', 'iteration', 'best_score', 'metrics', 'stale_iterations',
                'phase', 'games_completed', 'pending_losses', 'rng', 'replay_capacity'}
    if not isinstance(state, dict) or type(state.get('version')) is not int or state['version'] != 1:
        raise ValueError('Checkpoint lacks resumable training_session state; legacy weights cannot resume')
    if not required.issubset(state):
        raise ValueError('Incomplete training_session state')
    if state.get('rules_limits') != rules_limits(config):
        raise ValueError('Checkpoint rules limits differ from this configuration; use a fresh run')
    if state.get('iteration_settings') != iteration_settings(config):
        raise ValueError('Checkpoint iteration settings differ from this configuration')
    capacity = config['replay_buffer']['max_size']
    if (type(state['replay_capacity']) is not int or state['replay_capacity'] != capacity
            or getattr(replay, 'max_size', capacity) != capacity
            or not isinstance(state['replay'], dict)
            or state['replay'].get('max_size', capacity) != capacity):
        raise ValueError('Checkpoint replay buffer capacity differs from configured replay capacity')
    try:
        _validate_progress(state, checkpoint, config)
        candidate = deepcopy(model)
        candidate.load_state_dict(state['best_model'])
        if (not _same_weights(candidate.state_dict(), state['best_model'])
                or any(not torch.isfinite(value).all() for value in candidate.state_dict().values())):
            raise ValueError('best_model tensors must be finite with matching keys/dtypes/shapes')
        temporary_replay = deepcopy(replay)
        temporary_replay.load_state_dict(state['replay'])
        _validate_rng(state['rng'])
    except (ValueError, TypeError, KeyError, RuntimeError, OverflowError) as error:
        raise ValueError(f'Invalid training_session snapshot: {error}') from error
    return state, candidate, temporary_replay


def _same_weights(first: dict, second: dict) -> bool:
    return (isinstance(first, dict) and isinstance(second, dict) and first.keys() == second.keys()
            and all(isinstance(value, torch.Tensor) and isinstance(second[key], torch.Tensor)
                    and value.dtype == second[key].dtype and value.shape == second[key].shape
                    and torch.equal(value.cpu(), second[key].cpu()) for key, value in first.items()))


def _validate_best_output(path: Path, state: dict, model: Any,
                          publish_missing: bool) -> bool:
    """A stale resume must not silently downgrade an existing accepted artifact."""
    if not path.exists():
        if not publish_missing:
            raise ValueError('Accepted-best artifact is missing; set publish_missing_best=True '
                             'to explicitly publish the checkpoint incumbent to a new path')
        return True
    message = ('Checkpoint accepted-best differs from the on-disk accepted-best artifact; '
               'choose new output paths to fork this checkpoint')
    try:
        artifact = safe_torch_load(path)
        if hasattr(model, 'config'):
            validate_model_artifact(artifact, model.config)
            if artifact['metadata']['config'] != build_model_artifact(model)['metadata']['config']:
                raise ValueError(message)
            weights = artifact['state_dict']
        else:
            weights = artifact  # Compatibility for simple tensor-only trainer protocols.
        if not _same_weights(weights, state['best_model']):
            raise ValueError(message)
    except (ValueError, OSError, KeyError, TypeError, RuntimeError) as error:
        raise ValueError(message) from error
    return False


def validate_output_paths(config: dict[str, Any]) -> None:
    """Reject equal/nested output targets, resolving aliases before any writes."""
    paths = [(key, Path(config['paths'][key]).resolve())
             for key in ('best_model_path', 'checkpoint_dir', 'log_dir')]
    for index, (name, path) in enumerate(paths):
        for other_name, other in paths[index + 1:]:
            if path == other or path in other.parents or other in path.parents:
                raise ValueError(f'Training output paths overlap: {name} and {other_name}; use separate paths')


def check_fresh_paths(config: dict[str, Any]) -> None:
    """Refuse occupied generated-output locations before creating a trainer."""
    validate_output_paths(config)
    paths = config['paths']
    for key in ('best_model_path', 'checkpoint_dir', 'log_dir'):
        path = Path(paths[key])
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise FileExistsError(f"Fresh training output is occupied: {path}; use new paths or --resume")


def capture_rng_state(saved_cuda: list[torch.Tensor] | None = None) -> dict[str, Any]:
    """Capture safe RNG metadata, retaining dormant CUDA state on CPU-only hosts."""
    name, keys, position, gaussian, cached = np.random.get_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else (saved_cuda or [])
    return {'python': random.getstate(),
            'numpy': [name, keys.tolist(), position, gaussian, cached],
            'torch': torch.get_rng_state(),
            'cuda': [value.cpu().clone() for value in cuda]}


def restore_rng_state(state: dict[str, Any]) -> None:
    """Restore generators after all model and optimizer initialization."""
    random.setstate(state['python'])
    name, keys, position, gaussian, cached = state['numpy']
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), position, gaussian, cached))
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state['cuda']])


def rules_limits(config: dict[str, Any]) -> dict[str, int]:
    """Rules-dependent replay compatibility includes concrete adjudication limits."""
    options = config['self_play']
    return {'max_game_plies': options.get('max_game_plies', 200),
            'no_capture_limit': options.get('no_capture_limit', 100)}


CONFIG_FIELDS = {
    'model': {'size', 'num_residual_blocks', 'num_channels', 'policy_channels', 'value_channels'},
    'training': {'learning_rate', 'weight_decay', 'batch_size', 'use_amp'},
    'self_play': {'num_games_per_iteration', 'mcts_simulations', 'temperature',
                  'temperature_threshold', 'dirichlet_alpha', 'dirichlet_epsilon',
                  'parallel_workers', 'max_game_plies', 'no_capture_limit'},
    'replay_buffer': {'max_size'},
    'training_loop': {'num_iterations', 'samples_per_iteration', 'evaluation_games',
                      'score_threshold', 'win_rate_threshold', 'checkpoint_every',
                      'auto_checkpoint_interval', 'early_stopping_patience',
                      'evaluation_mcts_simulations', 'evaluation_opening_plies',
                      'evaluation_opening_seed'},
    'optimization': {'gradient_accumulation_steps', 'max_batch_size', 'num_workers'},
    'paths': {'checkpoint_dir', 'best_model_path', 'log_dir'},
    'sparring': {'enabled', 'ratio', 'depth', 'threads', 'hash_mb', 'engine_path'},
}


def score_threshold(config: dict[str, Any]) -> float:
    """Read half-draw match score threshold; win_rate_threshold is a legacy alias."""
    loop = config['training_loop']
    if 'score_threshold' in loop and 'win_rate_threshold' in loop:
        raise ValueError('Specify score_threshold, not both threshold names')
    return loop['score_threshold'] if 'score_threshold' in loop else loop['win_rate_threshold']


def iteration_settings(config: dict[str, Any]) -> dict[str, Any]:
    """Record budgets whose changes could corrupt a partially resumed iteration."""
    loop = config['training_loop']
    sparring = validate_sparring_config(config.get('sparring'))
    workers = config['self_play'].get('parallel_workers', 1)
    return {'self_play': {**config['self_play'], **rules_limits(config),
                          'parallel_workers': workers},
            'samples_per_iteration': loop['samples_per_iteration'],
            'evaluation_games': loop['evaluation_games'],
            'evaluation_mcts_simulations': loop.get('evaluation_mcts_simulations', 200),
            'evaluation_opening_plies': loop.get('evaluation_opening_plies', 4),
            'evaluation_opening_seed': loop.get('evaluation_opening_seed', 0),
            'score_threshold': score_threshold(config),
            'sparring': {'ratio': sparring['ratio'], 'depth': sparring['depth']}}


def validate_session_config(config: dict[str, Any]) -> None:
    """Reject invalid budgets and options that are not implemented."""
    for section, options in config.items():
        if section not in CONFIG_FIELDS or not isinstance(options, dict):
            raise ValueError(f'Unsupported configuration section: {section}')
        unknown = set(options) - CONFIG_FIELDS[section]
        if unknown:
            raise ValueError(f'Unsupported {section} options: {sorted(unknown)}')
    validate_output_paths(config)
    workers = config['self_play'].get('parallel_workers', 1)
    if type(workers) is not int or not 1 <= workers <= MAX_PARALLEL_WORKERS:
        raise ValueError(
            f'self_play.parallel_workers must be an integer in [1, {MAX_PARALLEL_WORKERS}]')
    loop = config['training_loop']
    for key in ('num_iterations', 'samples_per_iteration', 'evaluation_games', 'checkpoint_every'):
        value = loop.get(key, 5) if key == 'checkpoint_every' else loop[key]
        if type(value) is not int or value <= 0:
            raise ValueError(f'training_loop.{key} must be a positive integer')
    if loop['evaluation_games'] % 2:
        raise ValueError('training_loop.evaluation_games must be even for color-balanced evaluation')
    patience = loop.get('early_stopping_patience', 0)
    if type(patience) is not int or patience < 0:
        raise ValueError('training_loop.early_stopping_patience must be a nonnegative integer')
    interval = loop.get('auto_checkpoint_interval', 0)
    if not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 0:
        raise ValueError('training_loop.auto_checkpoint_interval must be finite and nonnegative')
    games = config['self_play']['num_games_per_iteration']
    if type(games) is not int or games <= 0:
        raise ValueError('self_play.num_games_per_iteration must be a positive integer')
    for key, value in rules_limits(config).items():
        if type(value) is not int or value <= 0:
            raise ValueError(f'self_play.{key} must be a positive integer')
    _validate_search_options(config)
    if not 0 <= score_threshold(config) <= 1:
        raise ValueError('training_loop.score_threshold must be between 0 and 1')
    validate_sparring_config(config.get('sparring'))


def _validate_search_options(config: dict[str, Any]) -> None:
    self_play, loop = config['self_play'], config['training_loop']
    integers = [(self_play.get('mcts_simulations', 400), 1),
                (self_play.get('temperature_threshold', 15), 0),
                (loop.get('evaluation_mcts_simulations', 200), 1),
                (loop.get('evaluation_opening_plies', 4), 0)]
    if any(type(value) is not int or value < minimum for value, minimum in integers):
        raise ValueError('Search simulation/ply budgets must be valid integers')
    if type(loop.get('evaluation_opening_seed', 0)) is not int:
        raise ValueError('evaluation_opening_seed must be an integer')
    for key, default, minimum, maximum in [('temperature', 1., 0., math.inf),
                                          ('dirichlet_alpha', .3, 0., math.inf),
                                          ('dirichlet_epsilon', .25, 0., 1.)]:
        value = self_play.get(key, default)
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not minimum <= value <= maximum:
            raise ValueError(f'self_play.{key} outside valid range')
        if key == 'dirichlet_alpha' and value == 0:
            raise ValueError('dirichlet_alpha must be positive')


class TrainingSession:
    """Train a continuous learner; only accepted incumbents generate self-play.

    Rejection does not roll back learner weights or optimizer. The separate
    incumbent remains frozen until the evaluation gate accepts a candidate.
    """

    def __init__(self, trainer: Any, replay_buffer: Any, config: dict[str, Any],
                 worker_factory: Callable = SelfPlayWorker,
                 evaluator: Any = None,
                 clock: Callable[[], float] = time.monotonic,
                 fresh_paths_checked: bool = False,
                 runner_factory: Callable | None = None) -> None:
        validate_session_config(config)
        self.trainer = trainer
        self.replay_buffer = replay_buffer
        self.config = config
        self.worker_factory = worker_factory
        self.fresh_paths_checked = fresh_paths_checked
        loop = config['training_loop']
        self.evaluator = evaluator if evaluator is not None else Evaluator(
            mcts_simulations=loop.get('evaluation_mcts_simulations', 200),
            opening_plies=loop.get('evaluation_opening_plies', 4),
            opening_seed=loop.get('evaluation_opening_seed', 0), **rules_limits(config))
        self.best_model = deepcopy(trainer.model)
        self.iteration = 0
        self.best_score = 0.
        self.metrics = []
        self.stale_iterations = 0
        self.clock = clock
        self.last_checkpoint = clock()
        self.phase = 'self_play'
        self.games_completed = 0
        self.pending_losses = []
        self._worker = None
        self._self_play_runner = None
        self.runner_factory = runner_factory
        self._sparring_player = None
        self._opponent_is_red = False
        self._sparring_record = _empty_sparring_record()
        self._iteration_sparring = _empty_sparring_record()
        self._saved_cuda_rng = []

    def start(self, resume: str | None = None, *, publish_missing_best: bool = False) -> None:
        """Restore a session; publishing a missing incumbent requires explicit opt-in."""
        validate_output_paths(self.config)
        if resume is not None:
            checkpoint = safe_torch_load(resume, 'cpu')
            state, incumbent, replay = _validate_session_snapshot(
                checkpoint, self.config, self.best_model, self.replay_buffer)
            validator = getattr(self.trainer, 'validate_checkpoint', None)
            if callable(validator):
                validator(checkpoint)
            best_path = Path(self.config['paths']['best_model_path'])
            publish_best = _validate_best_output(best_path, state, incumbent, publish_missing_best)
            if callable(validator):
                self.trainer.load_checkpoint(resume, checkpoint=checkpoint)
            else:
                self.trainer.load_checkpoint(resume)
            self.best_model.load_state_dict(incumbent.state_dict())
            self.replay_buffer.load_state_dict(replay.state_dict())
            self.iteration = state['iteration']
            self.best_score = state['best_score']
            self.metrics = state['metrics']
            self.stale_iterations = state['stale_iterations']
            self.phase = state['phase']
            self.games_completed = state['games_completed']
            self.pending_losses = state['pending_losses']
            self._opponent_is_red = bool(state.get('opponent_is_red', False))
            self._sparring_record = _copy_sparring_record(state.get('sparring_record'))
            self._iteration_sparring = _copy_sparring_record(state.get('iteration_sparring'))
            self._saved_cuda_rng = [value.cpu().clone() for value in state['rng']['cuda']]
            restore_rng_state(state['rng'])
            if publish_best:
                artifact = (build_model_artifact(self.best_model) if hasattr(self.best_model, 'config')
                            else self.best_model.state_dict())
                atomic_torch_save(artifact, best_path)
            return
        if not self.fresh_paths_checked:
            check_fresh_paths(self.config)
        elif Path(self.config['paths']['best_model_path']).exists():
            raise FileExistsError('Fresh best-model output is occupied; use new paths or --resume')
        self.trainer.save_model(self.config['paths']['best_model_path'])

    def save_progress(self, path: str) -> None:
        """Persist learner/optimizer plus accepted incumbent, replay, and progress."""
        state = {'version': 1, 'best_model': self.best_model.state_dict(),
                 'replay': self.replay_buffer.state_dict(), 'iteration': self.iteration,
                 'best_score': self.best_score, 'metrics': self.metrics,
                 'stale_iterations': self.stale_iterations, 'rng': capture_rng_state(self._saved_cuda_rng),
                 'phase': self.phase, 'games_completed': self.games_completed,
                 'pending_losses': self.pending_losses, 'rules_limits': rules_limits(self.config),
                 'replay_capacity': self.config['replay_buffer']['max_size'],
                 'iteration_settings': iteration_settings(self.config),
                 'opponent_is_red': self._opponent_is_red,
                 'sparring_record': self._sparring_record,
                 'iteration_sparring': self._iteration_sparring}
        self.trainer.save_checkpoint(path, additional_info={'training_session': state})
        self.last_checkpoint = self.clock()

    def _checkpoint_boundary(self) -> None:
        interval = self.config['training_loop'].get('auto_checkpoint_interval', 0)
        if interval and self.clock() - self.last_checkpoint >= interval:
            path = Path(self.config['paths']['checkpoint_dir']) / 'checkpoint_progress.pt'
            self.save_progress(str(path))

    def _self_play_options(self) -> dict[str, Any]:
        return {key: value for key, value in self.config['self_play'].items()
                if key not in ('num_games_per_iteration', 'parallel_workers')}

    def close(self) -> None:
        """Close the self-play pool and any persistent Pikafish subprocess."""
        self._close_self_play_runner()
        self._close_sparring_player()

    def _close_self_play_runner(self) -> None:
        runner = self._self_play_runner
        self._self_play_runner = None
        if runner is not None:
            runner.close()

    def _ensure_self_play_runner(self):
        if self._self_play_runner is None:
            workers = self.config['self_play'].get('parallel_workers', 1)
            factory = self.runner_factory
            if factory is None:
                from .parallel_self_play import ParallelSelfPlayRunner
                factory = ParallelSelfPlayRunner
            self._self_play_runner = factory(
                self.best_model, self._self_play_options(), workers)
        return self._self_play_runner

    def _close_sparring_player(self) -> None:
        player = self._sparring_player
        self._sparring_player = None
        if player is not None:
            player.close()

    def _ensure_sparring_player(self):
        if self._sparring_player is None:
            from ..engine.player import load_pikafish_player
            cfg = validate_sparring_config(self.config.get('sparring'))
            path = Path(cfg['engine_path']) if cfg['engine_path'] else None
            self._sparring_player = load_pikafish_player(
                depth=cfg['depth'], path=path, threads=cfg['threads'], hash_mb=cfg['hash_mb'])
        return self._sparring_player

    def _log_sparring_game(self, depth: int, opponent_is_red: bool,
                           winner: float, plies: int) -> None:
        model_is_red = not opponent_is_red
        if winner == 0.0:
            result = 'draw'
        elif (winner == 1.0 and model_is_red) or (winner == -1.0 and not model_is_red):
            result = 'win'
        else:
            result = 'loss'
        color = 'red' if model_is_red else 'black'
        _LOGGER.info('sparring | depth %s | model %s | result %s | plies %s',
                     depth, color, result, plies)
        bucket = depth if depth in self._sparring_record else 1
        for record in (self._sparring_record[bucket], self._iteration_sparring[bucket]):
            record['games'] += 1
            record[{'win': 'wins', 'loss': 'losses', 'draw': 'draws'}[result]] += 1

    def _log_sparring_record(self) -> None:
        r1, r2 = self._iteration_sparring[1], self._iteration_sparring[2]
        if r1['games'] or r2['games']:
            _LOGGER.info('sparring record: %sW-%sL depth1, %sW-%sL depth2',
                         r1['wins'], r1['losses'], r2['wins'], r2['losses'])
        self._iteration_sparring = _empty_sparring_record()

    def _play_sparring_game(self) -> tuple[list[np.ndarray], list[np.ndarray], float]:
        from .sparring_game import SparringWorker
        player = self._ensure_sparring_player()
        depth = choose_sparring_depth(
            random, self._sparring_record[1]['wins'], self._sparring_record[1]['games'],
            self._sparring_record[2]['wins'], self._sparring_record[2]['games'])
        player.depth = depth
        opponent_is_red = self._opponent_is_red
        worker = SparringWorker(model=self.best_model, opponent=player,
                                opponent_is_red=opponent_is_red,
                                **self._self_play_options())
        states, policies, winner = worker.play_game(verbose=False)
        self._log_sparring_game(depth, opponent_is_red, winner, len(states))
        self._opponent_is_red = not opponent_is_red
        return states, policies, winner

    def _play_generation_game(self) -> None:
        sparring = validate_sparring_config(self.config.get('sparring'))
        kind = choose_worker_for_game(random, sparring['ratio'], sparring['enabled'])
        if kind != 'sparring':
            if self._worker is None:
                self._worker = self.worker_factory(
                    model=self.best_model, **self._self_play_options())
            self.replay_buffer.add_game(*self._worker.play_game(verbose=False))
            return
        self.replay_buffer.add_game(*self._play_sparring_game())

    def _play_generation_batch(self) -> None:
        """Play up to ``parallel_workers`` games, overlapping GPU and Pikafish work."""
        total = self.config['self_play']['num_games_per_iteration']
        workers = self.config['self_play'].get('parallel_workers', 1)
        count = min(workers, total - self.games_completed)
        sparring = validate_sparring_config(self.config.get('sparring'))
        kinds = [choose_worker_for_game(random, sparring['ratio'], sparring['enabled'])
                 for _ in range(count)]
        self_play_positions = [index for index, kind in enumerate(kinds)
                               if kind != 'sparring']
        runner = self._ensure_self_play_runner() if self_play_positions else None
        futures = []
        if runner is not None:
            futures = [(index, runner.submit(random.getrandbits(32)))
                       for index in self_play_positions]
        results: dict[int, tuple] = {}
        for index, kind in enumerate(kinds):
            if kind == 'sparring':
                results[index] = self._play_sparring_game()
        for index, future in futures:
            results[index] = future.result()
        for index in range(count):
            self.replay_buffer.add_game(*results[index])
        self.games_completed += count
        if self.games_completed == total:
            self._finish_generation_phase()

    def _finish_generation_phase(self) -> None:
        self._log_sparring_record()
        self._close_sparring_player()
        self._close_self_play_runner()
        self.phase = 'train'

    def advance(self) -> dict[str, Any] | None:
        """Execute one whole game or batch, epoch, or evaluation and checkpoint if due."""
        if self.phase == 'self_play':
            if self.config['self_play'].get('parallel_workers', 1) > 1:
                self._play_generation_batch()
            else:
                self._play_generation_game()
                self.games_completed += 1
                if self.games_completed == self.config['self_play']['num_games_per_iteration']:
                    self._finish_generation_phase()
        elif self.phase == 'train':
            self.pending_losses = list(self.trainer.train_epoch(
                self.replay_buffer, self.config['training_loop']['samples_per_iteration']))
            self.phase = 'evaluate'
        else:
            metrics = self._evaluate()
            self._checkpoint_boundary()
            return metrics
        self._checkpoint_boundary()
        return None

    def run_iteration(self) -> dict[str, Any]:
        """Finish the current iteration, including a partially resumed one."""
        result = None
        while result is None:
            result = self.advance()
        return result

    def run(self, on_iteration: Callable | None = None,
            should_stop: Callable[[], bool] | None = None) -> None:
        """Run until configured total iterations, a stop callback, or rejection limit."""
        loop = self.config['training_loop']
        directory = Path(self.config['paths']['checkpoint_dir'])
        patience = loop.get('early_stopping_patience', 0)
        try:
            while self.iteration < loop['num_iterations']:
                if patience and self.stale_iterations >= patience:
                    break
                if should_stop is not None and should_stop():
                    break
                result = None
                while result is None:
                    if should_stop is not None and should_stop():
                        break
                    result = self.advance()
                if result is None:
                    break
                if self.iteration % loop.get('checkpoint_every', 5) == 0:
                    self.save_progress(str(directory / f'checkpoint_iter_{self.iteration}.pt'))
                if on_iteration is not None:
                    on_iteration(result)
            self.save_progress(str(directory / 'checkpoint_progress.pt'))
        finally:
            self._close_sparring_player()

    def _evaluate(self) -> dict[str, Any]:
        loop = self.config['training_loop']
        score, wins, losses_count, draws = self.evaluator.evaluate(
            new_model=self.trainer.model, best_model=self.best_model,
            num_games=loop['evaluation_games'], verbose=False)
        accepted = score > score_threshold(self.config)
        if accepted:
            self.best_model.load_state_dict(self.trainer.model.state_dict())
            self.best_score = score
            self.stale_iterations = 0
            self.trainer.save_model(self.config['paths']['best_model_path'])
        else:
            self.stale_iterations += 1
        self.iteration += 1
        metrics = {'iteration': self.iteration, 'score': score, 'wins': wins,
                   'losses': losses_count, 'draws': draws, 'accepted': accepted,
                   'training_losses': self.pending_losses}
        self.metrics.append(metrics)
        self.phase = 'self_play'
        self.games_completed = 0
        self.pending_losses = []
        self._worker = None
        self._close_self_play_runner()
        self._close_sparring_player()
        return metrics
