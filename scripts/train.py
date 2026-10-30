"""Command-line entry point for resumable AlphaZero training."""
import argparse
from pathlib import Path
import signal
import sys
import time
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet

from src.training.pipeline import TrainingSession, check_fresh_paths, validate_session_config
from src.training.replay_buffer import ReplayBuffer
from src.training.trainer import Trainer
from src.utils.logger import setup_logger
from src.utils.model_io import ARCHITECTURE_FIELDS, load_model_artifact

MODEL_CONFIG_FIELDS = ('num_residual_blocks', 'num_channels', 'policy_channels', 'value_channels')


def load_config(config_path: str) -> dict[str, Any]:
    """Load a YAML mapping; validation happens before output files are created."""
    with open(config_path, 'r', encoding='utf-8') as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError(f'Configuration must be a YAML mapping: {config_path}')
    return config


def build_model_config(config: dict[str, Any]) -> ModelConfig:
    """Combine presets and training settings; ``rules`` overrides ``self_play``."""
    model_config = ModelConfig.from_size(config['model']['size'])
    for field in MODEL_CONFIG_FIELDS:
        if field in config['model']:
            setattr(model_config, field, config['model'][field])
    for field in ('learning_rate', 'weight_decay', 'batch_size', 'use_amp'):
        setattr(model_config, field, config['training'][field])
    rules = config.get('rules', config.get('self_play', {}))
    model_config.max_game_plies = rules.get('max_game_plies', 200)
    model_config.no_capture_limit = rules.get('no_capture_limit', 100)
    model_config.validate()
    return model_config


def create_session(config: dict[str, Any], resume: str | None = None,
                   init_model: str | None = None) -> TrainingSession:
    """Build configured components, checking fresh paths before opening logs."""
    validate_session_config(config)
    if resume is None:
        check_fresh_paths(config)
    model_config = build_model_config(config)
    model = ChineseChessNet(model_config)
    if init_model is not None:
        loaded = load_model_artifact(init_model)
        mismatch = [name for name in ARCHITECTURE_FIELDS
                    if getattr(loaded.config, name) != getattr(model_config, name)]
        if mismatch:
            raise ValueError(f'init-model architecture mismatch: {", ".join(mismatch)}')
        model.load_state_dict(loaded.state_dict())
    options = config.get('optimization', {})
    trainer = Trainer(
        model=model, config=model_config,
        checkpoint_dir=config['paths']['checkpoint_dir'], log_dir=config['paths']['log_dir'],
        num_workers=options.get('num_workers', 0),
        gradient_accumulation_steps=options.get('gradient_accumulation_steps', 1),
        max_batch_size=options.get('max_batch_size'))
    try:
        return TrainingSession(trainer, ReplayBuffer(config['replay_buffer']['max_size']),
                               config, fresh_paths_checked=resume is None)
    except (ValueError, TypeError, KeyError):
        trainer.close()
        raise


def main(argv: list[str] | None = None) -> None:
    """Run training; resume means total configured iterations, not extra ones."""
    parser = argparse.ArgumentParser(description='Train Chinese Chess AI')
    parser.add_argument('--config', default='configs/training_config.yaml')
    parser.add_argument('--resume', help='Full session checkpoint, not a weights-only artifact')
    parser.add_argument('--init-model', help='Versioned weights to seed a fresh run (not a resume)')
    parser.add_argument('--max-seconds', type=float, default=0,
                        help='Stop after this many seconds at a game/epoch/eval boundary; 0 disables')
    parser.add_argument('--publish-missing-best', action='store_true',
                        help='With --resume and new output paths, publish the checkpoint incumbent')
    args = parser.parse_args(argv)
    if args.publish_missing_best and not args.resume:
        parser.error('--publish-missing-best requires --resume')
    if args.init_model and args.resume:
        parser.error('--init-model cannot be combined with --resume')
    if args.max_seconds < 0:
        parser.error('--max-seconds must be nonnegative')
    config = load_config(args.config)
    session = create_session(config, args.resume, init_model=args.init_model)
    try:
        session.start(resume=args.resume, publish_missing_best=args.publish_missing_best)
        logger = setup_logger(log_dir=config['paths']['log_dir'])
        if 'win_rate_threshold' in config['training_loop']:
            logger.warning('win_rate_threshold is deprecated: interpreted as half-draw match score; use score_threshold')
        logger.info('Continuous learner; only accepted incumbent generates self-play')
        deadline = time.monotonic() + args.max_seconds if args.max_seconds else None
        stop = {'flag': False}

        def request_stop(signum, frame):
            stop['flag'] = True
            logger.info('Stop requested; finishing current game/epoch/eval')

        def should_stop() -> bool:
            if stop['flag']:
                return True
            return deadline is not None and time.monotonic() >= deadline

        previous_sigint = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, request_stop)
        try:
            session.run(on_iteration=lambda metrics: logger.info('Iteration metrics: %s', metrics),
                        should_stop=should_stop)
        finally:
            signal.signal(signal.SIGINT, previous_sigint)
        logger.info('Stopped at iteration %s; incumbent gate score %s',
                    session.iteration, session.best_score)
        if stop['flag']:
            raise SystemExit(130)
    finally:
        session.close()
        session.trainer.close()


if __name__ == '__main__':
    main()