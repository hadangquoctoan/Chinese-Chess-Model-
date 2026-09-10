import argparse
from pathlib import Path
import sys
from typing import Any, Dict

import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.neural_network import ChineseChessNet
from src.model.model_config import ModelConfig
from src.training.trainer import Trainer
from src.training.self_play import SelfPlayWorker
from src.training.replay_buffer import ReplayBuffer
from src.training.evaluator import Evaluator
from src.utils.logger import setup_logger

MODEL_CONFIG_FIELDS = (
    'num_residual_blocks',
    'num_channels',
    'policy_channels',
    'value_channels',
)


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Configuration must be a YAML mapping: {config_path}")
    return config


def build_model_config(config: Dict[str, Any]) -> ModelConfig:
    """Build a model configuration from the project YAML settings."""
    model_settings = config['model']
    training_settings = config['training']
    model_config = ModelConfig.from_size(model_settings['size'])

    for field in MODEL_CONFIG_FIELDS:
        if field in model_settings:
            setattr(model_config, field, model_settings[field])

    model_config.learning_rate = training_settings['learning_rate']
    model_config.weight_decay = training_settings['weight_decay']
    model_config.batch_size = training_settings['batch_size']
    model_config.use_amp = training_settings['use_amp']
    return model_config


def main() -> None:
    parser = argparse.ArgumentParser(description='Train Chinese Chess AI')
    parser.add_argument('--config', type=str, default='configs/training_config.yaml',
                      help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                      help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Setup logger
    logger = setup_logger(log_dir=config['paths'].get('log_dir', 'logs'))
    logger.info("Starting Chinese Chess AI Training")
    logger.info(f"Config: {args.config}")
    
    model_config = build_model_config(config)
    
    logger.info(f"\n{model_config}")
    
    model = ChineseChessNet(model_config)
    logger.info(f"\n{model}")
    
    # Setup trainer
    trainer = Trainer(
        model=model,
        config=model_config,
        checkpoint_dir=config['paths']['checkpoint_dir'],
        log_dir=config['paths']['log_dir']
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
        logger.info(f"Resumed from checkpoint: {args.resume}")
    
    # Create replay buffer
    replay_buffer = ReplayBuffer(max_size=config['replay_buffer']['max_size'])
    
    # Setup evaluator
    evaluator = Evaluator(mcts_simulations=200)
    
    # Keep the evaluation baseline on the same device as the trained model.
    best_model = ChineseChessNet(model_config).to(trainer.device)
    best_model.load_state_dict(model.state_dict())
    best_win_rate = 0.0
    
    # Training loop
    num_iterations = config['training_loop']['num_iterations']
    checkpoint_every = config['training_loop'].get('checkpoint_every', 5)
    if checkpoint_every <= 0:
        raise ValueError("training_loop.checkpoint_every must be positive")

    # Create the documented best-model artifact even if no iteration improves it.
    trainer.save_model(config['paths']['best_model_path'])
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting training loop for {num_iterations} iterations")
    logger.info(f"{'='*60}\n")
    
    for iteration in range(num_iterations):
        logger.info(f"\n{'='*60}")
        logger.info(f"Iteration {iteration + 1}/{num_iterations}")
        logger.info(f"{'='*60}")
        
        # Phase 1: Self-play
        logger.info("\n[Phase 1] Generating self-play games...")
        self_play_worker = SelfPlayWorker(
            model=model,
            mcts_simulations=config['self_play']['mcts_simulations'],
            temperature=config['self_play']['temperature'],
            temperature_threshold=config['self_play']['temperature_threshold']
        )
        
        games = self_play_worker.generate_games(
            num_games=config['self_play']['num_games_per_iteration'],
            verbose=True
        )
        
        # Add games to replay buffer
        for states, policies, winner in games:
            replay_buffer.add_game(states, policies, winner)
        
        logger.info(f"Replay buffer size: {len(replay_buffer)}")
        buffer_stats = replay_buffer.get_stats()
        logger.info(f"Buffer stats: {buffer_stats}")
        
        # Phase 2: Training
        logger.info("\n[Phase 2] Training neural network...")
        p_loss, v_loss, total_loss = trainer.train_epoch(
            replay_buffer=replay_buffer,
            num_samples=config['training_loop']['samples_per_iteration']
        )
        
        logger.info(f"Training losses: Policy={p_loss:.4f}, Value={v_loss:.4f}, Total={total_loss:.4f}")
        
        # Phase 3: Evaluation
        logger.info("\n[Phase 3] Evaluating new model...")
        win_rate, wins, losses, draws = evaluator.evaluate(
            new_model=model,
            best_model=best_model,
            num_games=config['training_loop']['evaluation_games'],
            verbose=True
        )
        
        # Update best model if new model is better
        if win_rate > config['training_loop']['win_rate_threshold']:
            logger.info(
                "\nNew best model: "
                f"win rate {win_rate:.1%} > {config['training_loop']['win_rate_threshold']:.1%}"
            )
            best_model.load_state_dict(model.state_dict())
            best_win_rate = win_rate
            
            # Save best model
            best_model_path = config['paths']['best_model_path']
            trainer.save_model(best_model_path)
            logger.info(f"Best model saved: {best_model_path}")
        else:
            logger.info(f"\nModel not improved. Win rate: {win_rate:.1%} <= {config['training_loop']['win_rate_threshold']:.1%}")
        
        # Save checkpoint periodically
        if (iteration + 1) % checkpoint_every == 0:
            checkpoint_path = Path(config['paths']['checkpoint_dir']) / f"checkpoint_iter_{iteration+1}.pt"
            trainer.save_checkpoint(
                str(checkpoint_path),
                additional_info={
                    'iteration': iteration + 1,
                    'win_rate': win_rate,
                    'best_win_rate': best_win_rate
                }
            )
        
        logger.info(f"\nIteration {iteration + 1} completed.")
        logger.info(f"Current best win rate: {best_win_rate:.1%}")
    
    # Training completed
    logger.info(f"\n{'='*60}")
    logger.info("Training completed!")
    logger.info(f"Best model win rate: {best_win_rate:.1%}")
    logger.info(f"Best model saved at: {config['paths']['best_model_path']}")
    logger.info(f"{'='*60}\n")
    
    trainer.close()

if __name__ == '__main__':
    main()
