from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.neural_network import ChineseChessNet
from src.model.model_config import ModelConfig
from src.training.self_play import SelfPlayWorker
from src.training.trainer import Trainer
from src.training.replay_buffer import ReplayBuffer
from src.game.game_state import GameState


def quick_demo() -> None:
    """Demo nhanh để test toàn bộ pipeline"""
    
    print("="*60)
    print("Chinese Chess AI - Quick Demo")
    print("="*60)
    
    # 1. Create model
    print("\n[1/5] Creating model...")
    config = ModelConfig.from_size('small')  # Small model for quick test
    model = ChineseChessNet(config)
    print(f"[OK] Model created: {model.count_parameters():,} parameters")
    
    # 2. Test game
    print("\n[2/5] Testing game logic...")
    game = GameState()
    print(game)
    legal_moves = game.get_legal_moves()
    print(f"[OK] {len(legal_moves)} legal moves available")
    
    # 3. Generate self-play game
    print("\n[3/5] Generating self-play game (this may take 1-2 minutes)...")
    worker = SelfPlayWorker(model, mcts_simulations=50)  # Low sims for speed
    states, policies, winner = worker.play_game(verbose=True)
    print(f"[OK] Game generated: {len(states)} positions, winner: {winner}")
    
    # 4. Create replay buffer
    print("\n[4/5] Creating replay buffer...")
    replay_buffer = ReplayBuffer(max_size=10000)
    replay_buffer.add_game(states, policies, winner)
    print(f"[OK] Replay buffer: {len(replay_buffer)} positions")
    
    # 5. Train for 1 epoch
    print("\n[5/5] Training for 1 epoch (this may take 1-2 minutes)...")
    trainer = Trainer(model, config)
    p_loss, v_loss, total_loss = trainer.train_epoch(
        replay_buffer, 
        num_samples=min(1000, len(replay_buffer)),
        batch_size=64
    )
    print("[OK] Training completed!")
    print(f"   Policy Loss: {p_loss:.4f}")
    print(f"   Value Loss: {v_loss:.4f}")
    print(f"   Total Loss: {total_loss:.4f}")
    
    # Save test model
    print("\n[Bonus] Saving test model...")
    Path("models").mkdir(exist_ok=True)
    trainer.save_model("models/test_model.pt")
    print("[OK] Model saved: models/test_model.pt")
    
    print("\n" + "="*60)
    print("Quick demo completed successfully!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run tests: python tests/test_game.py")
    print("2. Train on Colab: Open notebooks/02_training_colab.ipynb")
    print("3. Play against AI: python scripts/play_cli.py --model models/best_model.pt")

if __name__ == '__main__':
    quick_demo()
