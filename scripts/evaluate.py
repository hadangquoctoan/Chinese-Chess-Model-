import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.neural_network import ChineseChessNet
from src.model.model_config import ModelConfig
from src.training.evaluator import Evaluator


def main() -> None:
    import torch

    parser = argparse.ArgumentParser(description='Evaluate Chinese Chess AI model')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model file (.pt)')
    parser.add_argument('--games', type=int, default=100,
                      help='Number of games to play')
    parser.add_argument('--mcts-sims', type=int, default=200,
                      help='MCTS simulations')
    args = parser.parse_args()
    if args.games <= 0:
        parser.error('--games must be positive')
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = ModelConfig.from_size('medium')
    model = ChineseChessNet(config)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Playing {args.games} games...")
    
    # Setup evaluator
    evaluator = Evaluator(mcts_simulations=args.mcts_sims)
    
    # Evaluate (model plays against itself)
    win_rate, wins, losses, draws = evaluator.evaluate(
        new_model=model,
        best_model=model,
        num_games=args.games,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Total games: {args.games}")
    print("Results:")
    print(f"  Red wins:  {wins} ({wins/args.games:.1%})")
    print(f"  Black wins: {losses} ({losses/args.games:.1%})")
    print(f"  Draws:     {draws} ({draws/args.games:.1%})")
    print("="*60)

if __name__ == '__main__':
    main()
