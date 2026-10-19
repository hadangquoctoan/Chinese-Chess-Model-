import argparse
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.utils.model_io import load_model_artifact
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
    if args.games <= 0 or args.games % 2:
        parser.error('--games must be positive and even for paired-color evaluation')
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = load_model_artifact(args.model, device=device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Self-play sanity check: {args.games} games (not a strength benchmark)")
    
    # Setup evaluator
    evaluator = Evaluator(mcts_simulations=args.mcts_sims,
                          max_game_plies=model.config.max_game_plies,
                          no_capture_limit=model.config.no_capture_limit)
    
    # Evaluate (model plays against itself)
    score, wins, losses, draws = evaluator.evaluate(
        new_model=model,
        best_model=model,
        num_games=args.games,
        verbose=True
    )
    
    print("\n" + "="*60)
    print("Evaluation completed!")
    print(f"Total games: {args.games}")
    print("Results:")
    print(f"  Candidate-role wins: {wins} ({wins/args.games:.1%})")
    print(f"  Candidate-role losses: {losses} ({losses/args.games:.1%})")
    print(f"  Score (half draws): {score:.1%}")
    print(f"  Draws:     {draws} ({draws/args.games:.1%})")
    print("="*60)

if __name__ == '__main__':
    main()
