import argparse
from pathlib import Path
import sys
from typing import Optional, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.game_state import GameState


Move = Tuple[int, int, int, int]


def parse_move(move_str: str) -> Optional[Move]:
    """Parse move string like 'a0-a1' to (from_row, from_col, to_row, to_col)"""
    normalized_move = move_str.strip().lower()
    if len(normalized_move) != 5 or normalized_move[2] != '-':
        return None

    from_col_char, from_row_char, _, to_col_char, to_row_char = normalized_move
    if not ('a' <= from_col_char <= 'i' and 'a' <= to_col_char <= 'i'):
        return None
    if not (from_row_char.isdigit() and to_row_char.isdigit()):
        return None

    return (
        int(from_row_char),
        ord(from_col_char) - ord('a'),
        int(to_row_char),
        ord(to_col_char) - ord('a'),
    )


def main() -> None:
    import torch
    from src.model.mcts import MCTS
    from src.model.model_config import ModelConfig
    from src.model.neural_network import ChineseChessNet

    parser = argparse.ArgumentParser(description='Play Chinese Chess against AI')
    parser.add_argument('--model', type=str, required=True,
                      help='Path to model file (.pt)')
    parser.add_argument('--mcts-sims', type=int, default=400,
                      help='MCTS simulations (more = stronger but slower)')
    parser.add_argument('--human-color', type=str, default='red',
                      choices=['red', 'black'],
                      help='Color for human player')
    args = parser.parse_args()
    
    # Load model
    print("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    config = ModelConfig.from_size('medium')  # Adjust if needed
    model = ChineseChessNet(config)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"MCTS simulations: {args.mcts_sims}")
    
    # Setup MCTS
    mcts = MCTS(model, num_simulations=args.mcts_sims, temperature=0.1)
    
    # Start game
    game_state = GameState()
    human_is_red = (args.human_color == 'red')
    
    print("\n" + "="*60)
    print("Chinese Chess - Play against AI")
    print("="*60)
    print(f"You are playing as: {'Red' if human_is_red else 'Black'}")
    print("Move format: a0-a1 (from column+row to column+row)")
    print("Columns: a-i (left to right)")
    print("Rows: 0-9 (bottom to top for Red view)")
    print("Type 'quit' to exit")
    print("="*60 + "\n")
    
    while not game_state.is_terminal():
        print(game_state)
        print()
        
        is_human_turn = (game_state.is_red_turn == human_is_red)
        
        if is_human_turn:
            # Human move
            while True:
                move_str = input(f"Your move ({'Red' if game_state.is_red_turn else 'Black'}): ").strip()
                
                if move_str.lower() == 'quit':
                    print("Game ended by player.")
                    return
                
                move = parse_move(move_str)
                if move is None:
                    print("Invalid format. Use: a0-a1")
                    continue
                
                from_row, from_col, to_row, to_col = move
                
                if game_state.make_move(from_row, from_col, to_row, to_col):
                    break
                else:
                    print("Illegal move. Try again.")
                    legal_moves = game_state.get_legal_moves()
                    print(f"You have {len(legal_moves)} legal moves.")
        else:
            # AI move
            print(f"AI thinking ({'Red' if game_state.is_red_turn else 'Black'})...")
            
            move, _ = mcts.get_action_probs(game_state, temperature=0.1)
            
            if move is None:
                print("AI has no legal moves!")
                break
            
            from_row, from_col, to_row, to_col = move
            game_state.make_move(from_row, from_col, to_row, to_col)
            
            notation = game_state.get_move_notation(from_row, from_col, to_row, to_col)
            print(f"AI played: {notation}")
        
        print()
    
    # Game ended
    print("\n" + "="*60)
    print(game_state)
    print("="*60)
    
    winner = game_state.get_winner()
    if winner is None:
        print("Game ended in a draw!")
    elif winner == human_is_red:
        print("You won!")
    else:
        print("AI won!")

if __name__ == '__main__':
    main()
