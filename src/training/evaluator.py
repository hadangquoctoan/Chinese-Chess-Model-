from typing import Tuple

import torch
from tqdm import tqdm

from ..game.game_state import GameState
from ..model.neural_network import ChineseChessNet
from ..model.mcts import MCTS

MAX_GAME_PLIES = 200
PROGRESS_INTERVAL = 10

class Evaluator:
    """Evaluator để đánh giá model"""
    
    def __init__(self, mcts_simulations: int = 200):
        """
        Args:
            mcts_simulations: Số simulations cho MCTS (ít hơn để đánh giá nhanh)
        """
        self.mcts_simulations = mcts_simulations
    
    def play_game(self, model1: ChineseChessNet, model2: ChineseChessNet,
                  verbose: bool = False) -> float:
        """
        Cho 2 model chơi 1 ván
        
        Args:
            model1: Model chơi đỏ
            model2: Model chơi đen
            verbose: In ra nước đi
        
        Returns:
            1.0 nếu model1 thắng, -1.0 nếu model2 thắng, 0.0 nếu hòa
        """
        game_state = GameState()
        mcts1 = MCTS(model1, num_simulations=self.mcts_simulations, temperature=0.1)
        mcts2 = MCTS(model2, num_simulations=self.mcts_simulations, temperature=0.1)
        
        move_count = 0
        
        while not game_state.is_terminal():
            # Chọn model tương ứng với lượt đi
            current_mcts = mcts1 if game_state.is_red_turn else mcts2
            
            # MCTS search
            move, _ = current_mcts.get_action_probs(game_state, temperature=0.1)
            
            if move is None:
                raise RuntimeError("MCTS returned no move for a non-terminal game")
            
            # Make move
            from_row, from_col, to_row, to_col = move
            if not game_state.make_move(from_row, from_col, to_row, to_col):
                raise RuntimeError(f"MCTS selected an illegal move: {move}")
            
            if verbose:
                notation = game_state.get_move_notation(from_row, from_col, to_row, to_col)
                print(f"Move {move_count + 1}: {notation}")
            
            move_count += 1
            
            if move_count >= MAX_GAME_PLIES:
                if verbose:
                    print(f"Game reached the {MAX_GAME_PLIES}-ply safety limit; recording a draw")
                break
        
        # Determine winner
        winner = game_state.get_winner()
        
        if winner is None:
            return 0.0  # Draw
        elif winner:
            return 1.0  # Model1 (Red) wins
        else:
            return -1.0  # Model2 (Black) wins
    
    def evaluate(self, new_model: ChineseChessNet, best_model: ChineseChessNet,
                num_games: int = 100, verbose: bool = True) -> Tuple[float, int, int, int]:
        """
        Đánh giá new_model vs best_model
        
        Args:
            new_model: Model mới
            best_model: Model hiện tại tốt nhất
            num_games: Số games để chơi
            verbose: In progress
        
        Returns:
            win_rate, wins, losses, draws
        """
        if num_games <= 0:
            raise ValueError("num_games must be positive")

        new_model.eval()
        best_model.eval()
        
        wins = 0
        losses = 0
        draws = 0
        
        iterator = range(num_games)
        if verbose:
            iterator = tqdm(iterator, desc="Evaluation")
        
        with torch.no_grad():
            for i in iterator:
                # Alternate colors
                if i % 2 == 0:
                    # New model plays Red
                    result = self.play_game(new_model, best_model, verbose=False)
                    if result > 0:
                        wins += 1
                    elif result < 0:
                        losses += 1
                    else:
                        draws += 1
                else:
                    # New model plays Black
                    result = self.play_game(best_model, new_model, verbose=False)
                    if result < 0:
                        wins += 1
                    elif result > 0:
                        losses += 1
                    else:
                        draws += 1
                
                if verbose and (i + 1) % PROGRESS_INTERVAL == 0:
                    current_wr = wins / (i + 1)
                    tqdm.write(f"After {i+1} games: W:{wins} L:{losses} D:{draws} (WR: {current_wr:.1%})")
        
        win_rate = wins / num_games if num_games > 0 else 0.0
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Wins:     {wins}/{num_games} ({wins/num_games:.1%})")
            print(f"  Losses:   {losses}/{num_games} ({losses/num_games:.1%})")
            print(f"  Draws:    {draws}/{num_games} ({draws/num_games:.1%})")
            print(f"  Win Rate: {win_rate:.1%}")
        
        return win_rate, wins, losses, draws
