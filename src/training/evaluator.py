import random
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
    
    def __init__(self, mcts_simulations: int = 200,
                 max_game_plies: int = MAX_GAME_PLIES, no_capture_limit: int = 100,
                 opening_plies: int = 4, opening_seed: int = 0):
        """
        Args:
            mcts_simulations: Số simulations cho MCTS (ít hơn để đánh giá nhanh)
        """
        if type(mcts_simulations) is not int or mcts_simulations <= 0:
            raise ValueError("mcts_simulations must be a positive integer")
        GameState(max_plies=max_game_plies, no_capture_limit=no_capture_limit)
        self.mcts_simulations = mcts_simulations
        self.max_game_plies = max_game_plies
        self.no_capture_limit = no_capture_limit
        if type(opening_plies) is not int or opening_plies < 0:
            raise ValueError("opening_plies must be a non-negative integer")
        if type(opening_seed) is not int:
            raise ValueError("opening_seed must be an integer")
        self.opening_plies = opening_plies
        self.opening_seed = opening_seed
        self.last_termination_reason = "Game in progress"
    
    def play_game(self, model1: ChineseChessNet, model2: ChineseChessNet,
                  verbose: bool = False, opening_moves: tuple = ()) -> float:
        """
        Cho 2 model chơi 1 ván
        
        Args:
            model1: Model chơi đỏ
            model2: Model chơi đen
            verbose: In ra nước đi
        
        Returns:
            1.0 nếu model1 thắng, -1.0 nếu model2 thắng, 0.0 nếu hòa
        """
        game_state = GameState(max_plies=self.max_game_plies,
                               no_capture_limit=self.no_capture_limit)
        for move in opening_moves:
            if game_state.is_terminal() or not game_state.make_move(*move):
                raise ValueError(f"Invalid evaluation opening move: {move}")
        mcts1 = MCTS(model1, num_simulations=self.mcts_simulations,
                     temperature=0.0, dirichlet_epsilon=0.0)
        mcts2 = MCTS(model2, num_simulations=self.mcts_simulations,
                     temperature=0.0, dirichlet_epsilon=0.0)
        
        move_count = 0
        
        while not game_state.is_terminal():
            # Chọn model tương ứng với lượt đi
            current_mcts = mcts1 if game_state.is_red_turn else mcts2
            
            # MCTS search
            move, _ = current_mcts.get_action_probs(game_state, temperature=0.0)
            
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
            

        # Determine winner
        self.last_termination_reason = game_state.get_termination_reason()
        winner = game_state.get_winner()
        
        if winner is None:
            return 0.0  # Draw
        elif winner:
            return 1.0  # Model1 (Red) wins
        else:
            return -1.0  # Model2 (Black) wins
    
    def _opening(self, pair_index: int) -> tuple:
        """Generate a reproducible opening without consuming training RNG."""
        generator = random.Random(self.opening_seed + pair_index)
        game = GameState(max_plies=self.max_game_plies,
                         no_capture_limit=self.no_capture_limit)
        opening = []
        for _ in range(min(self.opening_plies, self.max_game_plies - 1)):
            if game.is_terminal():
                break
            move = generator.choice(game.get_legal_moves())
            next_state = game.copy()
            next_state.make_move(*move)
            if next_state.is_terminal():
                break
            opening.append(move)
            game = next_state
        return tuple(opening)

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
            score (wins + half draws), wins, losses, draws
        """
        if type(num_games) is not int or num_games <= 0:
            raise ValueError("num_games must be a positive integer")
        if num_games % 2:
            raise ValueError("num_games must be even for paired-color evaluation")
        modes = {model: model.training for model in (new_model, best_model)}
        wins = losses = draws = 0
        iterator = tqdm(range(num_games), desc="Evaluation") if verbose else range(num_games)
        try:
            for model in modes:
                model.eval()
            with torch.no_grad():
                for index in iterator:
                    if index % 2 == 0:
                        opening = self._opening(index // 2)
                        result = self.play_game(new_model, best_model, verbose=False,
                                                opening_moves=opening)
                    else:
                        result = -self.play_game(best_model, new_model, verbose=False,
                                                 opening_moves=opening)
                    wins += int(result > 0)
                    losses += int(result < 0)
                    draws += int(result == 0)
        finally:
            for model, was_training in modes.items():
                model.train(was_training)
        score = (wins + 0.5 * draws) / num_games
        if verbose:
            print(f"Evaluation: W:{wins} L:{losses} D:{draws}; score={score:.1%}")
            print(f"Outright win rate: {wins / num_games:.1%}")
        return score, wins, losses, draws
