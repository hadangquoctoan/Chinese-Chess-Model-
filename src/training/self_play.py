from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from ..game.game_state import GameState
from ..model.mcts import MCTS
from ..model.neural_network import ChineseChessNet

GREEDY_TEMPERATURE = 0.1
MAX_GAME_PLIES = 200
PROGRESS_INTERVAL = 10

class SelfPlayWorker:
    """Worker để chạy self-play games"""
    
    def __init__(self, model: ChineseChessNet, mcts_simulations: int = 400,
                 temperature: float = 1.0, temperature_threshold: int = 15):
        """
        Args:
            model: Neural network
            mcts_simulations: Số simulations cho MCTS
            temperature: Temperature ban đầu (exploration)
            temperature_threshold: Sau bao nhiêu nước thì giảm temperature
        """
        if temperature < 0:
            raise ValueError("temperature cannot be negative")
        if temperature_threshold < 0:
            raise ValueError("temperature_threshold cannot be negative")

        self.model = model
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
    
    def play_game(self, verbose: bool = False) -> Tuple[List[np.ndarray], List[np.ndarray], float]:
        """
        Chơi một ván cờ hoàn chỉnh
        
        Returns:
            states: List of board tensors
            policies: List of MCTS policies
            winner: 1.0 (red wins), -1.0 (black wins), 0.0 (draw)
        """
        game_state = GameState()
        mcts = MCTS(self.model, num_simulations=self.mcts_simulations)
        
        states = []
        policies = []
        move_count = 0
        
        while not game_state.is_terminal():
            state_tensor = game_state.to_tensor()
            current_temperature = (
                self.temperature
                if move_count < self.temperature_threshold
                else GREEDY_TEMPERATURE
            )
            
            move, policy = mcts.get_action_probs(
                game_state,
                temperature=current_temperature,
            )
            
            if move is None:
                raise RuntimeError("MCTS returned no move for a non-terminal game")
            
            from_row, from_col, to_row, to_col = move
            if not game_state.make_move(from_row, from_col, to_row, to_col):
                raise RuntimeError(f"MCTS selected an illegal move: {move}")

            states.append(state_tensor)
            policies.append(policy)
            
            if verbose:
                notation = game_state.get_move_notation(from_row, from_col, to_row, to_col)
                print(f"Move {move_count + 1}: {notation}")
            
            move_count += 1
            
            if move_count >= MAX_GAME_PLIES:
                if verbose:
                    print(f"Game reached the {MAX_GAME_PLIES}-ply safety limit; recording a draw")
                break
        
        # Determine winner
        winner_bool = game_state.get_winner()
        if winner_bool is None:
            winner = 0.0  # Draw
        elif winner_bool:
            winner = 1.0  # Red wins
        else:
            winner = -1.0  # Black wins
        
        if verbose:
            print(f"Game ended after {move_count} moves. Result: {winner}")
        
        return states, policies, winner
    
    def generate_games(self, num_games: int, verbose: bool = False) -> List[Tuple[List[np.ndarray], List[np.ndarray], float]]:
        """
        Tạo nhiều games
        
        Args:
            num_games: Số lượng games
            verbose: In progress
        
        Returns:
            List of (states, policies, winner)
        """
        if num_games <= 0:
            raise ValueError("num_games must be positive")

        games = []
        
        iterator = range(num_games)
        if verbose:
            iterator = tqdm(iterator, desc="Self-play")
        
        for game_index in iterator:
            game_data = self.play_game(verbose=False)
            games.append(game_data)
            
            if verbose and (game_index + 1) % PROGRESS_INTERVAL == 0:
                winners = [game[2] for game in games[-PROGRESS_INTERVAL:]]
                red_wins = sum(1 for winner in winners if winner > 0.5)
                black_wins = sum(1 for winner in winners if winner < -0.5)
                draws = PROGRESS_INTERVAL - red_wins - black_wins
                print(f"Last {PROGRESS_INTERVAL} games: R:{red_wins} B:{black_wins} D:{draws}")
        
        return games

def generate_self_play_data(model: ChineseChessNet, num_games: int,
                           mcts_simulations: int = 400,
                           temperature: float = 1.0,
                           verbose: bool = True) -> Tuple[List[np.ndarray], List[np.ndarray], List[float]]:
    """
    Convenience function để generate self-play data
    
    Returns:
        all_states: List of all board tensors
        all_policies: List of all MCTS policies
        all_values: List of all values (with perspective flipped)
    """
    worker = SelfPlayWorker(model, mcts_simulations, temperature)
    games = worker.generate_games(num_games, verbose)
    
    # Flatten games into individual positions
    all_states = []
    all_policies = []
    all_values = []
    
    for states, policies, winner in games:
        for i, (state, policy) in enumerate(zip(states, policies)):
            all_states.append(state)
            all_policies.append(policy)
            
            # Flip value perspective for each move
            value = winner if i % 2 == 0 else -winner
            all_values.append(value)
    
    return all_states, all_policies, all_values
