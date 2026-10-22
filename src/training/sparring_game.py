"""One model-vs-Pikafish game with every ply stored for ReplayBuffer."""
import math
from typing import List, Tuple

import numpy as np

from ..engine.coordinates import move_to_uci
from ..engine.player import PikafishPlayer
from ..game.game_state import GameState
from ..model.mcts import MCTS
from ..model.neural_network import ChineseChessNet
from .self_play import GREEDY_TEMPERATURE, MAX_GAME_PLIES
from .teacher_label import build_teacher_label

GameData = Tuple[List[np.ndarray], List[np.ndarray], float]


def _red_centric_winner(game_state: GameState) -> float:
    winner_bool = game_state.get_winner()
    if winner_bool is None:
        return 0.0
    return 1.0 if winner_bool else -1.0


def _one_hot_policy(game_state: GameState, move) -> np.ndarray:
    policy, _ = build_teacher_label(
        game_state.board, game_state.is_red_turn, move_to_uci(move), 0)
    return policy


def play_sparring_game(model: ChineseChessNet, opponent: PikafishPlayer,
                       opponent_is_red: bool, mcts_simulations: int = 400,
                       temperature: float = 1.0, temperature_threshold: int = 15,
                       dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                       max_game_plies: int = MAX_GAME_PLIES,
                       no_capture_limit: int = 100,
                       verbose: bool = False) -> GameData:
    """Play one MCTS-vs-Pikafish game and keep every ply, including the opponent's.

    Opponent plies use a one-hot policy so ``ReplayBuffer.add_game`` can assign
    values by ply index ``i % 2`` (red always starts).
    """
    game_state = GameState(max_plies=max_game_plies, no_capture_limit=no_capture_limit)
    mcts = MCTS(model, num_simulations=mcts_simulations,
                dirichlet_alpha=dirichlet_alpha, dirichlet_epsilon=dirichlet_epsilon)
    states: List[np.ndarray] = []
    policies: List[np.ndarray] = []
    move_count = 0
    while not game_state.is_terminal():
        state_tensor = game_state.to_tensor()
        model_to_move = game_state.is_red_turn != opponent_is_red
        if model_to_move:
            current_temperature = (
                temperature if move_count < temperature_threshold else GREEDY_TEMPERATURE)
            move, policy = mcts.get_action_probs(game_state, temperature=current_temperature)
            if move is None:
                raise RuntimeError("MCTS returned no move for a non-terminal game")
        else:
            move = opponent.choose_move(game_state)
            if move is None:
                raise RuntimeError("Sparring opponent returned no move")
            policy = _one_hot_policy(game_state, move)
        if not game_state.make_move(*move):
            raise RuntimeError(f"Sparring selected an illegal move: {move}")
        states.append(state_tensor)
        policies.append(policy)
        if verbose:
            notation = game_state.get_move_notation(*move)
            print(f"Move {move_count + 1}: {notation}")
        move_count += 1
    winner = _red_centric_winner(game_state)
    if verbose:
        print(f"Game ended after {move_count} moves. Result: {winner}")
    return states, policies, winner


class SparringWorker:
    """Same ``play_game`` signature as ``SelfPlayWorker``, versus a Pikafish opponent."""

    def __init__(self, model: ChineseChessNet, opponent: PikafishPlayer,
                 opponent_is_red: bool, mcts_simulations: int = 400,
                 temperature: float = 1.0, temperature_threshold: int = 15,
                 dirichlet_alpha: float = 0.3, dirichlet_epsilon: float = 0.25,
                 max_game_plies: int = MAX_GAME_PLIES, no_capture_limit: int = 100):
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and non-negative")
        if temperature_threshold < 0:
            raise ValueError("temperature_threshold cannot be negative")
        self.model = model
        self.opponent = opponent
        self.opponent_is_red = opponent_is_red
        self.mcts_simulations = mcts_simulations
        self.temperature = temperature
        self.temperature_threshold = temperature_threshold
        self.dirichlet_alpha = dirichlet_alpha
        self.dirichlet_epsilon = dirichlet_epsilon
        self.max_game_plies = max_game_plies
        self.no_capture_limit = no_capture_limit
        self.last_termination_reason = "Game in progress"

    def play_game(self, verbose: bool = False) -> GameData:
        """Play one sparring game; returns ``(states, policies, winner)``."""
        states, policies, winner = play_sparring_game(
            self.model, self.opponent, self.opponent_is_red,
            mcts_simulations=self.mcts_simulations, temperature=self.temperature,
            temperature_threshold=self.temperature_threshold,
            dirichlet_alpha=self.dirichlet_alpha, dirichlet_epsilon=self.dirichlet_epsilon,
            max_game_plies=self.max_game_plies, no_capture_limit=self.no_capture_limit,
            verbose=verbose)
        self.last_termination_reason = f"plies {len(states)}"
        return states, policies, winner
