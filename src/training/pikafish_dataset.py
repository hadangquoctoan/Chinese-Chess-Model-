"""Generate game-level supervised datasets from Pikafish self-play labels."""

from collections.abc import Callable
from dataclasses import dataclass
import random

import numpy as np

from ..engine.coordinates import parse_uci_move
from ..engine.fen import board_to_fen
from ..engine.uci import PikafishEngine
from ..game.game_state import GameState, NO_CAPTURE_DRAW_PLY_LIMIT
from .teacher_label import build_teacher_label

Move = tuple[int, int, int, int]


@dataclass
class LabeledGame:
    """One Pikafish game with a parallel label for every visited position."""

    game_id: int
    states: list[np.ndarray]
    policies: list[np.ndarray]
    values: list[float]


ProgressCallback = Callable[[int, LabeledGame], None]


def _search_label(engine: PikafishEngine, state: GameState, depth: int,
                  timeout: float) -> tuple[np.ndarray, np.ndarray, float, Move]:
    """Label one position with Pikafish's best move and score."""
    token, score_cp = engine.search_with_eval(
        board_to_fen(state.board, state.is_red_turn), [], depth=depth,
        timeout=timeout)
    move = parse_uci_move(token)
    policy, value = build_teacher_label(
        state.board, state.is_red_turn, token, score_cp)
    return state.to_tensor(), policy, value, move


def generate_pikafish_games(
        engine: PikafishEngine, num_games: int, depth: int, seed: int,
        opening_plies: int = 8, max_game_plies: int = 160,
        timeout: float = 60.0, game_id_start: int = 0,
        progress: ProgressCallback | None = None) -> list[LabeledGame]:
    """Play complete Pikafish games and label every visited position.

    Every stored position is labeled by Pikafish, while the first
    ``opening_plies`` played moves are sampled uniformly from legal moves to
    diversify games. The remaining moves follow Pikafish's best move.
    ``progress`` receives the number of completed games and the finished game.
    """
    if num_games <= 0 or depth <= 0 or max_game_plies <= 0:
        raise ValueError('num_games, depth, and max_game_plies must be positive')
    if opening_plies < 0 or timeout <= 0:
        raise ValueError('opening_plies must be nonnegative and timeout positive')
    rng = random.Random(seed)
    games: list[LabeledGame] = []
    for game_index in range(num_games):
        state = GameState(max_plies=max_game_plies,
                          no_capture_limit=NO_CAPTURE_DRAW_PLY_LIMIT)
        game = LabeledGame(game_id_start + game_index, [], [], [])
        ply = 0
        while not state.is_terminal():
            board, policy, value, teacher_move = _search_label(
                engine, state, depth, timeout)
            game.states.append(board)
            game.policies.append(policy)
            game.values.append(value)
            move = teacher_move
            if ply < opening_plies:
                legal = state.get_legal_moves()
                move = legal[rng.randrange(len(legal))]
            if not state.make_move(*move):
                raise RuntimeError(f'Generated move is illegal: {move}')
            ply += 1
        games.append(game)
        if progress is not None:
            progress(game_index + 1, game)
    return games


def split_train_validation(
        games: list[LabeledGame], validation_games: int
) -> tuple[list[LabeledGame], list[LabeledGame]]:
    """Split whole games so train and validation positions never overlap."""
    if (type(validation_games) is not int or validation_games <= 0
            or validation_games >= len(games)):
        raise ValueError(
            'validation_games must leave at least one train and validation game')
    return games[:-validation_games], games[-validation_games:]


def flatten_games(
        games: list[LabeledGame]) -> tuple[list[np.ndarray], list[np.ndarray],
                                           list[float], list[int]]:
    """Flatten labeled games while repeating each position's game ID."""
    for game in games:
        if not (len(game.states) == len(game.policies) == len(game.values)):
            raise ValueError(
                f'game {game.game_id} has misaligned states/policies/values')
    states = [state for game in games for state in game.states]
    policies = [policy for game in games for policy in game.policies]
    values = [value for game in games for value in game.values]
    game_ids = [game.game_id for game in games for _ in game.states]
    if not states:
        raise ValueError('Cannot flatten an empty dataset')
    return states, policies, values, game_ids
