"""Build supervised teacher labels from Pikafish output.

Single responsibility: convert one position plus engine output into a
(policy, value) training label. Move encoding is delegated to
``Rules.move_to_action_index``; this module never encodes by hand.
"""

import math

import numpy as np

from ..engine.coordinates import parse_uci_move
from ..game.board import Board
from ..game.rules import ACTION_SPACE_SIZE, Rules

SCORE_SCALE = 500.0


def build_teacher_label(board: Board, is_red_turn: bool, bestmove_token: str,
                        score_cp: int) -> tuple[np.ndarray, float]:
    """Convert one Pikafish result into a (policy, value) label.

    Args:
        board: Current board (passed through to Rules.move_to_action_index).
        is_red_turn: True if red is to move.
        bestmove_token: Pikafish UCI token such as ``h2e2``.
        score_cp: Engine score in centipawns, from red/black side to move.

    Returns:
        Tuple of (policy, value) where policy is a float32 one-hot of
        shape (1800,) and value is red-centric in [-1, 1].

    Raises:
        ValueError: if the token cannot be parsed or encoded.
    """
    try:
        from_row, from_col, to_row, to_col = parse_uci_move(bestmove_token)
    except (ValueError, TypeError, AttributeError) as exc:
        raise ValueError(f"Invalid bestmove token: {bestmove_token!r}") from exc

    try:
        index = Rules.move_to_action_index(from_row, from_col, to_row, to_col, board)
    except ValueError as exc:
        raise ValueError(f"Cannot encode move {bestmove_token!r}: {exc}") from exc
    if not isinstance(index, int) or not 0 <= index < ACTION_SPACE_SIZE:
        raise ValueError(f"Move {bestmove_token!r} produced invalid index {index!r}")

    policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
    policy[index] = 1.0

    value = math.tanh(score_cp / SCORE_SCALE)
    if not is_red_turn:
        value = -value
    value = max(-1.0, min(1.0, value))

    return policy, float(value)
