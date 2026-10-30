"""Terminal outcomes for the explicit xiangqi-training-v2 ruleset."""
from typing import Optional

from ..board import Board
from ..rules import RULESET_VERSION, Move, Rules
from .move_history import MoveHistory
from .position_tracker import PositionTracker
from .repetition import adjudicate

NO_CAPTURE_DRAW_PLY_LIMIT = 100
MAX_GAME_PLIES = 200


class GameResult:
    """Use one outcome function so search, UI and training agree."""

    @staticmethod
    def outcome(board: Board, is_red_turn: bool, move_history: MoveHistory,
                position_tracker: PositionTracker, max_plies: int = MAX_GAME_PLIES,
                no_capture_limit: int = NO_CAPTURE_DRAW_PLY_LIMIT, *,
                legal_moves: Optional[list[Move]] = None) -> tuple:
        """Return the outcome, optionally reusing legal moves for this position."""
        Rules.validate_position(board)
        has_moves = (bool(legal_moves) if legal_moves is not None
                     else Rules.has_legal_move(board, is_red_turn))
        if not has_moves:
            winner = not is_red_turn
            name = 'Red' if winner else 'Black'
            kind = 'Checkmate' if Rules.is_in_check(board, is_red_turn) else 'Stalemate'
            return True, winner, f'{kind} - {name} wins'
        if position_tracker.is_draw_by_repetition():
            winner, reason = adjudicate(move_history, position_tracker)
            return True, winner, reason
        if move_history.count_moves_since_capture() >= no_capture_limit:
            return True, None, f'Draw - {no_capture_limit} plies without capture'
        if len(move_history) >= max_plies:
            return True, None, 'Draw - Maximum game plies reached'
        return False, None, 'Game in progress'

    @staticmethod
    def is_terminal(*args, **kwargs) -> bool:
        """Return whether the training ruleset ends the game."""
        return GameResult.outcome(*args, **kwargs)[0]

    @staticmethod
    def get_winner(*args, **kwargs) -> Optional[bool]:
        """Return winner color, or None for a draw/in-progress game."""
        return GameResult.outcome(*args, **kwargs)[1]

    @staticmethod
    def get_result_value(board: Board, is_red_turn: bool, *args, **kwargs) -> float:
        """Return the outcome from the side-to-move perspective."""
        winner = GameResult.get_winner(board, is_red_turn, *args, **kwargs)
        return 0.0 if winner is None else (1.0 if winner == is_red_turn else -1.0)

    @staticmethod
    def get_termination_reason(*args, **kwargs) -> str:
        """Describe the same outcome used by terminal and winner queries."""
        return GameResult.outcome(*args, **kwargs)[2]
