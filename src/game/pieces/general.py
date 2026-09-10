"""General movement rules."""
from typing import TYPE_CHECKING, List

from .base import Piece
from .types import PieceType, Position

if TYPE_CHECKING:
    from ..board import Board


GENERAL_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))


class General(Piece):
    """A general moves one orthogonal square within its palace."""

    def __init__(self, is_red: bool):
        super().__init__(PieceType.GENERAL, is_red)

    def get_possible_moves(self, row: int, col: int, board: 'Board') -> List[Position]:
        """Return palace-constrained orthogonal moves."""
        moves = []
        for row_delta, col_delta in GENERAL_DIRECTIONS:
            target_row = row + row_delta
            target_col = col + col_delta
            if not self.in_palace(target_row, target_col, self.is_red):
                continue
            if self.can_move_to(board, target_row, target_col):
                moves.append((target_row, target_col))
        return moves
