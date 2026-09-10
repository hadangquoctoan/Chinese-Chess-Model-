"""Elephant movement rules."""
from typing import TYPE_CHECKING, List

from .base import Piece
from .types import PieceType, Position

if TYPE_CHECKING:
    from ..board import Board


ELEPHANT_PATTERNS = (
    ((2, 2), (1, 1)),
    ((2, -2), (1, -1)),
    ((-2, 2), (-1, 1)),
    ((-2, -2), (-1, -1)),
)


class Elephant(Piece):
    """An elephant moves two diagonal squares without crossing the river."""

    def __init__(self, is_red: bool):
        super().__init__(PieceType.ELEPHANT, is_red)

    def get_possible_moves(self, row: int, col: int, board: 'Board') -> List[Position]:
        """Return moves with an unblocked eye on the elephant's own half."""
        moves = []
        for (row_delta, col_delta), (block_row_delta, block_col_delta) in ELEPHANT_PATTERNS:
            target_row = row + row_delta
            target_col = col + col_delta
            block_row = row + block_row_delta
            block_col = col + block_col_delta
            if not self.is_in_bounds(target_row, target_col):
                continue
            if not self.in_own_half(target_row, self.is_red):
                continue
            if board.get_piece(block_row, block_col).piece_type != PieceType.EMPTY:
                continue
            if self.can_move_to(board, target_row, target_col):
                moves.append((target_row, target_col))
        return moves
