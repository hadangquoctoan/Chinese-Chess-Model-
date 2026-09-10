"""Horse movement rules."""
from typing import TYPE_CHECKING, List

from .base import Piece
from .types import PieceType, Position

if TYPE_CHECKING:
    from ..board import Board


HORSE_PATTERNS = (
    ((1, 2), (0, 1)),
    ((1, -2), (0, -1)),
    ((-1, 2), (0, 1)),
    ((-1, -2), (0, -1)),
    ((2, 1), (1, 0)),
    ((2, -1), (1, 0)),
    ((-2, 1), (-1, 0)),
    ((-2, -1), (-1, 0)),
)


class Horse(Piece):
    """A horse moves in an L-shape when its leg is not blocked."""

    def __init__(self, is_red: bool):
        super().__init__(PieceType.HORSE, is_red)

    def get_possible_moves(self, row: int, col: int, board: 'Board') -> List[Position]:
        """Return L-shaped moves whose adjacent leg is unblocked."""
        moves = []
        for (row_delta, col_delta), (block_row_delta, block_col_delta) in HORSE_PATTERNS:
            target_row = row + row_delta
            target_col = col + col_delta
            block_row = row + block_row_delta
            block_col = col + block_col_delta
            if not self.is_in_bounds(target_row, target_col):
                continue
            if board.get_piece(block_row, block_col).piece_type != PieceType.EMPTY:
                continue
            if self.can_move_to(board, target_row, target_col):
                moves.append((target_row, target_col))
        return moves
