"""Chariot movement rules."""
from typing import TYPE_CHECKING, List

from .base import Piece
from .types import PieceType, Position

if TYPE_CHECKING:
    from ..board import Board


ORTHOGONAL_DIRECTIONS = ((0, 1), (0, -1), (1, 0), (-1, 0))


class Chariot(Piece):
    """A chariot moves any distance orthogonally until blocked."""

    def __init__(self, is_red: bool):
        super().__init__(PieceType.CHARIOT, is_red)

    def get_possible_moves(self, row: int, col: int, board: 'Board') -> List[Position]:
        """Return orthogonal moves, including one enemy capture per ray."""
        moves = []
        for row_delta, col_delta in ORTHOGONAL_DIRECTIONS:
            moves.extend(self._moves_in_direction(row, col, row_delta, col_delta, board))
        return moves

    def _moves_in_direction(
        self,
        row: int,
        col: int,
        row_delta: int,
        col_delta: int,
        board: 'Board',
    ) -> List[Position]:
        """Return pseudo-legal moves in one orthogonal direction."""
        moves = []
        target_row = row + row_delta
        target_col = col + col_delta
        while self.is_in_bounds(target_row, target_col):
            target = board.get_piece(target_row, target_col)
            if target.piece_type == PieceType.EMPTY:
                moves.append((target_row, target_col))
            elif target.is_red != self.is_red:
                moves.append((target_row, target_col))
                break
            else:
                break
            target_row += row_delta
            target_col += col_delta
        return moves
