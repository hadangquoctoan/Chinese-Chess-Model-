"""Cannon movement rules."""
from typing import TYPE_CHECKING, List

from .base import Piece
from .chariot import ORTHOGONAL_DIRECTIONS
from .types import PieceType, Position

if TYPE_CHECKING:
    from ..board import Board


class Cannon(Piece):
    """A cannon moves freely but captures only across one screen."""

    def __init__(self, is_red: bool):
        super().__init__(PieceType.CANNON, is_red)

    def get_possible_moves(self, row: int, col: int, board: 'Board') -> List[Position]:
        """Return movement and capture moves along each orthogonal ray."""
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
        """Return moves along one ray, enforcing the single-screen rule."""
        moves = []
        has_screen = False
        target_row = row + row_delta
        target_col = col + col_delta
        while self.is_in_bounds(target_row, target_col):
            target = board.get_piece(target_row, target_col)
            if target.piece_type == PieceType.EMPTY:
                if not has_screen:
                    moves.append((target_row, target_col))
            elif not has_screen:
                has_screen = True
            else:
                if target.is_red != self.is_red:
                    moves.append((target_row, target_col))
                break
            target_row += row_delta
            target_col += col_delta
        return moves
