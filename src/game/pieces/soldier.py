"""Soldier movement rules."""
from typing import TYPE_CHECKING, List

from .base import Piece
from .types import PieceType, Position

if TYPE_CHECKING:
    from ..board import Board


class Soldier(Piece):
    """A soldier advances and gains sideways movement after crossing the river."""

    def __init__(self, is_red: bool):
        super().__init__(PieceType.SOLDIER, is_red)

    def get_possible_moves(self, row: int, col: int, board: 'Board') -> List[Position]:
        """Return forward moves and sideways moves after the river crossing."""
        moves = []
        forward_row = row + (1 if self.is_red else -1)
        if self.can_move_to(board, forward_row, col):
            moves.append((forward_row, col))

        if self._has_crossed_river(row):
            moves.extend(self._sideways_moves(row, col, board))
        return moves

    def _has_crossed_river(self, row: int) -> bool:
        """Return whether the soldier has crossed its opponent's river bank."""
        return row > 4 if self.is_red else row < 5

    def _sideways_moves(self, row: int, col: int, board: 'Board') -> List[Position]:
        """Return legal one-square horizontal moves after crossing the river."""
        moves = []
        for col_delta in (-1, 1):
            target_col = col + col_delta
            if self.can_move_to(board, row, target_col):
                moves.append((row, target_col))
        return moves
