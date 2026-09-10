"""Empty board square implementation."""
from typing import TYPE_CHECKING, List

from .base import Piece
from .types import PieceType, Position

if TYPE_CHECKING:
    from ..board import Board


class EmptyPiece(Piece):
    """A non-movable placeholder used for unoccupied board squares."""

    def __init__(self, is_red: bool = True):
        super().__init__(PieceType.EMPTY, is_red)

    def get_possible_moves(self, row: int, col: int, board: 'Board') -> List[Position]:
        """Empty squares cannot move."""
        return []
