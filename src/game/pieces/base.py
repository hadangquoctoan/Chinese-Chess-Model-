"""Base behavior shared by all Xiangqi pieces."""
from typing import TYPE_CHECKING, List

from .types import BOARD_COLS, BOARD_ROWS, PIECE_SYMBOLS, PieceType, Position

if TYPE_CHECKING:
    from ..board import Board


class Piece:
    """Base class for a board piece with pseudo-legal move generation."""

    def __init__(self, piece_type: PieceType, is_red: bool):
        self.piece_type = piece_type
        self.is_red = is_red

    def __str__(self) -> str:
        """Return the display symbol for this piece."""
        if self.piece_type == PieceType.EMPTY:
            return '·'
        return PIECE_SYMBOLS[self.piece_type][0 if self.is_red else 1]

    def __repr__(self) -> str:
        color = 'R' if self.is_red else 'B'
        return f"{color}{self.piece_type.name[0]}"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Piece):
            return False
        return self.piece_type == other.piece_type and self.is_red == other.is_red

    @staticmethod
    def is_in_bounds(row: int, col: int) -> bool:
        """Return whether a coordinate belongs to the 10-by-9 board."""
        return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS

    @staticmethod
    def in_palace(row: int, col: int, is_red: bool) -> bool:
        """Return whether a coordinate belongs to a side's palace."""
        if is_red:
            return 0 <= row <= 2 and 3 <= col <= 5
        return 7 <= row <= 9 and 3 <= col <= 5

    @staticmethod
    def in_own_half(row: int, is_red: bool) -> bool:
        """Return whether a row remains on a side's half of the board."""
        if is_red:
            return 0 <= row <= 4
        return 5 <= row <= 9

    def can_move_to(self, board: 'Board', row: int, col: int) -> bool:
        """Return whether the destination is empty or occupied by an opponent."""
        if not self.is_in_bounds(row, col):
            return False
        target = board.get_piece(row, col)
        return target.piece_type == PieceType.EMPTY or target.is_red != self.is_red

    def get_possible_moves(
        self,
        row: int,
        col: int,
        board: 'Board',
    ) -> List[Position]:
        """Return pseudo-legal moves without checking the moving side's king."""
        raise NotImplementedError
