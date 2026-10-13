"""Versioned, injective 20-slot policy encoding at each absolute origin.

Slots enumerate sorted geometric destinations for the origin piece, ignoring
blockers, palace/river restrictions and check. Unused slots remain masked.
This deliberately changes policy semantics: old policies require retraining.
"""
from functools import lru_cache
from typing import Optional, Tuple

from .board import Board
from .pieces import PieceType

ACTION_SPACE_SIZE = 1800
ACTION_DESTINATION_BUCKETS = 20
ACTION_ENCODING_VERSION = 'piece-slots-v2'
BOARD_ROWS = 10
BOARD_COLS = 9
Move = Tuple[int, int, int, int]


def valid_coordinate(row: int, col: int) -> bool:
    """Reject malformed as well as out-of-board coordinates."""
    return type(row) is int and type(col) is int and 0 <= row < 10 and 0 <= col < 9


@lru_cache(maxsize=1260)
def destinations(kind: PieceType, red: bool, row: int, col: int) -> tuple:
    """Return the stable sorted geometric slot table, never board-dependent."""
    if kind in (PieceType.CANNON, PieceType.CHARIOT):
        return tuple(sorted([(r, col) for r in range(10) if r != row] +
                            [(row, c) for c in range(9) if c != col]))
    deltas = {
        PieceType.HORSE: ((1, 2), (1, -2), (-1, 2), (-1, -2),
                          (2, 1), (2, -1), (-2, 1), (-2, -1)),
        PieceType.ELEPHANT: ((2, 2), (2, -2), (-2, 2), (-2, -2)),
        PieceType.ADVISOR: ((1, 1), (1, -1), (-1, 1), (-1, -1)),
        PieceType.GENERAL: ((0, 1), (0, -1), (1, 0), (-1, 0)),
        PieceType.SOLDIER: ((1 if red else -1, 0), (0, -1), (0, 1)),
    }.get(kind, ())
    return tuple(sorted((row + dr, col + dc) for dr, dc in deltas
                        if valid_coordinate(row + dr, col + dc)))


def encode(from_row: int, from_col: int, to_row: int, to_col: int, board: Board) -> int:
    """Encode geometry only; legality remains the caller's responsibility."""
    if not valid_coordinate(from_row, from_col) or not valid_coordinate(to_row, to_col):
        raise ValueError('Move coordinates are outside the board or not integers')
    piece = board.get_piece(from_row, from_col)
    slots = destinations(piece.piece_type, piece.is_red, from_row, from_col)
    try:
        slot = slots.index((to_row, to_col))
    except ValueError as exc:
        raise ValueError('Empty origin or non-geometric piece move') from exc
    return (from_row * BOARD_COLS + from_col) * ACTION_DESTINATION_BUCKETS + slot


def decode_geometry(index: int, board: Board, is_red: bool) -> Optional[Move]:
    """Decode one geometric candidate; Rules then verifies its legality."""
    if type(index) is not int or not 0 <= index < ACTION_SPACE_SIZE:
        return None
    origin, slot = divmod(index, ACTION_DESTINATION_BUCKETS)
    row, col = divmod(origin, BOARD_COLS)
    piece = board.get_piece(row, col)
    if piece.is_red != is_red:
        return None
    slots = destinations(piece.piece_type, piece.is_red, row, col)
    return (row, col, *slots[slot]) if slot < len(slots) else None
