"""Shared types and constants for Xiangqi pieces."""
from enum import Enum
from typing import Tuple


BOARD_ROWS = 10
BOARD_COLS = 9
Position = Tuple[int, int]


class PieceType(Enum):
    """The eight board-square states used by the game engine."""

    GENERAL = 0
    ADVISOR = 1
    ELEPHANT = 2
    HORSE = 3
    CHARIOT = 4
    CANNON = 5
    SOLDIER = 6
    EMPTY = 7


PIECE_SYMBOLS = {
    PieceType.GENERAL: ('帥', '將'),
    PieceType.ADVISOR: ('仕', '士'),
    PieceType.ELEPHANT: ('相', '象'),
    PieceType.HORSE: ('傌', '馬'),
    PieceType.CHARIOT: ('俥', '車'),
    PieceType.CANNON: ('炮', '砲'),
    PieceType.SOLDIER: ('兵', '卒'),
}
