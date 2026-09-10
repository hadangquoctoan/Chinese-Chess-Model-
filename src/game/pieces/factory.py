"""Piece construction for board setup and state restoration."""
from typing import Dict, Type

from .advisor import Advisor
from .base import Piece
from .cannon import Cannon
from .chariot import Chariot
from .elephant import Elephant
from .empty import EmptyPiece
from .general import General
from .horse import Horse
from .soldier import Soldier
from .types import PieceType


PIECE_CLASSES: Dict[PieceType, Type[Piece]] = {
    PieceType.GENERAL: General,
    PieceType.ADVISOR: Advisor,
    PieceType.ELEPHANT: Elephant,
    PieceType.HORSE: Horse,
    PieceType.CHARIOT: Chariot,
    PieceType.CANNON: Cannon,
    PieceType.SOLDIER: Soldier,
    PieceType.EMPTY: EmptyPiece,
}


def create_piece(piece_type: PieceType, is_red: bool) -> Piece:
    """Create the specialized piece class for a board-square type."""
    try:
        piece_class = PIECE_CLASSES[piece_type]
    except KeyError as error:
        raise ValueError(f"Unsupported piece type: {piece_type!r}") from error
    return piece_class(is_red)
