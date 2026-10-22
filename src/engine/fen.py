"""Export the project board to Xiangqi FEN for UCI engines like Pikafish."""

from ..game.board import Board
from ..game.pieces import PieceType

_FEN_LETTER = {
    PieceType.CHARIOT: "r",
    PieceType.HORSE: "n",
    PieceType.ELEPHANT: "b",
    PieceType.ADVISOR: "a",
    PieceType.GENERAL: "k",
    PieceType.CANNON: "c",
    PieceType.SOLDIER: "p",
}

STARTPOS_FEN = (
    "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w - - 0 1"
)


def board_to_fen(board: Board, is_red_turn: bool) -> str:
    """Return a FEN string with ranks ordered 9 down to 0, red as uppercase."""
    ranks = []
    for row in range(board.rows - 1, -1, -1):
        empty = 0
        rank = []
        for col in range(board.cols):
            piece = board.get_piece(row, col)
            if piece.piece_type == PieceType.EMPTY:
                empty += 1
                continue
            if empty:
                rank.append(str(empty))
                empty = 0
            letter = _FEN_LETTER[piece.piece_type]
            rank.append(letter.upper() if piece.is_red else letter)
        if empty:
            rank.append(str(empty))
        ranks.append("".join(rank))
    turn = "w" if is_red_turn else "b"
    return f"{'/'.join(ranks)} {turn} - - 0 1"
