"""Adjudicate terminal draws by material for self-play training targets."""
import math

from ..game.pieces import PieceType

MATERIAL_VALUES = {
    PieceType.CHARIOT: 9.0,
    PieceType.CANNON: 4.5,
    PieceType.HORSE: 4.0,
    PieceType.ADVISOR: 2.0,
    PieceType.ELEPHANT: 2.0,
    PieceType.SOLDIER: 1.0,
    PieceType.GENERAL: 0.0,
    PieceType.EMPTY: 0.0,
}


def material_balance(state) -> float:
    """Return red material minus black material in the project's units."""
    balance = 0.0
    for row in range(state.board.rows):
        for col in range(state.board.cols):
            piece = state.board.get_piece(row, col)
            value = MATERIAL_VALUES[piece.piece_type]
            balance += value if piece.is_red else -value
    return balance


def adjudicated_winner(state, margin: float = 0.0) -> bool | None:
    """Return True/False/None; only terminal draws with a material edge are decided."""
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError("adjudication margin must be a finite number")
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("adjudication margin must be finite and non-negative")
    if not state.is_terminal():
        return None
    winner = state.get_winner()
    if winner is not None or margin == 0:
        return winner
    balance = material_balance(state)
    if abs(balance) < margin:
        return None
    return balance > 0
