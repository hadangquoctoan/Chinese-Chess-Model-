"""Repetition adjudication for xiangqi-training-v2, NOT federation rules.

At the third occurrence (including side to move), inspect all moves between
its last three occurrences. One color checking on every move loses. Mutual
continuous check draws. Otherwise one color chasing the same piece on every
move loses; mutual chase draws. Check takes precedence over chase.

A chase is a newly created legal capture threat by the MOVED piece against
an opposing non-general/non-soldier. General/soldier attackers are exempt.
A target is exempt if, after the hypothetical capture, any opponent has a
legal immediate recapture. The same physical target must be threatened on
every chasing move; stable IDs survive movement, copy, and undo.

Deliberate limitations: no discovered/multi-piece chase, material-value
exceptions, alternating check/chase classification, federation-specific
protected-piece exceptions, or referee warnings. Non-covered cycles draw.
No claim of full WXF/AXF/CXA adjudication is made.
"""
from typing import Optional

from ..board import Board
from ..pieces import PieceType
from ..rules import Rules
from .move_history import MoveHistory
from .position_tracker import PositionTracker

EXEMPT_ATTACKERS = (PieceType.GENERAL, PieceType.SOLDIER)
EXEMPT_TARGETS = (PieceType.GENERAL, PieceType.SOLDIER, PieceType.EMPTY)


def assign_piece_ids(board: Board, reset: bool = False) -> None:
    """Assign unique in-game identities; reset only when installing a setup."""
    pieces = [piece for row in board.board for piece in row
              if piece.piece_type != PieceType.EMPTY]
    if reset:
        next_id = 0
    else:
        existing = [piece.repetition_id for piece in pieces if getattr(piece, "repetition_id", None) is not None]  # type: ignore[attr-defined]
        next_id = (max(existing) + 1) if existing else 0
    for piece in pieces:
        if reset or piece.repetition_id is None:
            piece.repetition_id = next_id
            next_id += 1


def chase_targets(board: Board, row: int, col: int) -> frozenset[int]:
    """Find legal, unprotected capture targets of this piece only."""
    attacker = board.get_piece(row, col)
    if attacker.piece_type in EXEMPT_ATTACKERS:
        return frozenset()
    targets = set()
    for tr, tc in attacker.get_possible_moves(row, col, board):
        target = board.get_piece(tr, tc)
        if target.piece_type in EXEMPT_TARGETS:
            continue
        if not Rules.is_move_legal(board, row, col, tr, tc, attacker.is_red):
            continue
        captured_board = board.copy()
        captured_board.move_piece(row, col, tr, tc)
        if _can_recapture(captured_board, tr, tc, not attacker.is_red):
            continue
        if target.repetition_id is not None:
            targets.add(target.repetition_id)
    return frozenset(targets)


def _can_recapture(board: Board, row: int, col: int, red: bool) -> bool:
    for fr, fc, piece in board.get_all_pieces(red):
        if (row, col) in piece.get_possible_moves(fr, fc, board):
            if Rules.is_move_legal(board, fr, fc, row, col, red):
                return True
    return False


def adjudicate(history: MoveHistory, tracker: PositionTracker) -> tuple[Optional[bool], str]:
    """Return (winner, reason) for a repeated position."""
    positions = tracker.get_positions()
    matches = [i for i, position in enumerate(positions) if position == positions[-1]]
    moves = list(history)[matches[-3]:matches[-1]]
    by_color = {red: [move for move in moves if move.mover_is_red == red]
                for red in (True, False)}
    checking = {red: bool(records) and all(move.gave_check for move in records)
                for red, records in by_color.items()}
    if checking[True] != checking[False]:
        loser = checking[True]
        return not loser, f'Perpetual check - {"Black" if loser else "Red"} wins'
    if checking[True] and checking[False]:
        return None, 'Draw - Mutual perpetual check'
    chasing = {red: _continuous_chase(records) for red, records in by_color.items()}
    if chasing[True] != chasing[False]:
        loser = chasing[True]
        return not loser, f'Perpetual chase - {"Black" if loser else "Red"} wins'
    return None, 'Draw - Position repeated 3 times'


def _continuous_chase(records: list) -> bool:
    if not records:
        return False
    common = set(records[0].chased_piece_ids)
    for move in records[1:]:
        common.intersection_update(move.chased_piece_ids)
    return bool(common)
