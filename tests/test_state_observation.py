"""Prove early-exit terminal checks and complete countdown observations."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.game.game_state import GameState
from src.game.gamestate.state_encoder import StateEncoder
from src.game.pieces import PieceType, create_piece
from src.game.rules import Rules


def test_terminal_check_does_not_enumerate_all_legal_moves(monkeypatch):
    game = GameState()
    def forbidden(*args):
        raise AssertionError("terminal query must stop at the first legal move")
    monkeypatch.setattr(Rules, "get_all_legal_moves", forbidden)
    assert not game.is_terminal()


def test_side_to_move_plane_exposes_total_ply_budget_after_capture():
    game = GameState(max_plies=20)
    capture = create_piece(PieceType.SOLDIER, False)
    game._move_history.add_move(0, 0, 0, 1, capture)
    for _ in range(2):
        game._move_history.add_move(0, 0, 0, 1, None)
    before = game.to_tensor()
    game._move_history._moves.insert(0, game._move_history.get_last_move())
    game._move_history._moves.insert(0, game._move_history.get_last_move())
    after = game.to_tensor()
    assert not np.array_equal(before[16], after[16])
    assert np.array_equal(before[17], after[17])
    assert np.all(before[16] == pytest.approx(0.575))
    game.is_red_turn = False
    assert np.all(game.to_tensor()[16] == pytest.approx(-0.625))


def test_board_copy_isolates_piece_identity_and_metadata():
    game = GameState()
    copied = game.copy()
    original = game.board.get_piece(0, 0)
    other = copied.board.get_piece(0, 0)
    assert original is not other
    assert original.repetition_id == other.repetition_id
    other.repetition_id = -1
    other.is_red = False
    assert original.is_red
    assert original.repetition_id != -1
