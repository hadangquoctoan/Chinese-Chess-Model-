"""Coordinate/FEN conversion and Pikafish move validation against project Rules."""
import io
from pathlib import Path

import pytest

from src.engine.coordinates import FLIP_RANK, move_to_uci, parse_uci_move
from src.engine.fen import STARTPOS_FEN, board_to_fen
from src.engine.player import PikafishPlayer
from src.game.game_state import GameState
from src.game.pieces import PieceType, create_piece
from src.game.rules import Rules


class FakeEngine:
    def __init__(self, token: str) -> None:
        self.token = token

    def bestmove(self, fen, moves=None, depth=8, timeout=60.0):
        return self.token

    def close(self) -> None:
        return None


def test_uci_round_trip_matches_project_notation():
    assert FLIP_RANK is False
    assert parse_uci_move("h2e2") == (2, 7, 2, 4)
    assert move_to_uci((2, 7, 2, 4)) == "h2e2"
    assert parse_uci_move("b0c2") == (0, 1, 2, 2)
    assert move_to_uci((0, 1, 2, 2)) == "b0c2"
    assert parse_uci_move("a9a8") == (9, 0, 8, 0)


@pytest.mark.parametrize("token", ["", "h2e", "h2e21", "j2e2", "h10e2", "H2E2", "0000"])
def test_uci_parser_rejects_malformed_tokens(token):
    with pytest.raises(ValueError, match="Invalid Pikafish move"):
        parse_uci_move(token)


def test_startpos_fen_matches_project_board():
    game = GameState()
    assert board_to_fen(game.board, game.is_red_turn) == STARTPOS_FEN
    game.make_move(2, 7, 2, 4)
    fen = board_to_fen(game.board, game.is_red_turn)
    assert fen.startswith("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C2C4/9/RNBAKABNR b")


def test_common_opening_uci_moves_are_project_legal():
    game = GameState()
    legal = set(game.get_legal_moves())
    for token in ("h2e2", "b2e2", "b0c2", "h0g2", "a3a4"):
        move = parse_uci_move(token)
        assert move in legal
        assert Rules.is_move_legal(game.board, *move, True)


def test_all_opening_legal_moves_round_trip_uci():
    game = GameState()
    for move in game.get_legal_moves():
        assert parse_uci_move(move_to_uci(move)) == move


def test_player_accepts_legal_engine_move_and_rejects_illegal():
    game = GameState()
    player = PikafishPlayer(FakeEngine("h2e2"), depth=1)
    assert player.choose_move(game) == (2, 7, 2, 4)
    with pytest.raises(ValueError, match="not legal"):
        PikafishPlayer(FakeEngine("e0e2"), depth=1).choose_move(game)


def test_live_pikafish_opening_moves_are_project_legal():
    """Needs the bundled Pikafish binary; skipped elsewhere."""
    from src.engine.coordinates import parse_uci_move
    from src.engine.fen import board_to_fen
    from src.engine.uci import ENGINE_DIR, PikafishEngine, find_engine_binary

    try:
        binary = find_engine_binary()
    except FileNotFoundError:
        pytest.skip("Pikafish executable is not available.")
    if not all((ENGINE_DIR / name).is_file()
               for name in (binary.name, "pikafish.nnue")):
        pytest.skip("Pikafish binary or NNUE weights are missing.")
    game = GameState()
    engine = PikafishEngine()
    try:
        token = engine.bestmove(
            board_to_fen(game.board, game.is_red_turn), [], depth=2, timeout=60.0)
        move = parse_uci_move(token)
    finally:
        engine.close()
    assert move in game.get_legal_moves()
    assert Rules.is_move_legal(game.board, *move, True)


def test_live_search_with_eval_returns_token_and_score():
    from src.engine.fen import board_to_fen
    from src.engine.uci import ENGINE_DIR, PikafishEngine, find_engine_binary

    try:
        binary = find_engine_binary()
    except FileNotFoundError:
        pytest.skip("Pikafish executable is not available.")
    if not all((ENGINE_DIR / name).is_file()
               for name in (binary.name, "pikafish.nnue")):
        pytest.skip("Pikafish binary or NNUE weights are missing.")
    game = GameState()
    engine = PikafishEngine()
    try:
        token, score_cp = engine.search_with_eval(
            board_to_fen(game.board, game.is_red_turn), [], depth=2, timeout=60.0)
    finally:
        engine.close()
    move = parse_uci_move(token)
    assert move in game.get_legal_moves()
    assert type(score_cp) is int


def test_parse_bestmove_token_rejects_none():
    from src.engine.uci import PikafishEngine

    assert PikafishEngine._parse_bestmove_token("bestmove h2e2") == "h2e2"
    with pytest.raises(ValueError, match="no move"):
        PikafishEngine._parse_bestmove_token("bestmove (none)")
    with pytest.raises(ValueError, match="no move"):
        PikafishEngine._parse_bestmove_token("bestmove 0000")


def test_player_rejects_move_that_would_capture_a_general():
    game = GameState()
    empty = create_piece(PieceType.EMPTY, is_red=True)
    for row in range(10):
        for col in range(9):
            game.board.set_piece(row, col, empty)
    game.board.set_piece(0, 4, create_piece(PieceType.GENERAL, is_red=True))
    game.board.set_piece(9, 4, create_piece(PieceType.GENERAL, is_red=False))
    game.board.set_piece(8, 4, create_piece(PieceType.CHARIOT, is_red=True))
    game.reset_history()
    with pytest.raises(ValueError, match="not legal"):
        PikafishPlayer(FakeEngine("e8e9"), depth=1).choose_move(game)


class _FakeProcess:
    def __init__(self) -> None:
        self.stdout = io.StringIO()
        self.stdin = io.StringIO()

    def poll(self):
        return None

    def wait(self, timeout=None):
        return 0

    def kill(self) -> None:
        return None


def test_relative_engine_path_is_resolved_before_launch(monkeypatch):
    """Popen runs with cwd=pikafish/, so a relative engine path would be doubled."""
    from src.engine import uci

    monkeypatch.chdir(uci.ENGINE_DIR.parent)
    launches = []

    def fake_popen(args, **kwargs):
        launches.append((args, kwargs))
        return _FakeProcess()

    monkeypatch.setattr(uci.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(uci.PikafishEngine, "_send", lambda self, command: None)
    monkeypatch.setattr(
        uci.PikafishEngine, "_wait_for",
        lambda self, token, timeout, prefix=False: token)
    relative = Path("pikafish") / uci.NNUE_NAME
    engine = uci.PikafishEngine(path=relative)
    try:
        assert launches[0][0] == [str(uci.ENGINE_DIR / uci.NNUE_NAME)]
        assert launches[0][1]["cwd"] == str(uci.ENGINE_DIR)
    finally:
        engine.close()
