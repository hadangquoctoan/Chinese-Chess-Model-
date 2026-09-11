"""Regression coverage for the local human-versus-human web game service."""
from scripts.play_web import LocalGame
from src.game.pieces import PieceType, create_piece


def test_local_game_applies_a_legal_move_and_undoes_it():
    """The web service must defer movement and undo to GameState."""
    game = LocalGame()
    initial_state = game.snapshot()
    move = tuple(initial_state["legalMoves"][0])

    response = game.move(move)

    assert response["ok"]
    assert response["state"]["moveCount"] == 1
    assert not response["state"]["isRedTurn"]

    response = game.undo()

    assert response["ok"]
    assert response["state"]["moveCount"] == 0
    assert response["state"]["isRedTurn"]


def test_local_game_rejects_an_illegal_move_without_mutating_state():
    """Illegal browser payloads cannot bypass the engine rules."""
    game = LocalGame()

    response = game.move((0, 0, 9, 0))

    assert not response["ok"]
    assert response["state"]["moveCount"] == 0
    assert response["state"]["isRedTurn"]


def test_local_game_serializes_capture_and_undo_from_engine_history():
    """Captured pieces must disappear and return when GameState undoes a move."""
    game = LocalGame()
    board = game._state.board
    empty_piece = create_piece(PieceType.EMPTY, is_red=True)
    for row in range(board.rows):
        for col in range(board.cols):
            board.set_piece(row, col, empty_piece)

    board.set_piece(0, 4, create_piece(PieceType.GENERAL, is_red=True))
    board.set_piece(9, 4, create_piece(PieceType.GENERAL, is_red=False))
    board.set_piece(5, 4, create_piece(PieceType.SOLDIER, is_red=True))
    board.set_piece(5, 0, create_piece(PieceType.CHARIOT, is_red=True))
    board.set_piece(5, 3, create_piece(PieceType.SOLDIER, is_red=False))

    response = game.move((5, 0, 5, 3))

    assert response["ok"]
    assert response["state"]["board"][5][3]["type"] == "CHARIOT"
    assert response["state"]["board"][5][3]["isRed"]

    response = game.undo()

    assert response["ok"]
    assert response["state"]["board"][5][0]["type"] == "CHARIOT"
    assert response["state"]["board"][5][3]["type"] == "SOLDIER"
