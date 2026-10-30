"""Search snapshots and trusted moves must preserve all game-rule state."""
import random

import numpy as np
import pytest

from src.game.game_state import GameState
from src.game.gamestate.game_result import GameResult
from src.game.pieces import PieceType as P, create_piece
from src.game.rules import Rules


def assert_same_state(first: GameState, second: GameState) -> None:
    """Compare observation, pieces, history metadata, outcome and limits."""
    np.testing.assert_array_equal(first.to_tensor(), second.to_tensor())
    assert first.position_history == second.position_history
    assert (first.max_plies, first.no_capture_limit, first.is_red_turn) == (
        second.max_plies, second.no_capture_limit, second.is_red_turn)
    for row in range(10):
        for col in range(9):
            assert vars(first.board.get_piece(row, col)) == vars(second.board.get_piece(row, col))
    assert first.move_count == second.move_count
    for left, right in zip(first._move_history, second._move_history):
        left_data, right_data = vars(left).copy(), vars(right).copy()
        left_piece = left_data.pop('captured_piece')
        right_piece = right_data.pop('captured_piece')
        assert left_data == right_data
        assert (vars(left_piece) if left_piece is not None else None) == (
            vars(right_piece) if right_piece is not None else None)
    assert first.get_termination_reason() == second.get_termination_reason()
    assert first.get_game_result() == second.get_game_result()


def position(pieces: list, red: bool = True, **limits: int) -> GameState:
    """Build a small position with fresh history and stable piece identities."""
    game = GameState(**limits)
    for row in range(10):
        for col in range(9):
            game.board.set_piece(row, col, create_piece(P.EMPTY, True))
    for row, col, kind, color in pieces:
        game.board.set_piece(row, col, create_piece(kind, color))
    game.is_red_turn = red
    game.reset_history()
    return game


def assert_search_info(game: GameState) -> None:
    """The combined snapshot must agree with the established public queries."""
    moves, value = game.get_search_info()
    assert moves == game.get_legal_moves()
    assert value == (game.get_game_result() if game.is_terminal() else None)


def test_trusted_moves_match_every_legal_opening_and_undo():
    game = GameState()
    for move in game.get_legal_moves():
        normal, trusted = game.copy(), game.copy()
        assert normal.make_move(*move)
        assert trusted.make_move_trusted(*move)
        assert_same_state(normal, trusted)
        assert normal.undo_move() and trusted.undo_move()
        assert_same_state(normal, trusted)
        assert_same_state(game, trusted)


def test_trusted_move_skips_revalidation_of_a_confirmed_move(monkeypatch):
    game = GameState()
    move = (3, 0, 4, 0)
    assert move in game.get_legal_moves()

    def unexpected(*args, **kwargs):
        raise AssertionError('confirmed move was revalidated')

    monkeypatch.setattr(GameState, 'is_terminal', unexpected)
    monkeypatch.setattr(Rules, 'is_move_legal', unexpected)
    assert game.make_move_trusted(*move)
    assert game.move_count == 1
    assert not game.is_red_turn


@pytest.mark.parametrize('seed', [7, 29])
def test_trusted_random_play_and_full_undo_match_normal_moves(seed):
    normal, trusted = GameState(), GameState()
    rng = random.Random(seed)
    for ply in range(40):
        moves = normal.get_legal_moves()
        if not moves:
            break
        move = (2, 1, 9, 1) if ply == 0 else rng.choice(moves)
        assert normal.make_move(*move)
        assert trusted.make_move_trusted(*move)
        assert_same_state(normal, trusted)
        assert_search_info(trusted)
    while normal.move_count:
        assert normal.undo_move() and trusted.undo_move()
        assert_same_state(normal, trusted)
    assert_same_state(GameState(), trusted)


@pytest.mark.parametrize('red', [True, False])
@pytest.mark.parametrize('kind', ['check', 'chase', 'neutral'])
def test_trusted_repetition_metadata_outcome_and_undo(kind, red):
    setups = {
        'check': ([(0, 3, P.GENERAL, True), (9, 4, P.GENERAL, False),
                   (7, 5, P.CHARIOT, True)],
                  [(7, 5, 7, 4), (9, 4, 9, 5), (7, 4, 7, 5), (9, 5, 9, 4)]),
        'chase': ([(0, 3, P.GENERAL, True), (9, 5, P.GENERAL, False),
                   (4, 2, P.CHARIOT, True), (6, 1, P.HORSE, False)],
                  [(4, 2, 4, 1), (6, 1, 8, 2), (4, 1, 4, 2), (8, 2, 6, 1)]),
        'neutral': ([(0, 3, P.GENERAL, True), (9, 5, P.GENERAL, False),
                     (4, 0, P.CHARIOT, True), (6, 8, P.CHARIOT, False)],
                    [(4, 0, 4, 1), (6, 8, 6, 7), (4, 1, 4, 0), (6, 7, 6, 8)]),
    }
    pieces, cycle = setups[kind]
    if not red:
        pieces = [(9-r, c, piece, not color) for r, c, piece, color in pieces]
        cycle = [(9-fr, fc, 9-tr, tc) for fr, fc, tr, tc in cycle]
    normal = position(pieces, red)
    trusted = normal.copy()
    for move in cycle * 2:
        assert move in normal.get_legal_moves()
        assert normal.make_move(*move)
        assert trusted.make_move_trusted(*move)
        assert_same_state(normal, trusted)
        assert_search_info(trusted)
    assert normal.is_terminal()
    assert normal.get_winner() is (None if kind == 'neutral' else not red)
    assert trusted.undo_move() and normal.undo_move()
    assert_same_state(normal, trusted)
    assert_search_info(trusted)
    assert not trusted.is_terminal()


@pytest.mark.parametrize('limit', ['max_plies', 'no_capture_limit'])
def test_search_snapshot_handles_draw_limits_and_mutation_without_stale_cache(limit):
    game = GameState(**{limit: 2})
    assert_search_info(game)
    assert game.make_move(3, 0, 4, 0)
    assert_search_info(game)
    assert game.make_move(6, 0, 5, 0)
    assert game.get_search_info() == ([], 0.0)
    assert not game.make_move(3, 2, 4, 2)
    assert game.undo_move()
    assert_search_info(game)
    assert game.get_search_info()[1] is None
    game.reset_history()
    assert_search_info(game)


@pytest.mark.parametrize('checked', [True, False])
@pytest.mark.parametrize('red', [True, False])
def test_search_snapshot_checkmate_and_stalemate_precede_draw_limits(checked, red):
    pieces = [(0, 4, P.GENERAL, True), (9, 4, P.GENERAL, False),
              (5, 4, P.CHARIOT if checked else P.SOLDIER, False),
              (1, 0, P.CHARIOT, False), (2, 3, P.CHARIOT, False),
              (2, 5, P.CHARIOT, False)]
    if not red:
        pieces = [(9-r, c, piece, not color) for r, c, piece, color in pieces]
    game = position(pieces, red, max_plies=1, no_capture_limit=1)
    game._move_history.add_move(0, 0, 0, 1, None)
    assert Rules.is_in_check(game.board, red) is checked
    assert game.get_search_info() == ([], -1.0)
    assert_search_info(game)


def test_search_snapshot_reuses_legal_moves_in_terminal_adjudication(monkeypatch):
    game = GameState()

    def unexpected(*args, **kwargs):
        raise AssertionError('legal-move availability was checked twice')

    monkeypatch.setattr(Rules, 'has_legal_move', unexpected)
    moves, value = game.get_search_info()
    assert moves and value is None


def test_outcome_keeps_default_early_exit_path(monkeypatch):
    game = GameState()

    def unexpected(*args, **kwargs):
        raise AssertionError('ordinary terminal query generated the full move list')

    monkeypatch.setattr(Rules, 'get_all_legal_moves', unexpected)
    assert not game.is_terminal()


def test_empty_known_legal_moves_means_stalemate_not_unknown():
    game = position([(0, 4, P.GENERAL, True), (9, 4, P.GENERAL, False),
                     (5, 4, P.SOLDIER, False), (1, 0, P.CHARIOT, False),
                     (2, 3, P.CHARIOT, False), (2, 5, P.CHARIOT, False)])
    outcome = GameResult.outcome(game.board, game.is_red_turn, game._move_history,
                                 game._position_tracker, legal_moves=[])
    assert outcome == (True, False, 'Stalemate - Black wins')


@pytest.mark.parametrize('duplicate', [True, False])
def test_search_snapshot_rejects_invalid_general_counts(duplicate):
    game = GameState()
    row = 1 if duplicate else 0
    game.board.set_piece(row, 4, create_piece(P.GENERAL if duplicate else P.EMPTY, True))
    with pytest.raises(ValueError, match='general'):
        game.get_search_info()
