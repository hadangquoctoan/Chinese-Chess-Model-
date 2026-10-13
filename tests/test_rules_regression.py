"""Position-based regression tests for the explicit training ruleset."""
import numpy as np
import pytest

from src.game.game_state import GameState
from src.game.pieces import PieceType as P, create_piece
from src.game.rules import Rules


def position(pieces, red=True, **options):
    game = GameState(**options)
    for row in range(10):
        for col in range(9):
            game.board.set_piece(row, col, create_piece(P.EMPTY, True))
    for row, col, kind, color in pieces:
        game.board.set_piece(row, col, create_piece(kind, color))
    game.is_red_turn = red
    game.reset_history()
    return game


def test_stalemate_is_loss_not_draw():
    game = GameState()
    for row in range(10):
        for col in range(9):
            game.board.set_piece(row, col, create_piece(P.EMPTY, True))
    for row, col, kind, red in [(0, 4, P.GENERAL, True), (9, 4, P.GENERAL, False),
                                (5, 4, P.SOLDIER, False), (1, 0, P.CHARIOT, False),
                                (2, 3, P.CHARIOT, False), (2, 5, P.CHARIOT, False)]:
        game.board.set_piece(row, col, create_piece(kind, red))
    assert not Rules.is_in_check(game.board, True)
    assert Rules.is_stalemate(game.board, True)
    assert game.is_terminal()
    assert game.get_winner() is False
    assert game.get_game_result() == -1.0
    assert 'Stalemate' in game.get_termination_reason()


def test_copies_isolate_mutable_pieces_and_capture_records():
    game = GameState()
    assert game.make_move(2, 1, 9, 1)
    clone = game.copy()
    clone.board.get_piece(0, 0).is_red = False
    clone._move_history.get_last_move().captured_piece.is_red = True
    clone._move_history.get_last_move().from_row = 8
    assert game.board.get_piece(0, 0).is_red
    assert not game._move_history.get_last_move().captured_piece.is_red
    assert game._move_history.get_last_move().from_row == 2
    assert clone.position_history is not game.position_history


@pytest.mark.parametrize('limit_name', ['max_plies', 'no_capture_limit'])
def test_configured_limits_are_terminal_guarded_and_undoable(limit_name):
    game = GameState(**{limit_name: 2})
    assert game.make_move(3, 0, 4, 0)
    assert game.make_move(6, 0, 5, 0)
    assert game.is_terminal()
    assert game.get_winner() is None
    assert game.get_game_result() == 0.0
    assert game.get_legal_moves() == []
    assert not game.make_move(3, 2, 4, 2)
    clone = game.copy()
    assert getattr(clone, limit_name) == 2
    assert clone.is_terminal()
    assert game.undo_move()
    assert not game.is_terminal()
    assert game.get_legal_moves()
    if limit_name == 'no_capture_limit':
        assert (game.to_tensor()[17] == 0.5).all()


def test_reset_history_initializes_edited_position_and_counters():
    game = GameState()
    assert game.make_move(3, 0, 4, 0)
    game.reset_history()
    assert game.move_count == 0
    assert len(game.position_history) == 1
    assert game.position_history[0][0] is False
    assert (game.to_tensor()[17] == 0.0).all()
    assert not game.undo_move()


@pytest.mark.parametrize('color', [True, False])
@pytest.mark.parametrize('duplicate', [False, True])
def test_invalid_general_counts_cannot_produce_moves(color, duplicate):
    game = GameState()
    row = 0 if color else 9
    if duplicate:
        game.board.set_piece(1 if color else 8, 4, create_piece(P.GENERAL, color))
    else:
        game.board.set_piece(row, 4, create_piece(P.EMPTY, True))
    assert not Rules.is_move_legal(game.board, 3, 0, 4, 0, True)
    assert Rules.get_all_legal_moves(game.board, True) == []
    with pytest.raises(ValueError, match='general'):
        game.reset_history()
    with pytest.raises(ValueError, match='general'):
        game.is_terminal()


@pytest.mark.parametrize('options', [{'max_plies': 0}, {'no_capture_limit': -1},
                                     {'max_plies': True}, {'no_capture_limit': 1.5}])
def test_invalid_limits_raise(options):
    with pytest.raises(ValueError):
        GameState(**options)


CHECK_CYCLE = [(7, 5, 7, 4), (9, 4, 9, 5), (7, 4, 7, 5), (9, 5, 9, 4)]
CHASE_CYCLE = [(4, 2, 4, 1), (6, 1, 8, 2), (4, 1, 4, 2), (8, 2, 6, 1)]


def play_cycle(game, cycle):
    for move in cycle * 2:
        assert not game.is_terminal(), (move, game.get_termination_reason())
        assert game.make_move(*move), move
    assert game.is_terminal()
    assert game.get_legal_moves() == []


@pytest.mark.parametrize('red_checker', [True, False])
def test_actual_perpetual_check_loses_for_checker(red_checker):
    pieces = [(0, 3, P.GENERAL, True), (9, 4, P.GENERAL, False),
              (7, 5, P.CHARIOT, True)]
    cycle = CHECK_CYCLE
    if not red_checker:
        pieces = [(9-r, c, kind, not red) for r, c, kind, red in pieces]
        cycle = [(9-fr, fc, 9-tr, tc) for fr, fc, tr, tc in cycle]
    game = position(pieces, red=red_checker, max_plies=8)
    play_cycle(game, cycle)
    assert game.get_winner() is not red_checker
    assert 'Perpetual check' in game.get_termination_reason()
    assert game.get_game_result() == -1.0
    records = list(game._move_history)
    assert all(record.gave_check for record in records[::2])
    assert all(not record.gave_check for record in records[1::2])
    assert game.undo_move()
    assert not game.is_terminal()
    assert game.make_move(*cycle[-1])
    assert game.get_winner() is not red_checker


def test_actual_neutral_repetition_draw_and_undo():
    game = position([(0, 3, P.GENERAL, True), (9, 5, P.GENERAL, False),
                     (4, 0, P.CHARIOT, True), (6, 8, P.CHARIOT, False)])
    play_cycle(game, [(4, 0, 4, 1), (6, 8, 6, 7), (4, 1, 4, 0), (6, 7, 6, 8)])
    assert game.get_winner() is None
    assert game.get_game_result() == 0.0
    assert game.undo_move()
    assert not game.is_terminal()


def chase_position(extra=()):
    return position([(0, 3, P.GENERAL, True), (9, 5, P.GENERAL, False),
                     (4, 2, P.CHARIOT, True), (6, 1, P.HORSE, False), *extra])


@pytest.mark.parametrize('red_chaser', [True, False])
def test_actual_unprotected_same_piece_perpetual_chase_loses(red_chaser):
    game = chase_position()
    cycle = CHASE_CYCLE
    if not red_chaser:
        pieces = [(9-r, c, piece.piece_type, not piece.is_red)
                  for color in (True, False)
                  for r, c, piece in game.board.get_all_pieces(color)]
        game = position(pieces, red=False)
        cycle = [(9-fr, fc, 9-tr, tc) for fr, fc, tr, tc in cycle]
    play_cycle(game, cycle)
    assert game.get_winner() is (not red_chaser)
    assert 'Perpetual chase' in game.get_termination_reason()
    records = list(game._move_history)[::2]
    assert all(record.chased_piece_ids for record in records)
    assert len(set.intersection(*(set(record.chased_piece_ids) for record in records))) == 1


def test_protected_chase_is_exempt_and_repeats_draw():
    game = chase_position([(6, 8, P.CHARIOT, False), (8, 8, P.CHARIOT, False)])
    play_cycle(game, CHASE_CYCLE)
    assert game.get_winner() is None
    assert all(not record.chased_piece_ids for record in list(game._move_history)[::2])


@pytest.mark.parametrize('attacker,target,origin,destination', [
    (P.CHARIOT, P.SOLDIER, (4, 2), (4, 1)),
    (P.GENERAL, P.HORSE, (1, 3), (1, 4)),
    (P.SOLDIER, P.HORSE, (4, 4), (5, 4)),
])
def test_training_chase_exempts_soldier_targets_and_general_soldier_attackers(
        attacker, target, origin, destination):
    generals = [(0, 3, P.GENERAL, True), (9, 5, P.GENERAL, False)]
    if attacker == P.GENERAL:
        generals = generals[1:]
    target_pos = (2, 4) if attacker == P.GENERAL else ((6, 4) if attacker == P.SOLDIER else (6, 1))
    game = position([*generals, (*origin, attacker, True), (*target_pos, target, False)])
    assert game.make_move(*origin, *destination)
    assert not game._move_history.get_last_move().chased_piece_ids


def test_cannon_requires_exactly_one_screen_for_capture():
    game = position([(0, 3, P.GENERAL, True), (9, 5, P.GENERAL, False),
                     (4, 0, P.CANNON, True), (4, 5, P.HORSE, False)])
    cannon = game.board.get_piece(4, 0)
    assert (4, 5) not in cannon.get_possible_moves(4, 0, game.board)
    game.board.set_piece(4, 2, create_piece(P.SOLDIER, True))
    moves = cannon.get_possible_moves(4, 0, game.board)
    assert (4, 5) in moves
    assert (4, 3) not in moves
    assert (4, 2) not in moves
    game.board.set_piece(4, 3, create_piece(P.SOLDIER, False))
    assert (4, 5) not in cannon.get_possible_moves(4, 0, game.board)


def test_horse_leg_and_elephant_eye_and_river():
    game = position([(0, 3, P.GENERAL, True), (9, 5, P.GENERAL, False),
                     (2, 2, P.HORSE, True), (2, 6, P.ELEPHANT, True)])
    horse = game.board.get_piece(2, 2)
    assert set(horse.get_possible_moves(2, 2, game.board)) == {
        (3, 4), (3, 0), (1, 4), (1, 0), (4, 3), (4, 1), (0, 1)
    }  # (0, 3) contains its own general.
    game.board.set_piece(3, 2, create_piece(P.SOLDIER, False))
    assert (4, 1) not in horse.get_possible_moves(2, 2, game.board)
    assert (4, 3) not in horse.get_possible_moves(2, 2, game.board)
    elephant = game.board.get_piece(2, 6)
    assert (4, 8) in elephant.get_possible_moves(2, 6, game.board)
    game.board.set_piece(3, 7, create_piece(P.SOLDIER, True))
    assert (4, 8) not in elephant.get_possible_moves(2, 6, game.board)
    assert all(r <= 4 for r, c in elephant.get_possible_moves(4, 6, game.board))
    black = create_piece(P.ELEPHANT, False)
    assert all(r >= 5 for r, c in black.get_possible_moves(5, 6, game.board))


@pytest.mark.parametrize('red', [True, False])
def test_palace_and_soldier_direction_and_river(red):
    game = GameState()
    row = 0 if red else 9
    general = create_piece(P.GENERAL, red)
    advisor = create_piece(P.ADVISOR, red)
    assert all(general.in_palace(r, c, red) for r, c in general.get_possible_moves(row, 4, game.board))
    assert all(advisor.in_palace(r, c, red) for r, c in advisor.get_possible_moves(row, 3, game.board))
    soldier = create_piece(P.SOLDIER, red)
    before, after, delta = (4, 5, 1) if red else (5, 4, -1)
    for r in range(10):
        for c in range(9):
            game.board.set_piece(r, c, create_piece(P.EMPTY, True))
    assert soldier.get_possible_moves(before, 4, game.board) == [(before+delta, 4)]
    assert set(soldier.get_possible_moves(after, 4, game.board)) == {(after+delta, 4), (after, 3), (after, 5)}
    edge = 9 if red else 0
    assert set(soldier.get_possible_moves(edge, 0, game.board)) == {(edge, 1)}


def test_flying_generals_and_pinned_blocker_cannot_move_aside():
    game = position([(0, 4, P.GENERAL, True), (9, 4, P.GENERAL, False),
                     (4, 4, P.CHARIOT, True)])
    assert not Rules.is_in_check(game.board, True)
    assert not Rules.is_move_legal(game.board, 4, 4, 4, 5, True)
    assert Rules.is_move_legal(game.board, 4, 4, 5, 4, True)
    game.board.set_piece(4, 4, create_piece(P.EMPTY, True))
    assert Rules.is_in_check(game.board, True)
    assert Rules.is_in_check(game.board, False)


def test_actual_checkmate_loss_and_terminal_guard():
    game = position([(0, 4, P.GENERAL, True), (9, 4, P.GENERAL, False),
                     (5, 4, P.CHARIOT, False), (1, 0, P.CHARIOT, False),
                     (2, 3, P.CHARIOT, False), (2, 5, P.CHARIOT, False)], max_plies=1)
    assert Rules.is_checkmate(game.board, True)
    assert game.get_winner() is False
    assert game.get_game_result() == -1.0
    assert 'Checkmate' in game.get_termination_reason()
    assert not game.make_move(0, 4, 0, 3)


def test_capture_counter_and_tensor_reset_and_undo_restore_exact_state():
    game = GameState(no_capture_limit=7)
    assert game.make_move(3, 0, 4, 0)
    assert game.make_move(6, 0, 5, 0)
    before = game.to_tensor().copy()
    history = game.position_history
    assert game.make_move(4, 0, 5, 0)
    assert (game.to_tensor()[17] == 0.0).all()
    assert game.undo_move()
    np.testing.assert_array_equal(game.to_tensor(), before)
    assert game.position_history == history
    assert game.board.get_piece(5, 0).piece_type == P.SOLDIER
    assert not game.board.get_piece(5, 0).is_red


def test_mutual_continuous_check_metadata_draws():
    from src.game.gamestate.repetition import adjudicate
    game = position([(0, 3, P.GENERAL, True), (9, 5, P.GENERAL, False),
                     (4, 0, P.CHARIOT, True), (6, 8, P.CHARIOT, False)])
    play_cycle(game, [(4, 0, 4, 1), (6, 8, 6, 7), (4, 1, 4, 0), (6, 7, 6, 8)])
    # Branch-level metadata test, not claimed as an actual mutual-check position.
    for record in game._move_history:
        record.gave_check = True
    assert adjudicate(game._move_history, game._position_tracker) == (None, 'Draw - Mutual perpetual check')


def test_mutual_chase_and_alternating_targets_do_not_select_a_loser():
    from src.game.gamestate.repetition import adjudicate
    game = chase_position()
    play_cycle(game, CHASE_CYCLE)
    for record in game._move_history:
        record.chased_piece_ids = frozenset({123})
    assert adjudicate(game._move_history, game._position_tracker)[0] is None
    for index, record in enumerate(game._move_history):
        record.chased_piece_ids = frozenset({index})
    assert adjudicate(game._move_history, game._position_tracker)[0] is None


def test_pinned_capture_and_already_existing_threat_are_not_new_chase():
    from src.game.gamestate.repetition import chase_targets
    game = position([(0, 4, P.GENERAL, True), (9, 4, P.GENERAL, False),
                     (4, 4, P.CHARIOT, True), (4, 6, P.HORSE, False)])
    assert chase_targets(game.board, 4, 4) == frozenset()
    game = position([(0, 3, P.GENERAL, True), (9, 5, P.GENERAL, False),
                     (4, 1, P.CHARIOT, True), (6, 1, P.HORSE, False)])
    assert chase_targets(game.board, 4, 1)
    assert game.make_move(4, 1, 5, 1)
    assert game._move_history.get_last_move().chased_piece_ids == frozenset()


def test_ruleset_and_action_versions_are_public():
    from src.game.gamestate.game_result import RULESET_VERSION
    from src.game.rules import ACTION_ENCODING_VERSION
    assert RULESET_VERSION == 'xiangqi-training-v2'
    assert ACTION_ENCODING_VERSION == 'piece-slots-v2'



