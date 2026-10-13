"""Policy slots are absolute, piece-conditioned, and blocker-independent."""
import random

import pytest

from src.game.board import Board
from src.game.game_state import GameState
from src.game.pieces import PieceType, create_piece
from src.game.rules import Rules


def test_initial_cannon_382_collision_regression():
    game = GameState()
    moves = [(2, 1, 2, 4), (2, 1, 9, 1)]
    assert all(move in game.get_legal_moves() for move in moves)
    # Both were index 382 under destination modulo 20.
    indices = [Rules.move_to_action_index(*move, board=game.board) for move in moves]
    assert len(set(indices)) == 2
    for index, move in zip(indices, moves):
        assert Rules.action_index_to_move(index, game.board, True) == move


def geometric_destinations(kind, red, row, col):
    if kind in (PieceType.CANNON, PieceType.CHARIOT):
        return sorted([(r, col) for r in range(10) if r != row] +
                      [(row, c) for c in range(9) if c != col])
    if kind == PieceType.HORSE:
        deltas = [(dr, dc) for dr in (-2, -1, 1, 2) for dc in (-2, -1, 1, 2)
                  if abs(dr) + abs(dc) == 3]
    elif kind == PieceType.ELEPHANT:
        deltas = [(dr, dc) for dr in (-2, 2) for dc in (-2, 2)]
    elif kind == PieceType.ADVISOR:
        deltas = [(dr, dc) for dr in (-1, 1) for dc in (-1, 1)]
    elif kind == PieceType.GENERAL:
        deltas = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    else:
        deltas = [(1 if red else -1, 0), (0, -1), (0, 1)]
    return sorted((row + dr, col + dc) for dr, dc in deltas
                  if 0 <= row + dr < 10 and 0 <= col + dc < 9)


@pytest.mark.parametrize('red', [True, False])
@pytest.mark.parametrize('kind', [kind for kind in PieceType if kind != PieceType.EMPTY])
def test_exhaustive_geometric_slots_are_unique_and_stable(kind, red):
    board = Board()
    for row in range(10):
        for col in range(9):
            board.set_piece(row, col, create_piece(kind, red))
            destinations = geometric_destinations(kind, red, row, col)
            indices = [Rules.move_to_action_index(row, col, *dest, board=board)
                       for dest in destinations]
            assert len(indices) == len(set(indices))
            assert all((row * 9 + col) * 20 <= index < (row * 9 + col + 1) * 20
                       for index in indices)
            assert [index % 20 for index in indices] == list(range(len(destinations)))
            changed = board.copy()
            for r, c in destinations:
                changed.set_piece(r, c, create_piece(PieceType.SOLDIER, not red))
            assert indices == [Rules.move_to_action_index(row, col, *dest, board=changed)
                               for dest in destinations]


def assert_roundtrips(game):
    for red in (True, False):
        legal = Rules.get_all_legal_moves(game.board, red)
        indices = [Rules.move_to_action_index(*move, board=game.board) for move in legal]
        assert len(indices) == len(set(indices))
        for move, index in zip(legal, indices):
            assert Rules.action_index_to_move(index, game.board, red) == move


def test_initial_and_seeded_random_legal_positions_roundtrip():
    rng = random.Random(382)
    game = GameState()
    assert_roundtrips(game)
    for _ in range(60):
        moves = game.get_legal_moves()
        if not moves:
            break
        assert game.make_move(*rng.choice(moves))
        assert_roundtrips(game)


@pytest.mark.parametrize('index', [-1, 1800, 1.5, True, None])
def test_decoder_rejects_invalid_indices(index):
    board = Board()
    assert Rules.action_index_to_move(index, board, True) is None


def test_encoding_rejects_empty_origin_and_non_geometric_destination():
    board = Board()
    with pytest.raises(ValueError):
        Rules.move_to_action_index(4, 4, 5, 4, board=board)
    with pytest.raises(ValueError):
        Rules.move_to_action_index(0, 0, 1, 1, board=board)
    with pytest.raises(ValueError):
        Rules.move_to_action_index(-1, 0, 1, 0, board=board)
