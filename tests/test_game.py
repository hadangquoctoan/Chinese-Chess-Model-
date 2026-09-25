"""Test game logic"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.game.game_state import GameState
from src.game.board import Board
from src.game.gamestate.game_result import NO_CAPTURE_DRAW_PLY_LIMIT
from src.game.gamestate.move_history import MoveHistory
from src.game.gamestate.position_tracker import PositionTracker
from src.game.gamestate.state_encoder import StateEncoder
from src.game.pieces import PieceType, create_piece
from src.game.rules import Rules
from scripts.play_cli import parse_move

def test_initial_position():
    """Test khởi tạo bàn cờ"""
    game = GameState()
    
    # Test lượt đi
    assert game.is_red_turn, "Red should move first"
    
    # Test legal moves
    legal_moves = game.get_legal_moves()
    assert len(legal_moves) > 0, "Should have legal moves"
    
    print(f"[PASS] Initial position test passed. {len(legal_moves)} legal moves.")

def test_make_move():
    """Test thực hiện nước đi"""
    game = GameState()
    
    # Get first legal move
    legal_moves = game.get_legal_moves()
    move = legal_moves[0]
    
    from_row, from_col, to_row, to_col = move
    
    # Make move
    success = game.make_move(from_row, from_col, to_row, to_col)
    assert success, "Move should be successful"
    
    # Check turn changed
    assert not game.is_red_turn, "Turn should change to black"
    
    print("[PASS] Make move test passed.")

def test_undo_move():
    """Test hoàn tác nước đi"""
    game = GameState()
    
    initial_turn = game.is_red_turn
    legal_moves = game.get_legal_moves()
    move = legal_moves[0]
    
    # Make and undo move
    game.make_move(*move)
    game.undo_move()
    
    assert game.is_red_turn == initial_turn, "Turn should be restored"
    
    print("[PASS] Undo move test passed.")

def test_game_over():
    """Test phát hiện kết thúc game"""
    game = GameState()
    
    # Initial position should not be terminal
    assert not game.is_terminal(), "Initial position should not be terminal"
    
    print("[PASS] Game over detection test passed.")


def test_generals_are_not_captured():
    """A legal move must not remove the opposing general from the board."""
    game = GameState()
    empty_piece = create_piece(PieceType.EMPTY, is_red=True)
    for row in range(game.board.rows):
        for col in range(game.board.cols):
            game.board.set_piece(row, col, empty_piece)

    game.board.set_piece(0, 4, create_piece(PieceType.GENERAL, is_red=True))
    game.board.set_piece(8, 4, create_piece(PieceType.CHARIOT, is_red=True))
    game.board.set_piece(9, 4, create_piece(PieceType.GENERAL, is_red=False))

    assert not game.make_move(8, 4, 9, 4), "Generals are checkmated, not captured"
    assert game.board.find_general(is_red=False) == (9, 4)

    print("[PASS] General capture prevention test passed.")


def test_action_decoding_returns_a_legal_move():
    """An encoded legal move must decode to a legal move for the same player."""
    game = GameState()
    legal_moves = game.get_legal_moves()
    action_index = Rules.move_to_action_index(*legal_moves[0])
    decoded_move = Rules.action_index_to_move(
        action_index,
        game.board,
        game.is_red_turn,
    )

    assert decoded_move in legal_moves
    assert Rules.action_index_to_move(-1, game.board, game.is_red_turn) is None
    assert Rules.action_index_to_move(1800, game.board, game.is_red_turn) is None

    print("[PASS] Action encoding test passed.")


def test_move_parser_rejects_malformed_input():
    """The CLI parser should accept only one-digit board coordinates."""
    assert parse_move('a0-a1') == (0, 0, 1, 0)
    assert parse_move('a10-a9') is None
    assert parse_move('z0-a1') is None
    assert parse_move('a0/a1') is None

    print("[PASS] Move parser validation test passed.")


def test_replay_buffer_rejects_invalid_batches():
    """Replay buffer should fail early for invalid training data."""
    from src.training import ReplayBuffer

    buffer = ReplayBuffer(max_size=2)
    try:
        buffer.sample(1)
    except ValueError:
        pass
    else:
        raise AssertionError("empty buffer sampling should fail")

    try:
        buffer.add_game([], [object()], 0.0)
    except ValueError:
        pass
    else:
        raise AssertionError("mismatched game data should fail")

    print("[PASS] Replay buffer validation test passed.")


def test_channel_17_tracks_moves_since_capture():
    """Channel 17 must follow and reset the no-capture draw counter."""
    history = MoveHistory()
    tracker = PositionTracker()
    board = Board()

    for _ in range(25):
        history.add_move(0, 0, 0, 1, captured_piece=None)

    tensor = StateEncoder.to_tensor(board, True, history, tracker)
    assert (tensor[17] == 0.25).all()

    threshold_history = MoveHistory()
    for _ in range(NO_CAPTURE_DRAW_PLY_LIMIT + 1):
        threshold_history.add_move(0, 0, 0, 1, captured_piece=None)
    tensor = StateEncoder.to_tensor(board, True, threshold_history, tracker)
    assert (tensor[17] == 1.0).all()

    captured_piece = create_piece(PieceType.SOLDIER, is_red=False)
    history.add_move(0, 0, 0, 1, captured_piece=captured_piece)
    tensor = StateEncoder.to_tensor(board, True, history, tracker)
    assert (tensor[17] == 0.0).all()

    history.add_move(0, 0, 0, 1, captured_piece=None)
    history.add_move(0, 0, 0, 1, captured_piece=None)
    tensor = StateEncoder.to_tensor(board, True, history, tracker)
    assert (tensor[17] == 0.02).all()

    history.pop_last_move()
    history.pop_last_move()
    history.pop_last_move()
    tensor = StateEncoder.to_tensor(board, True, history, tracker)
    assert (tensor[17] == 0.25).all()

    print("[PASS] Channel 17 no-capture counter test passed.")


if __name__ == '__main__':
    print("Running game logic tests...\n")
    
    test_initial_position()
    test_make_move()
    test_undo_move()
    test_game_over()
    test_generals_are_not_captured()
    test_action_decoding_returns_a_legal_move()
    test_move_parser_rejects_malformed_input()
    test_replay_buffer_rejects_invalid_batches()
    test_channel_17_tracks_moves_since_capture()
    
    print("\n[SUCCESS] All tests passed!")
