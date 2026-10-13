"""
Encode game state thành tensor cho neural network
"""
import numpy as np

from ..board import Board
from .game_result import NO_CAPTURE_DRAW_PLY_LIMIT
from .move_history import MoveHistory
from .position_tracker import PositionTracker, REPETITION_DRAW_COUNT


class StateEncoder:
    """Encode game state thành tensor cho neural network"""
    
    @staticmethod
    def to_tensor(
        board: Board,
        is_red_turn: bool,
        move_history: MoveHistory,
        position_tracker: PositionTracker,
        no_capture_limit: int = NO_CAPTURE_DRAW_PLY_LIMIT,
        max_plies: int = 200
    ) -> np.ndarray:
        """
        Chuyển game state thành tensor
        
        Args:
            board: Bàn cờ hiện tại
            is_red_turn: Lượt của ai
            move_history: Lịch sử nước đi
            position_tracker: Tracker vị trí
            
        Returns:
            np.ndarray shape (19, 10, 9)
            - Channels 0-15: Board features (từ Board.to_numpy)
            - Channel 16: Signed color and normalized total-ply progress
            - Channel 17: Normalized moves since the most recent capture
            - Channel 18: Position repetition count
        """
        # Lấy board representation từ góc nhìn người chơi hiện tại
        board_array = board.to_numpy(red_perspective=is_red_turn)
        
        # Sign preserves absolute color; magnitude exposes the total-ply draw clock.
        progress = min(move_history.get_move_count() / max_plies, 1.0)
        board_array[16, :, :] = (1.0 if is_red_turn else -1.0) * (0.5 + 0.5 * progress)
        
        # Channel 17: Progress toward the no-capture draw threshold
        board_array[17, :, :] = StateEncoder._encode_no_capture_count(move_history, no_capture_limit)
        
        # Channel 18: Position repetition count
        board_array[18, :, :] = StateEncoder._encode_repetition(position_tracker)
        
        return board_array
    
    @staticmethod
    def _encode_turn(is_red_turn: bool) -> float:
        """
        Encode lượt chơi hiện tại
        
        Args:
            is_red_turn: True nếu đến lượt đỏ
            
        Returns:
            1.0 nếu đỏ, 0.0 nếu đen
        """
        return 1.0 if is_red_turn else 0.0
    
    @staticmethod
    def _encode_no_capture_count(move_history: MoveHistory, no_capture_limit: int = NO_CAPTURE_DRAW_PLY_LIMIT) -> float:
        """
        Encode số nước đi kể từ lần ăn quân gần nhất (normalized)
        
        Args:
            move_history: Lịch sử nước đi
            
        Returns:
            Giá trị từ 0.0 đến 1.0, đạt 1.0 tại ngưỡng hòa
        """
        no_capture_count = move_history.count_moves_since_capture()
        return min(no_capture_count / no_capture_limit, 1.0)
    
    @staticmethod
    def _encode_repetition(position_tracker: PositionTracker) -> float:
        """
        Encode số lần lặp lại vị trí (normalized)
        
        Args:
            position_tracker: Tracker vị trí
            
        Returns:
            Giá trị từ 0.0 đến 1.0 (cap tại REPETITION_DRAW_COUNT)
        """
        rep_count = position_tracker.count_current_repetitions()
        return min(rep_count / REPETITION_DRAW_COUNT, 1.0)
