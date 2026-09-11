"""
Theo dõi vị trí bàn cờ và phát hiện lặp lại
"""
from typing import List, Tuple

from ..board import Board
from ..pieces import PieceType


# Số lần lặp lại vị trí để hòa
REPETITION_DRAW_COUNT = 3


class PositionTracker:
    """Theo dõi các vị trí bàn cờ để phát hiện lặp lại"""
    
    def __init__(self):
        """Khởi tạo tracker trống"""
        self._positions: List[Tuple] = []
    
    def save_position(self, board: Board, is_red_turn: bool) -> None:
        """
        Lưu vị trí hiện tại
        
        Args:
            board: Bàn cờ hiện tại
            is_red_turn: Lượt của ai
        """
        position = self._encode_position(board, is_red_turn)
        self._positions.append(position)
    
    def remove_last_position(self) -> None:
        """Xóa vị trí cuối cùng (dùng cho undo)"""
        if self._positions:
            self._positions.pop()
    
    def count_current_repetitions(self) -> int:
        """
        Đếm số lần vị trí hiện tại xuất hiện
        
        Returns:
            Số lần lặp lại (ít nhất là 1 nếu có vị trí)
        """
        if not self._positions:
            return 0
        
        current_position = self._positions[-1]
        return self._positions.count(current_position)

    def get_positions(self) -> List[Tuple]:
        """Return a snapshot of the recorded positions."""
        return self._positions.copy()
    
    def is_draw_by_repetition(self) -> bool:
        """
        Kiểm tra hòa do lặp lại vị trí
        
        Returns:
            True nếu vị trí lặp lại >= REPETITION_DRAW_COUNT
        """
        return self.count_current_repetitions() >= REPETITION_DRAW_COUNT
    
    def _encode_position(self, board: Board, is_red_turn: bool) -> Tuple:
        """
        Encode vị trí bàn cờ thành tuple để so sánh
        
        Args:
            board: Bàn cờ cần encode
            is_red_turn: Lượt của ai
            
        Returns:
            Tuple đại diện cho vị trí unique
            
        Note:
            Cùng bố trí quân nhưng khác lượt đi = vị trí khác nhau
        """
        pieces = tuple(
            (row, col, piece.piece_type.value, piece.is_red)
            for row in range(board.rows)
            for col in range(board.cols)
            if (piece := board.get_piece(row, col)).piece_type != PieceType.EMPTY
        )
        
        # Vị trí = (lượt đi, bố trí quân)
        return (is_red_turn, pieces)
    
    def copy(self) -> 'PositionTracker':
        """Tạo bản sao của tracker"""
        new_tracker = PositionTracker()
        new_tracker._positions = self._positions.copy()
        return new_tracker
    
    def clear(self) -> None:
        """Xóa toàn bộ lịch sử vị trí"""
        self._positions.clear()
    
    def __len__(self) -> int:
        return len(self._positions)
    
    def __repr__(self) -> str:
        return f"PositionTracker({len(self._positions)} positions)"
