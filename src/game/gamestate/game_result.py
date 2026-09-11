"""
Logic kiểm tra kết thúc game và xác định kết quả
"""
from typing import Optional

from ..board import Board
from ..rules import Rules
from .move_history import MoveHistory
from .position_tracker import PositionTracker


# Giới hạn số nước đi không ăn quân để hòa (50-move rule tương tự)
NO_CAPTURE_DRAW_PLY_LIMIT = 100


class GameResult:
    """Quản lý logic kiểm tra kết thúc game và kết quả"""
    
    @staticmethod
    def is_terminal(
        board: Board,
        is_red_turn: bool,
        move_history: MoveHistory,
        position_tracker: PositionTracker
    ) -> bool:
        """
        Kiểm tra game đã kết thúc chưa
        
        Args:
            board: Bàn cờ hiện tại
            is_red_turn: Lượt của ai
            move_history: Lịch sử nước đi
            position_tracker: Tracker vị trí
            
        Returns:
            True nếu game đã kết thúc
        """
        # Chiếu hết (checkmate)
        if Rules.is_checkmate(board, is_red_turn):
            return True
        
        # Hòa vì không có nước đi hợp lệ (stalemate)
        if Rules.is_stalemate(board, is_red_turn):
            return True
        
        # Hòa vì lặp lại vị trí 3 lần
        if position_tracker.is_draw_by_repetition():
            return True
        
        # Hòa vì quá nhiều nước đi không ăn quân
        if move_history.count_moves_since_capture() >= NO_CAPTURE_DRAW_PLY_LIMIT:
            return True
        
        return False
    
    @staticmethod
    def get_winner(
        board: Board,
        is_red_turn: bool,
        move_history: MoveHistory,
        position_tracker: PositionTracker
    ) -> Optional[bool]:
        """
        Xác định người thắng
        
        Args:
            board: Bàn cờ hiện tại
            is_red_turn: Lượt của ai
            move_history: Lịch sử nước đi
            position_tracker: Tracker vị trí
            
        Returns:
            True nếu đỏ thắng, False nếu đen thắng, None nếu hòa hoặc chưa kết thúc
        """
        # Chưa kết thúc
        if not GameResult.is_terminal(board, is_red_turn, move_history, position_tracker):
            return None
        
        # Chiếu hết - bên kia thắng
        if Rules.is_checkmate(board, is_red_turn):
            return not is_red_turn
        
        # Các trường hợp khác đều là hòa
        return None
    
    @staticmethod
    def get_result_value(
        board: Board,
        is_red_turn: bool,
        move_history: MoveHistory,
        position_tracker: PositionTracker
    ) -> float:
        """
        Lấy giá trị kết quả từ góc nhìn của bên hiện tại
        
        Args:
            board: Bàn cờ hiện tại
            is_red_turn: Lượt của ai (bên đang xét)
            move_history: Lịch sử nước đi
            position_tracker: Tracker vị trí
            
        Returns:
            1.0 nếu bên hiện tại thắng
            -1.0 nếu bên hiện tại thua
            0.0 nếu hòa hoặc chưa kết thúc
        """
        winner = GameResult.get_winner(
            board, is_red_turn, move_history, position_tracker
        )
        
        if winner is None:
            return 0.0
        
        return 1.0 if winner == is_red_turn else -1.0
    
    @staticmethod
    def get_termination_reason(
        board: Board,
        is_red_turn: bool,
        move_history: MoveHistory,
        position_tracker: PositionTracker
    ) -> str:
        """
        Lấy lý do game kết thúc
        
        Returns:
            String mô tả lý do kết thúc
        """
        if not GameResult.is_terminal(board, is_red_turn, move_history, position_tracker):
            return "Game in progress"
        
        if Rules.is_checkmate(board, is_red_turn):
            winner = "Red" if not is_red_turn else "Black"
            return f"Checkmate - {winner} wins"
        
        if Rules.is_stalemate(board, is_red_turn):
            return "Draw - Stalemate (no legal moves)"
        
        if position_tracker.is_draw_by_repetition():
            rep_count = position_tracker.count_current_repetitions()
            return f"Draw - Position repeated {rep_count} times"
        
        if move_history.count_moves_since_capture() >= NO_CAPTURE_DRAW_PLY_LIMIT:
            move_count = move_history.count_moves_since_capture()
            return f"Draw - {move_count} moves without capture"
        
        return "Draw"
