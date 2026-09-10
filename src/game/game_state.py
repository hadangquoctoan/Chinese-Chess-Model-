"""
Trạng thái ván cờ - quản lý lịch sử nước đi và kiểm tra kết thúc game
"""
from typing import List, Tuple, Optional
import numpy as np
from .board import Board
from .rules import Rules
from .pieces import PieceType, create_piece

REPETITION_DRAW_COUNT = 3
NO_CAPTURE_DRAW_PLY_LIMIT = 100

class GameState:
    """Quản lý trạng thái hoàn chỉnh của ván cờ"""
    
    def __init__(self):
        """Khởi tạo trạng thái game mới"""
        self.board = Board()
        self.is_red_turn = True  # Đỏ đi trước
        self.move_history = []  # List of (from_row, from_col, to_row, to_col, captured_piece)
        self.position_history = []  # List of board states for repetition detection
        self.move_count = 0
        
        # Save initial position
        self._save_position()
    
    def _save_position(self):
        """Lưu vị trí hiện tại để kiểm tra lặp lại"""
        pieces = tuple(
            (row, col, piece.piece_type.value, piece.is_red)
            for row in range(self.board.rows)
            for col in range(self.board.cols)
            if (piece := self.board.get_piece(row, col)).piece_type != PieceType.EMPTY
        )
        # The same layout with a different player to move is a different position.
        position = (self.is_red_turn, pieces)
        self.position_history.append(position)
    
    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int) -> bool:
        """
        Thực hiện nước đi
        
        Returns:
            True nếu nước đi hợp lệ
        """
        # Kiểm tra hợp lệ
        if not Rules.is_move_legal(self.board, from_row, from_col, to_row, to_col, self.is_red_turn):
            return False
        
        # Thực hiện di chuyển
        captured = self.board.move_piece(from_row, from_col, to_row, to_col)
        
        # Lưu lịch sử
        self.move_history.append((from_row, from_col, to_row, to_col, captured))

        # Chuyển lượt
        self.is_red_turn = not self.is_red_turn
        self.move_count += 1
        self._save_position()
        
        return True
    
    def undo_move(self) -> bool:
        """
        Hoàn tác nước đi cuối
        
        Returns:
            True nếu có nước đi để hoàn tác
        """
        if not self.move_history:
            return False
        
        from_row, from_col, to_row, to_col, captured = self.move_history.pop()
        
        # Di chuyển quân về vị trí cũ
        piece = self.board.get_piece(to_row, to_col)
        self.board.set_piece(from_row, from_col, piece)
        
        # Khôi phục quân bị ăn
        if captured:
            self.board.set_piece(to_row, to_col, captured)
        else:
            self.board.set_piece(
                to_row,
                to_col,
                create_piece(PieceType.EMPTY, is_red=True),
            )
        
        # Khôi phục trạng thái
        self.position_history.pop()
        self.is_red_turn = not self.is_red_turn
        self.move_count -= 1
        
        return True
    
    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """Lấy tất cả nước đi hợp lệ cho lượt hiện tại"""
        return Rules.get_all_legal_moves(self.board, self.is_red_turn)
    
    def is_terminal(self) -> bool:
        """Kiểm tra game đã kết thúc chưa"""
        # Chiếu hết
        if Rules.is_checkmate(self.board, self.is_red_turn):
            return True
        
        # Hòa (không có nước đi)
        if Rules.is_stalemate(self.board, self.is_red_turn):
            return True
        
        # Lặp lại vị trí 3 lần (hòa)
        if self._count_position_repetitions() >= REPETITION_DRAW_COUNT:
            return True

        # Quá nhiều nước đi không ăn quân (hòa)
        if self._moves_since_capture() >= NO_CAPTURE_DRAW_PLY_LIMIT:
            return True
        
        return False
    
    def get_winner(self) -> Optional[bool]:
        """
        Lấy người thắng
        
        Returns:
            True nếu đỏ thắng, False nếu đen thắng, None nếu hòa hoặc chưa kết thúc
        """
        if not self.is_terminal():
            return None
        
        # Chiếu hết - bên kia thắng
        if Rules.is_checkmate(self.board, self.is_red_turn):
            return not self.is_red_turn
        
        # Các trường hợp khác là hòa
        return None
    
    def get_game_result(self) -> float:
        """
        Lấy kết quả game từ góc nhìn của bên hiện tại
        
        Returns:
            1.0 nếu thắng, -1.0 nếu thua, 0.0 nếu hòa hoặc chưa kết thúc
        """
        winner = self.get_winner()
        if winner is None:
            return 0.0
        return 1.0 if winner == self.is_red_turn else -1.0
    
    def _count_position_repetitions(self) -> int:
        """Đếm số lần vị trí hiện tại lặp lại"""
        if not self.position_history:
            return 0
        
        current_position = self.position_history[-1]
        return self.position_history.count(current_position)
    
    def _moves_since_capture(self) -> int:
        """Đếm số nước đi kể từ lần ăn quân cuối"""
        count = 0
        for move in reversed(self.move_history):
            if move[4] is not None:  # captured piece
                break
            count += 1
        return count
    
    def copy(self):
        """Tạo bản sao của game state"""
        new_state = GameState.__new__(GameState)
        new_state.board = self.board.copy()
        new_state.is_red_turn = self.is_red_turn
        new_state.move_history = self.move_history.copy()
        new_state.position_history = self.position_history.copy()
        new_state.move_count = self.move_count
        return new_state
    
    def to_tensor(self) -> np.ndarray:
        """
        Chuyển game state thành tensor cho neural network
        
        Returns:
            np.ndarray shape (19, 10, 9)
        """
        # Get board representation from current player's perspective
        board_array = self.board.to_numpy(red_perspective=self.is_red_turn)
        
        # Channel 16: Current player (1 for red, 0 for black)
        board_array[16, :, :] = 1.0 if self.is_red_turn else 0.0
        
        # Channel 17: Normalized move count
        board_array[17, :, :] = min(self.move_count / 100.0, 1.0)
        
        # Channel 18: Position repetition count
        rep_count = self._count_position_repetitions()
        board_array[18, :, :] = min(rep_count / REPETITION_DRAW_COUNT, 1.0)
        
        return board_array
    
    def __str__(self):
        """Hiển thị trạng thái game"""
        result = []
        result.append(f"Move {self.move_count}: {'Red' if self.is_red_turn else 'Black'} to move")
        result.append(str(self.board))
        
        if self.is_terminal():
            winner = self.get_winner()
            if winner is None:
                result.append("Game Over: Draw")
            else:
                result.append(f"Game Over: {'Red' if winner else 'Black'} wins!")
        
        return "\n".join(result)
    
    def get_move_notation(self, from_row: int, from_col: int, 
                         to_row: int, to_col: int) -> str:
        """
        Chuyển nước đi sang ký hiệu (simplified notation)
        
        Returns:
            String như "e2-e4"
        """
        cols = 'abcdefghi'
        from_pos = f"{cols[from_col]}{from_row}"
        to_pos = f"{cols[to_col]}{to_row}"
        return f"{from_pos}-{to_pos}"
