"""
Trạng thái ván cờ - quản lý toàn bộ state của game
"""
from typing import List, Tuple, Optional, Any

import numpy as np

from ..board import Board
from ..rules import Rules
from ..pieces import PieceType, create_piece
from .move_history import MoveHistory
from .position_tracker import PositionTracker
from .game_result import GameResult
from .state_encoder import StateEncoder


class GameState:
    """
    Quản lý trạng thái hoàn chỉnh của ván cờ
    
    Orchestrates các components:
    - Board: Trạng thái bàn cờ
    - MoveHistory: Lịch sử nước đi
    - PositionTracker: Theo dõi lặp lại vị trí
    - GameResult: Logic kết thúc game
    - StateEncoder: Encode cho neural network
    """
    
    def __init__(self):
        """Khởi tạo trạng thái game mới"""
        self.board = Board()
        self.is_red_turn = True  # Đỏ đi trước
        
        # Delegate responsibilities to specialized components
        self._move_history = MoveHistory()
        self._position_tracker = PositionTracker()
        
        # Save initial position
        self._position_tracker.save_position(self.board, self.is_red_turn)
    
    def make_move(
        self,
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int
    ) -> bool:
        """
        Thực hiện nước đi
        
        Args:
            from_row: Hàng xuất phát
            from_col: Cột xuất phát
            to_row: Hàng đích
            to_col: Cột đích
            
        Returns:
            True nếu nước đi hợp lệ và đã thực hiện
        """
        # Kiểm tra tính hợp lệ
        if not Rules.is_move_legal(
            self.board, from_row, from_col, to_row, to_col, self.is_red_turn
        ):
            return False
        
        # Thực hiện di chuyển
        captured = self.board.move_piece(from_row, from_col, to_row, to_col)
        
        # Lưu vào lịch sử
        self._move_history.add_move(from_row, from_col, to_row, to_col, captured)
        
        # Chuyển lượt
        self.is_red_turn = not self.is_red_turn
        
        # Lưu vị trí mới
        self._position_tracker.save_position(self.board, self.is_red_turn)
        
        return True
    
    def undo_move(self) -> bool:
        """
        Hoàn tác nước đi cuối cùng
        
        Returns:
            True nếu có nước đi để hoàn tác
        """
        # Lấy nước đi cuối
        last_move = self._move_history.pop_last_move()
        if last_move is None:
            return False
        
        # Di chuyển quân về vị trí cũ
        piece = self.board.get_piece(last_move.to_row, last_move.to_col)
        self.board.set_piece(last_move.from_row, last_move.from_col, piece)
        
        # Khôi phục quân bị ăn
        if last_move.captured_piece:
            self.board.set_piece(
                last_move.to_row,
                last_move.to_col,
                last_move.captured_piece
            )
        else:
            # Đặt ô trống
            self.board.set_piece(
                last_move.to_row,
                last_move.to_col,
                create_piece(PieceType.EMPTY, is_red=True)
            )
        
        # Khôi phục trạng thái
        self._position_tracker.remove_last_position()
        self.is_red_turn = not self.is_red_turn
        
        return True
    
    def get_legal_moves(self) -> List[Tuple[int, int, int, int]]:
        """
        Lấy tất cả nước đi hợp lệ cho lượt hiện tại
        
        Returns:
            List of (from_row, from_col, to_row, to_col)
        """
        return Rules.get_all_legal_moves(self.board, self.is_red_turn)
    
    def is_terminal(self) -> bool:
        """
        Kiểm tra game đã kết thúc chưa
        
        Returns:
            True nếu game đã kết thúc (checkmate, stalemate, draw)
        """
        return GameResult.is_terminal(
            self.board,
            self.is_red_turn,
            self._move_history,
            self._position_tracker
        )
    
    def get_winner(self) -> Optional[bool]:
        """
        Lấy người thắng
        
        Returns:
            True nếu đỏ thắng
            False nếu đen thắng
            None nếu hòa hoặc chưa kết thúc
        """
        return GameResult.get_winner(
            self.board,
            self.is_red_turn,
            self._move_history,
            self._position_tracker
        )
    
    def get_game_result(self) -> float:
        """
        Lấy kết quả game từ góc nhìn của bên hiện tại
        
        Returns:
            1.0 nếu bên hiện tại thắng
            -1.0 nếu bên hiện tại thua
            0.0 nếu hòa hoặc chưa kết thúc
        """
        return GameResult.get_result_value(
            self.board,
            self.is_red_turn,
            self._move_history,
            self._position_tracker
        )
    
    def get_termination_reason(self) -> str:
        """
        Lấy lý do game kết thúc
        
        Returns:
            String mô tả lý do (hoặc "Game in progress")
        """
        return GameResult.get_termination_reason(
            self.board,
            self.is_red_turn,
            self._move_history,
            self._position_tracker
        )
    
    def to_tensor(self) -> np.ndarray:
        """
        Chuyển game state thành tensor cho neural network
        
        Returns:
            np.ndarray shape (19, 10, 9)
        """
        return StateEncoder.to_tensor(
            self.board,
            self.is_red_turn,
            self._move_history,
            self._position_tracker
        )
    
    def copy(self) -> 'GameState':
        """
        Tạo bản sao của game state
        
        Returns:
            GameState mới độc lập
        """
        new_state = GameState.__new__(GameState)
        new_state.board = self.board.copy()
        new_state.is_red_turn = self.is_red_turn
        new_state._move_history = self._move_history.copy()
        new_state._position_tracker = self._position_tracker.copy()
        return new_state
    
    # Properties for backward compatibility and external access
    @property
    def move_count(self) -> int:
        """Lấy số nước đi đã thực hiện"""
        return self._move_history.get_move_count()
    
    @property
    def move_history(self) -> List[Tuple[int, int, int, int, Optional[Any]]]:
        """
        Lấy lịch sử nước đi dạng list of tuples (backward compatibility)
        
        Returns:
            List of (from_row, from_col, to_row, to_col, captured_piece)
        """
        return [move.to_tuple() for move in self._move_history]

    @property
    def position_history(self) -> List[Tuple]:
        """Return recorded positions in the legacy list format."""
        return self._position_tracker.get_positions()
    
    def __str__(self) -> str:
        """Hiển thị trạng thái game"""
        result = []
        
        # Header
        move_num = self.move_count
        turn = 'Red' if self.is_red_turn else 'Black'
        result.append(f"Move {move_num}: {turn} to move")
        
        # Board
        result.append(str(self.board))
        
        # Game result nếu đã kết thúc
        if self.is_terminal():
            result.append(self.get_termination_reason())
        
        return "\n".join(result)
    
    def get_move_notation(
        self,
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int
    ) -> str:
        """
        Chuyển nước đi sang ký hiệu đơn giản
        
        Args:
            from_row, from_col: Vị trí xuất phát
            to_row, to_col: Vị trí đích
            
        Returns:
            String như "e2-e4"
        """
        cols = 'abcdefghi'
        from_pos = f"{cols[from_col]}{from_row}"
        to_pos = f"{cols[to_col]}{to_row}"
        return f"{from_pos}-{to_pos}"
