"""
Trạng thái ván cờ - quản lý toàn bộ state của game
"""
from typing import List, Tuple, Optional, Any

import numpy as np

from ..board import Board
from ..rules import Move, Rules
from ..pieces import PieceType, create_piece
from .move_history import MoveHistory
from .position_tracker import PositionTracker
from .game_result import GameResult
from .state_encoder import StateEncoder
from .repetition import assign_piece_ids, chase_targets


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
    
    def __init__(self, max_plies: int = 200, no_capture_limit: int = 100):
        """Khởi tạo trạng thái game mới"""
        if any(type(value) is not int or value <= 0
               for value in (max_plies, no_capture_limit)):
            raise ValueError('Game limits must be positive integers')
        self.max_plies = max_plies
        self.no_capture_limit = no_capture_limit
        self.board = Board()
        self.is_red_turn = True  # Đỏ đi trước
        
        # Delegate responsibilities to specialized components
        self._move_history = MoveHistory()
        self._position_tracker = PositionTracker()
        
        assign_piece_ids(self.board, reset=True)
        # Save initial position
        self._position_tracker.save_position(self.board, self.is_red_turn)
    
    def reset_history(self) -> None:
        """Start a fresh game from the edited board and current turn.

        Call after manual board setup; this clears undo and repetition data.
        Only general counts are validated, allowing focused test positions.
        """
        Rules.validate_position(self.board)
        assign_piece_ids(self.board, reset=True)
        self._move_history.clear()
        self._position_tracker.clear()
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
        if self.is_terminal():
            return False
        # Kiểm tra tính hợp lệ
        if not Rules.is_move_legal(
            self.board, from_row, from_col, to_row, to_col, self.is_red_turn
        ):
            return False
        return self.make_move_trusted(from_row, from_col, to_row, to_col)

    def make_move_trusted(
        self, from_row: int, from_col: int, to_row: int, to_col: int,
    ) -> bool:
        """Apply an already validated move during search, retaining all rule data.

        The caller must have checked that this exact position is non-terminal
        and the move passed Rules.is_move_legal, as search legal moves do.
        """
        assign_piece_ids(self.board)
        prior_threats = chase_targets(self.board, from_row, from_col)
        # Thực hiện di chuyển
        captured = self.board.move_piece(from_row, from_col, to_row, to_col)
        
        # Lưu vào lịch sử
        self._move_history.add_move(from_row, from_col, to_row, to_col, captured)
        record = self._move_history.get_last_move()
        record.mover_is_red = self.is_red_turn
        record.gave_check = Rules.is_in_check(self.board, not self.is_red_turn)
        record.chased_piece_ids = chase_targets(self.board, to_row, to_col) - prior_threats
        
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
        if self.is_terminal():
            return []
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
            self._position_tracker,
            self.max_plies,
            self.no_capture_limit
        )
    
    def get_search_info(self) -> Tuple[List[Move], Optional[float]]:
        """Return legal moves and a terminal value, or None while in progress.

        Search nodes can cache this snapshot while their state stays fixed.
        Ordinary mutable game states deliberately keep no persistent cache.
        """
        Rules.validate_position(self.board)
        legal_moves = Rules.get_all_legal_moves(self.board, self.is_red_turn)
        terminal, winner, _ = GameResult.outcome(
            self.board, self.is_red_turn, self._move_history,
            self._position_tracker, self.max_plies, self.no_capture_limit,
            legal_moves=legal_moves,
        )
        if not terminal:
            return legal_moves, None
        value = 0.0 if winner is None else (1.0 if winner == self.is_red_turn else -1.0)
        return [], value

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
            self._position_tracker,
            self.max_plies,
            self.no_capture_limit
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
            self._position_tracker,
            self.max_plies,
            self.no_capture_limit
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
            self._position_tracker,
            self.max_plies,
            self.no_capture_limit
        )
    
    def to_tensor(self) -> np.ndarray:
        """
        Chuyển game state thành tensor cho neural network
        
        Returns:
            np.ndarray shape (19, 10, 9)
        """
        return StateEncoder.to_tensor(
            self.board, self.is_red_turn, self._move_history,
            self._position_tracker, self.no_capture_limit, self.max_plies
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
        new_state.max_plies = self.max_plies
        new_state.no_capture_limit = self.no_capture_limit
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
    
    def get_move_notation(self, from_row: int, from_col: int,
                          to_row: int, to_col: int) -> str:
        """Return simple absolute coordinate notation, e.g. e2-e4."""
        cols = 'abcdefghi'
        from_pos = f"{cols[from_col]}{from_row}"
        to_pos = f"{cols[to_col]}{to_row}"
        return f"{from_pos}-{to_pos}"
