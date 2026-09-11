"""
Quản lý lịch sử nước đi và chức năng undo
"""
from typing import List, Tuple, Optional

from ..pieces import Piece


class MoveRecord:
    """Lưu trữ một nước đi đầy đủ"""
    
    def __init__(
        self,
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int,
        captured_piece: Optional[Piece] = None
    ):
        self.from_row = from_row
        self.from_col = from_col
        self.to_row = to_row
        self.to_col = to_col
        self.captured_piece = captured_piece
    
    def to_tuple(self) -> Tuple[int, int, int, int, Optional[Piece]]:
        """Convert to tuple format for backward compatibility"""
        return (
            self.from_row,
            self.from_col,
            self.to_row,
            self.to_col,
            self.captured_piece
        )
    
    def __repr__(self) -> str:
        return f"Move({self.from_row},{self.from_col}→{self.to_row},{self.to_col})"


class MoveHistory:
    """Quản lý lịch sử các nước đi"""
    
    def __init__(self):
        self._moves: List[MoveRecord] = []
    
    def add_move(
        self,
        from_row: int,
        from_col: int,
        to_row: int,
        to_col: int,
        captured_piece: Optional[Piece]
    ) -> None:
        """Thêm nước đi vào lịch sử"""
        move = MoveRecord(from_row, from_col, to_row, to_col, captured_piece)
        self._moves.append(move)
    
    def pop_last_move(self) -> Optional[MoveRecord]:
        """
        Lấy và xóa nước đi cuối cùng
        
        Returns:
            MoveRecord hoặc None nếu không có lịch sử
        """
        if not self._moves:
            return None
        return self._moves.pop()
    
    def get_last_move(self) -> Optional[MoveRecord]:
        """Lấy nước đi cuối cùng mà không xóa"""
        if not self._moves:
            return None
        return self._moves[-1]
    
    def count_moves_since_capture(self) -> int:
        """
        Đếm số nước đi kể từ lần ăn quân cuối cùng
        
        Returns:
            Số nước đi không có ăn quân
        """
        count = 0
        for move in reversed(self._moves):
            if move.captured_piece is not None:
                break
            count += 1
        return count
    
    def is_empty(self) -> bool:
        """Kiểm tra có lịch sử không"""
        return len(self._moves) == 0
    
    def get_move_count(self) -> int:
        """Lấy tổng số nước đi"""
        return len(self._moves)
    
    def copy(self) -> 'MoveHistory':
        """Tạo bản sao của lịch sử"""
        new_history = MoveHistory()
        new_history._moves = self._moves.copy()
        return new_history
    
    def clear(self) -> None:
        """Xóa toàn bộ lịch sử"""
        self._moves.clear()
    
    def __len__(self) -> int:
        return len(self._moves)
    
    def __iter__(self):
        return iter(self._moves)
    
    def __repr__(self) -> str:
        return f"MoveHistory({len(self._moves)} moves)"
