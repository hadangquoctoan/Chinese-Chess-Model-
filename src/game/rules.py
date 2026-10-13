"""
Luật chơi cờ tướng và kiểm tra hợp lệ
"""
from typing import List, Optional, Tuple

from .board import Board
from .pieces import PieceType

from .action_encoding import (
    ACTION_DESTINATION_BUCKETS, ACTION_ENCODING_VERSION, ACTION_SPACE_SIZE,
    BOARD_COLS, BOARD_ROWS, Move, decode_geometry, encode, valid_coordinate,
)

RULESET_VERSION = 'xiangqi-training-v2'

class Rules:
    """Quản lý luật chơi cờ tướng"""
    
    @staticmethod
    def has_valid_generals(board: Board) -> bool:
        """A playable board has exactly one general of each color."""
        counts = [0, 0]
        for row in board.board:
            for piece in row:
                if piece.piece_type == PieceType.GENERAL:
                    counts[int(piece.is_red)] += 1
        return counts == [1, 1]

    @staticmethod
    def validate_position(board: Board) -> None:
        """Reject missing or duplicate generals instead of silently playing."""
        if not Rules.has_valid_generals(board):
            raise ValueError('Position must contain exactly one general of each color')

    @staticmethod
    def is_in_check(board: Board, is_red: bool) -> bool:
        """
        Kiểm tra xem tướng có đang bị chiếu không
        
        Args:
            board: Bàn cờ hiện tại
            is_red: Màu của bên cần kiểm tra
        
        Returns:
            True nếu đang bị chiếu
        """
        # Tìm vị trí tướng
        general_pos = board.find_general(is_red)
        if general_pos is None:
            return True  # A missing general cannot be treated as safe.
        
        general_row, general_col = general_pos
        
        # Kiểm tra tất cả quân địch có thể chiếu tướng không
        opponent_pieces = board.get_all_pieces(not is_red)
        
        for row, col, piece in opponent_pieces:
            possible_moves = piece.get_possible_moves(row, col, board)
            if (general_row, general_col) in possible_moves:
                return True
        
        # Kiểm tra luật "tướng đối mặt" (Flying General)
        opponent_general_pos = board.find_general(not is_red)
        if opponent_general_pos:
            og_row, og_col = opponent_general_pos
            # Cùng cột
            if general_col == og_col:
                # Kiểm tra có quân nào giữa 2 tướng không
                min_row = min(general_row, og_row)
                max_row = max(general_row, og_row)
                pieces_between = 0
                for r in range(min_row + 1, max_row):
                    if board.get_piece(r, general_col).piece_type != PieceType.EMPTY:
                        pieces_between += 1
                
                if pieces_between == 0:
                    return True  # Tướng đối mặt
        
        return False
    
    @staticmethod
    def is_checkmate(board: Board, is_red: bool) -> bool:
        """
        Kiểm tra chiếu hết (checkmate)
        
        Args:
            board: Bàn cờ hiện tại
            is_red: Màu của bên cần kiểm tra
        
        Returns:
            True nếu bị chiếu hết
        """
        # Phải đang bị chiếu
        if not Rules.is_in_check(board, is_red):
            return False
        
        # Kiểm tra xem có nước đi nào thoát chiếu không
        legal_moves = Rules.get_all_legal_moves(board, is_red)
        
        return len(legal_moves) == 0
    
    @staticmethod
    def is_stalemate(board: Board, is_red: bool) -> bool:
        """
        Detect stalemate: no legal moves without check (a loss in Xiangqi)
        
        Args:
            board: Bàn cờ hiện tại
            is_red: Màu của bên cần kiểm tra
        
        Returns:
            True if stalemated
        """
        # Không bị chiếu
        if Rules.is_in_check(board, is_red):
            return False
        
        # Không có nước đi hợp lệ
        legal_moves = Rules.get_all_legal_moves(board, is_red)
        
        return len(legal_moves) == 0
    
    @staticmethod
    def is_move_legal(board: Board, from_row: int, from_col: int, 
                     to_row: int, to_col: int, is_red: bool) -> bool:
        """
        Kiểm tra nước đi có hợp lệ không
        
        Args:
            board: Bàn cờ hiện tại
            from_row, from_col: Vị trí xuất phát
            to_row, to_col: Vị trí đích
            is_red: Màu của bên đi
        
        Returns:
            True nếu hợp lệ
        """
        if (not Rules.has_valid_generals(board)
                or not valid_coordinate(from_row, from_col)
                or not valid_coordinate(to_row, to_col)):
            return False
        piece = board.get_piece(from_row, from_col)
        
        # Kiểm tra có quân không
        if piece.piece_type == PieceType.EMPTY:
            return False
        
        # Kiểm tra quân đúng màu
        if piece.is_red != is_red:
            return False

        # Generals are checkmated, not captured. Treat their square as attacked
        # for check detection, but do not let a game state remove a general.
        target = board.get_piece(to_row, to_col)
        if target.piece_type == PieceType.GENERAL:
            return False
        
        # Kiểm tra nước đi có trong danh sách nước đi có thể
        possible_moves = piece.get_possible_moves(from_row, from_col, board)
        if (to_row, to_col) not in possible_moves:
            return False
        
        # Kiểm tra sau khi đi, tướng có bị chiếu không
        test_board = board.copy()
        test_board.move_piece(from_row, from_col, to_row, to_col)
        
        if Rules.is_in_check(test_board, is_red):
            return False
        
        return True
    
    @staticmethod
    def has_legal_move(board: Board, is_red: bool) -> bool:
        """Stop at the first legal move for terminal queries."""
        if not Rules.has_valid_generals(board):
            return False
        for row, col, piece in board.get_all_pieces(is_red):
            for target_row, target_col in piece.get_possible_moves(row, col, board):
                if Rules.is_move_legal(board, row, col, target_row, target_col, is_red):
                    return True
        return False

    @staticmethod
    def get_all_legal_moves(board: Board, is_red: bool) -> List[Move]:
        """
        Lấy tất cả các nước đi hợp lệ
        
        Args:
            board: Bàn cờ hiện tại
            is_red: Màu của bên đi
        
        Returns:
            List of (from_row, from_col, to_row, to_col)
        """
        if not Rules.has_valid_generals(board):
            return []
        legal_moves = []
        
        # Duyệt tất cả quân cờ của bên đi
        pieces = board.get_all_pieces(is_red)
        
        for from_row, from_col, piece in pieces:
            # Lấy tất cả nước đi có thể
            possible_moves = piece.get_possible_moves(from_row, from_col, board)
            
            # Kiểm tra từng nước đi
            for to_row, to_col in possible_moves:
                if Rules.is_move_legal(board, from_row, from_col, to_row, to_col, is_red):
                    legal_moves.append((from_row, from_col, to_row, to_col))
        
        return legal_moves
    
    @staticmethod
    def move_to_action_index(from_row: int, from_col: int, to_row: int,
                             to_col: int, board: Board) -> int:
        """Encode a move using the origin's piece and absolute geometry."""
        return encode(from_row, from_col, to_row, to_col, board)

    @staticmethod
    def action_index_to_move(action_index: int, board: Board, is_red: bool) -> Optional[Move]:
        """Decode exactly one slot, returning None unless it is board-legal."""
        move = decode_geometry(action_index, board, is_red)
        if move is not None and Rules.is_move_legal(board, *move, is_red):
            return move
        return None

    _is_valid_coordinate = staticmethod(valid_coordinate)
