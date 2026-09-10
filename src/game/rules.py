"""
Luật chơi cờ tướng và kiểm tra hợp lệ
"""
from typing import List, Optional, Tuple

from .board import Board
from .pieces import PieceType

ACTION_SPACE_SIZE = 1800
ACTION_DESTINATION_BUCKETS = 20
BOARD_ROWS = 10
BOARD_COLS = 9
Move = Tuple[int, int, int, int]

class Rules:
    """Quản lý luật chơi cờ tướng"""
    
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
            return False  # Không có tướng (trường hợp test)
        
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
        Kiểm tra hòa (stalemate) - không bị chiếu nhưng không có nước đi hợp lệ
        
        Args:
            board: Bàn cờ hiện tại
            is_red: Màu của bên cần kiểm tra
        
        Returns:
            True nếu hòa
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
    def get_all_legal_moves(board: Board, is_red: bool) -> List[Move]:
        """
        Lấy tất cả các nước đi hợp lệ
        
        Args:
            board: Bàn cờ hiện tại
            is_red: Màu của bên đi
        
        Returns:
            List of (from_row, from_col, to_row, to_col)
        """
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
    def move_to_action_index(from_row: int, from_col: int, to_row: int, to_col: int) -> int:
        """
        Chuyển nước đi thành action index cho neural network
        
        Returns:
            Action index (0-1799)
        """
        if not Rules._is_valid_coordinate(from_row, from_col):
            raise ValueError("Move origin is outside the board")
        if not Rules._is_valid_coordinate(to_row, to_col):
            raise ValueError("Move destination is outside the board")

        # The current policy contract reserves 20 slots for every origin square.
        # It is shared with MCTS; do not change it without migrating checkpoints.
        from_pos = from_row * BOARD_COLS + from_col
        to_pos = to_row * BOARD_COLS + to_col
        return from_pos * ACTION_DESTINATION_BUCKETS + (to_pos % ACTION_DESTINATION_BUCKETS)
    
    @staticmethod
    def action_index_to_move(action_index: int, board: Board, is_red: bool) -> Optional[Move]:
        """
        Chuyển action index thành nước đi
        
        Returns:
            (from_row, from_col, to_row, to_col) or None if invalid
        """
        if not isinstance(action_index, int) or not 0 <= action_index < ACTION_SPACE_SIZE:
            return None

        from_pos = action_index // ACTION_DESTINATION_BUCKETS
        from_row = from_pos // BOARD_COLS
        from_col = from_pos % BOARD_COLS
        
        # Get possible moves for this piece
        piece = board.get_piece(from_row, from_col)
        if piece.piece_type == PieceType.EMPTY or piece.is_red != is_red:
            return None
        
        legal_moves = Rules.get_all_legal_moves(board, is_red)
        for move in legal_moves:
            if move[:2] != (from_row, from_col):
                continue
            if Rules.move_to_action_index(*move) == action_index:
                return move

        return None

    @staticmethod
    def _is_valid_coordinate(row: int, col: int) -> bool:
        """Return whether a board coordinate is in range."""
        return 0 <= row < BOARD_ROWS and 0 <= col < BOARD_COLS
