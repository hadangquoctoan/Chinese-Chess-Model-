"""
Quản lý bàn cờ tướng
"""
from typing import List, Optional, Tuple

import numpy as np

from .pieces import Piece, PieceType, create_piece

class Board:
    """Đại diện cho bàn cờ tướng 10x9"""

    def __init__(self):
        """Khởi tạo bàn cờ với vị trí ban đầu"""
        self.rows = 10
        self.cols = 9
        self.board = [[None for _ in range(self.cols)] for _ in range(self.rows)]
        self._setup_initial_position()

    def _setup_initial_position(self):
        """Đặt quân cờ vào vị trí ban đầu"""
        # Quân đỏ (phía dưới, hàng 0-4)
        # Hàng 0: Xe, Mã, Tượng, Sĩ, Tướng, Sĩ, Tượng, Mã, Xe
        red_back_row = [
            PieceType.CHARIOT, PieceType.HORSE, PieceType.ELEPHANT,
            PieceType.ADVISOR, PieceType.GENERAL, PieceType.ADVISOR,
            PieceType.ELEPHANT, PieceType.HORSE, PieceType.CHARIOT
        ]
        for col, piece_type in enumerate(red_back_row):
            self.board[0][col] = create_piece(piece_type, is_red=True)

        # Hàng 2: Pháo
        self.board[2][1] = create_piece(PieceType.CANNON, is_red=True)
        self.board[2][7] = create_piece(PieceType.CANNON, is_red=True)

        # Hàng 3: Tốt
        for col in [0, 2, 4, 6, 8]:
            self.board[3][col] = create_piece(PieceType.SOLDIER, is_red=True)

        # Quân đen (phía trên, hàng 5-9)
        # Hàng 9: Xe, Mã, Tượng, Sĩ, Tướng, Sĩ, Tượng, Mã, Xe
        black_back_row = [
            PieceType.CHARIOT, PieceType.HORSE, PieceType.ELEPHANT,
            PieceType.ADVISOR, PieceType.GENERAL, PieceType.ADVISOR,
            PieceType.ELEPHANT, PieceType.HORSE, PieceType.CHARIOT
        ]
        for col, piece_type in enumerate(black_back_row):
            self.board[9][col] = create_piece(piece_type, is_red=False)

        # Hàng 7: Pháo
        self.board[7][1] = create_piece(PieceType.CANNON, is_red=False)
        self.board[7][7] = create_piece(PieceType.CANNON, is_red=False)

        # Hàng 6: Tốt
        for col in [0, 2, 4, 6, 8]:
            self.board[6][col] = create_piece(PieceType.SOLDIER, is_red=False)

        # Fill remaining empty squares
        for row in range(10):
            for col in range(9):
                if self.board[row][col] is None:
                    self.board[row][col] = create_piece(PieceType.EMPTY, is_red=True)

    def get_piece(self, row: int, col: int) -> Piece:
        """Lấy quân cờ tại vị trí"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            return self.board[row][col]
        return create_piece(PieceType.EMPTY, is_red=True)

    def set_piece(self, row: int, col: int, piece: Piece):
        """Đặt quân cờ tại vị trí"""
        if 0 <= row < self.rows and 0 <= col < self.cols:
            self.board[row][col] = piece

    def move_piece(self, from_row: int, from_col: int, to_row: int, to_col: int) -> Optional[Piece]:
        """
        Di chuyển quân cờ

        Returns:
            Quân bị ăn (nếu có)
        """
        piece = self.get_piece(from_row, from_col)
        captured = self.get_piece(to_row, to_col)

        self.set_piece(to_row, to_col, piece)
        self.set_piece(
            from_row,
            from_col,
            create_piece(PieceType.EMPTY, is_red=True),
        )

        return captured if captured.piece_type != PieceType.EMPTY else None

    def copy(self):
        """Tạo bản sao của bàn cờ"""
        new_board = Board.__new__(Board)
        new_board.rows = self.rows
        new_board.cols = self.cols
        new_board.board = [[self.board[r][c] for c in range(self.cols)]
                          for r in range(self.rows)]
        return new_board

    def find_general(self, is_red: bool) -> Optional[Tuple[int, int]]:
        """Tìm vị trí tướng"""
        for row in range(self.rows):
            for col in range(self.cols):
                piece = self.board[row][col]
                if piece.piece_type == PieceType.GENERAL and piece.is_red == is_red:
                    return (row, col)
        return None

    def get_all_pieces(self, is_red: bool) -> List[Tuple[int, int, Piece]]:
        """Lấy tất cả quân cờ của một bên"""
        pieces = []
        for row in range(self.rows):
            for col in range(self.cols):
                piece = self.board[row][col]
                if piece.piece_type != PieceType.EMPTY and piece.is_red == is_red:
                    pieces.append((row, col, piece))
        return pieces

    def __str__(self):
        """Hiển thị bàn cờ dạng text"""
        result = []
        result.append("  0 1 2 3 4 5 6 7 8")
        result.append(" ╔═════════════════╗")

        for row in range(9, -1, -1):
            line = f"{row}║"
            for col in range(9):
                piece = self.board[row][col]
                line += str(piece) + " "
            line = line.rstrip() + "║"
            result.append(line)

            # River line
            if row == 5:
                result.append(" ╠═════════════════╣")

        result.append(" ╚═════════════════╝")
        return "\n".join(result)

    def _encode_piece_to_channel(
        self,
        piece: Piece,
        row: int,
        red_perspective: bool
    ) -> Tuple[int, int]:
        """
        Xác định channel và row từ perspective của người chơi.

        Args:
            piece: Quân cờ cần encode
            row: Hàng hiện tại
            red_perspective: True nếu nhìn từ phía đỏ

        Returns:
            Tuple (channel, actual_row) để đặt vào tensor
        """
        # Determine perspective
        if red_perspective:
            actual_row = row
            is_own = piece.is_red
        else:
            actual_row = 9 - row
            is_own = not piece.is_red

        # Channels 0-6: Own pieces
        # Channels 7-13: Opponent pieces
        channel_offset = 0 if is_own else 7
        piece_channel = piece.piece_type.value + channel_offset

        return piece_channel, actual_row

    def _encode_pieces_layer(self, red_perspective: bool) -> np.ndarray:
        """
        Encode tất cả quân cờ vào 14 channels đầu tiên.

        Args:
            red_perspective: True nếu nhìn từ phía đỏ

        Returns:
            np.ndarray shape (14, 10, 9) - channels 0-13
        """
        pieces_array = np.zeros((14, 10, 9), dtype=np.float32)

        for row in range(10):
            for col in range(9):
                piece = self.board[row][col]

                if piece.piece_type == PieceType.EMPTY:
                    continue

                channel, actual_row = self._encode_piece_to_channel(
                    piece, row, red_perspective
                )
                pieces_array[channel, actual_row, col] = 1.0

        return pieces_array

    def _encode_palace_layer(self) -> np.ndarray:
        """
        Encode vùng cung (palace) vào channel 14.

        Returns:
            np.ndarray shape (10, 9) - channel 14
        """
        palace_array = np.zeros((10, 9), dtype=np.float32)

        for row in range(10):
            for col in range(3, 6):
                if row <= 2 or row >= 7:
                    palace_array[row, col] = 1.0

        return palace_array

    def _encode_river_layer(self) -> np.ndarray:
        """
        Encode sông (river) vào channel 15.

        Returns:
            np.ndarray shape (10, 9) - channel 15
        """
        river_array = np.zeros((10, 9), dtype=np.float32)

        for row in range(5, 10):
            river_array[row, :] = 1.0

        return river_array

    def to_numpy(self, red_perspective: bool = True) -> np.ndarray:
        """
        Chuyển bàn cờ sang numpy array cho neural network.

        Args:
            red_perspective: True nếu encode từ góc nhìn đỏ, False từ góc nhìn đen

        Returns:
            np.ndarray shape (19, 10, 9)
            - Channels 0-6: Quân của mình (own pieces)
            - Channels 7-13: Quân địch (opponent pieces)
            - Channel 14: Vùng cung (palace)
            - Channel 15: Sông (river)
            - Channel 16-18: Sẽ được StateEncoder set (turn, move count, repetition)
        """
        board_array = np.zeros((19, 10, 9), dtype=np.float32)

        # Encode pieces (channels 0-13)
        board_array[0:14] = self._encode_pieces_layer(red_perspective)

        # Encode palace (channel 14)
        board_array[14] = self._encode_palace_layer()

        # Encode river (channel 15)
        board_array[15] = self._encode_river_layer()

        # Channels 16-18 are populated by StateEncoder:
        # - Channel 16: Current player turn
        # - Channel 17: Move count normalized
        # - Channel 18: Repetition count

        return board_array
