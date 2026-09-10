"""Xiangqi piece types, implementations, and construction helpers.

Ghi chú tên các quân cờ (tướng):
- General (Tướng / Soái):
    + Đỏ (Red): 帥 (Soái)
    + Đen (Black): 將 (Tướng)
- Advisor (Sĩ):
    + Đỏ (Red): 仕 (Sĩ)
    + Đen (Black): 士 (Sĩ)
- Elephant (Tượng / Voi):
    + Đỏ (Red): 相 (Tượng)
    + Đen (Black): 象 (Tượng)
- Horse (Mã / Ngựa):
    + Đỏ (Red): 傌 (Mã)
    + Đen (Black): 馬 (Mã)
- Chariot (Xe):
    + Đỏ (Red): 俥 (Xe)
    + Đen (Black): 車 (Xe)
- Cannon (Pháo):
    + Đỏ (Red): 炮 (Pháo)
    + Đen (Black): 砲 (Pháo)
- Soldier (Binh / Tốt):
    + Đỏ (Red): 兵 (Binh)
    + Đen (Black): 卒 (Tốt)
"""
from .advisor import Advisor      # Sĩ (仕 / 士)
from .base import Piece
from .cannon import Cannon        # Pháo (炮 / 砲)
from .chariot import Chariot      # Xe (俥 / 車)
from .elephant import Elephant    # Tượng (相 / 象)
from .empty import EmptyPiece     # Ô trống
from .factory import create_piece
from .general import General      # Tướng / Soái (帥 / 將)
from .horse import Horse          # Mã (傌 / 馬)
from .soldier import Soldier      # Tốt / Binh (兵 / 卒)
from .types import PieceType, Position

__all__ = [
    'Advisor',
    'Cannon',
    'Chariot',
    'Elephant',
    'EmptyPiece',
    'General',
    'Horse',
    'Piece',
    'PieceType',
    'Position',
    'Soldier',
    'create_piece',
]
