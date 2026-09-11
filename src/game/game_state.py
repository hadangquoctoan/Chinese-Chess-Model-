"""Backward-compatible facade for the refactored game-state package."""

from .gamestate.game_result import NO_CAPTURE_DRAW_PLY_LIMIT
from .gamestate.game_state import GameState
from .gamestate.position_tracker import REPETITION_DRAW_COUNT

__all__ = ["GameState", "REPETITION_DRAW_COUNT", "NO_CAPTURE_DRAW_PLY_LIMIT"]
