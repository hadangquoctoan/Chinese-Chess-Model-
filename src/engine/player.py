"""Pikafish opponent that speaks the same move contract as AIPlayer."""

from pathlib import Path

from ..game.game_state import GameState
from ..game.rules import Move, Rules
from .coordinates import move_to_uci, parse_uci_move
from .fen import STARTPOS_FEN, board_to_fen
from .uci import PikafishEngine, find_engine_binary

DEFAULT_DEPTH = 8
MAX_DEPTH = 30
DEFAULT_THREADS = 2
DEFAULT_HASH_MB = 64


class PikafishPlayer:
    """Ask Pikafish for one legal project-coordinate move."""

    def __init__(self, engine: PikafishEngine, depth: int = DEFAULT_DEPTH) -> None:
        if type(depth) is not int or not 1 <= depth <= MAX_DEPTH:
            raise ValueError(f"Search depth must be an integer from 1 to {MAX_DEPTH}.")
        self.name = "Pikafish"
        self.device = "cpu"
        self.max_game_plies = 200
        self.no_capture_limit = 100
        self.depth = depth
        self._engine = engine

    def choose_move(self, state: GameState) -> Move | None:
        """Search the current position and return a Rules-legal project move."""
        fen = board_to_fen(state.board, state.is_red_turn)
        token = self._engine.bestmove(fen, [], depth=self.depth)
        move = parse_uci_move(token)
        if not Rules.is_move_legal(state.board, *move, state.is_red_turn):
            raise ValueError(
                f"Pikafish move {token} -> {move} is not legal in the project rules."
            )
        return move

    def close(self) -> None:
        self._engine.close()


def load_pikafish_player(depth: int = DEFAULT_DEPTH,
                         path: Path | None = None,
                         threads: int = DEFAULT_THREADS,
                         hash_mb: int = DEFAULT_HASH_MB) -> PikafishPlayer:
    """Start a Pikafish subprocess and wrap it as a web-game opponent."""
    engine = PikafishEngine(
        path=path or find_engine_binary(),
        threads=threads,
        hash_mb=hash_mb,
    )
    try:
        return PikafishPlayer(engine, depth=depth)
    except Exception:
        engine.close()
        raise


def history_uci_moves(state: GameState) -> list[str]:
    """Convert recorded project moves to UCI tokens for a startpos search."""
    return [move_to_uci(record[:4]) for record in state.move_history]


def startpos_fen() -> str:
    return STARTPOS_FEN
