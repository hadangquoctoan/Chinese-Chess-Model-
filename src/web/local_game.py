"""Thread-safe local games for human and optional neural-network players."""
from threading import Lock
from typing import Any

from ..game.game_state import GameState
from ..game.pieces import PieceType
from ..game.rules import Move
from .ai_player import DEFAULT_SIMULATIONS, MAX_SIMULATIONS, AIPlayer, ModelRepository


class LocalGame:
    """Serialize game mutations and AI inference for one shared local board."""

    def __init__(self, models: ModelRepository | None = None) -> None:
        self.models = models if models is not None else ModelRepository()
        self._state = GameState()
        self._lock = Lock()
        self._player: AIPlayer | None = None
        self._mode = "human"
        self._human_is_red = True
        self._simulations = DEFAULT_SIMULATIONS

    def new_game(self, mode: str = "human", model: str | None = None,
                 human_color: str = "red",
                 simulations: int = DEFAULT_SIMULATIONS) -> dict[str, Any]:
        """Start a game only after the requested opponent has loaded successfully."""
        with self._lock:
            if not isinstance(mode, str) or not isinstance(human_color, str):
                return self._response(False, "Invalid game mode or human color.")
            mode = mode.strip().lower()
            human_color = human_color.strip().lower() or "red"
            if mode not in ("human", "ai", "pikafish"):
                return self._response(False, f"Invalid game mode: {mode!r}.")
            if human_color not in ("red", "black"):
                return self._response(False, f"Invalid human color: {human_color!r}.")
            if isinstance(simulations, float) and simulations.is_integer():
                simulations = int(simulations)
            if type(simulations) is not int:
                return self._response(False, "Simulations/depth must be an integer.")
            if mode == "pikafish":
                try:
                    from ..engine.player import MAX_DEPTH, load_pikafish_player
                except ImportError as error:
                    return self._response(False, f"Could not load Pikafish: {error}")
                if not 1 <= simulations <= MAX_DEPTH:
                    return self._response(
                        False,
                        f"Pikafish depth must be an integer from 1 to {MAX_DEPTH}.")
            elif not 1 <= simulations <= MAX_SIMULATIONS:
                return self._response(False, f"MCTS simulations must be an integer from 1 to {MAX_SIMULATIONS}.")
            player = None
            try:
                if mode == "ai":
                    player = self.models.load_player(model, simulations)
                elif mode == "pikafish":
                    player = load_pikafish_player(depth=simulations)
                state = (GameState(max_plies=player.max_game_plies,
                                   no_capture_limit=player.no_capture_limit)
                         if player else GameState())
            except ImportError:
                return self._response(False, "AI mode requires PyTorch. Install the project requirements in this Python environment.")
            except (ValueError, RuntimeError, OSError, TypeError, KeyError, EOFError, TimeoutError) as error:
                return self._response(False, f"Could not load AI opponent: {error}")
            old_player = self._player
            self._state = state
            self._player = player
            self._mode = mode if player else "human"
            self._human_is_red = human_color == "red"
            self._simulations = simulations
            if old_player is not None and old_player is not player:
                close = getattr(old_player, "close", None)
                if callable(close):
                    try:
                        close()
                    except (ValueError, RuntimeError, OSError):
                        pass
            return self._response(True)

    def move(self, move: Move) -> dict[str, Any]:
        """Apply a legal human move; never accept browser moves for the AI side."""
        with self._lock:
            if self._state.is_terminal():
                return self._response(False, "The game has already ended.")
            if self._is_ai_turn():
                return self._response(False, "It is the AI's turn.")
            if not self._state.make_move(*move):
                return self._response(False, "That move is not legal.")
            return self._response(True)

    def ai_move(self) -> dict[str, Any]:
        """Search and apply one AI reply, leaving the position intact on failure."""
        with self._lock:
            player = self._player
            if player is None or self._state.is_terminal() or not self._is_ai_turn():
                return self._response(False, "There is no pending AI move.")
            try:
                move = player.choose_move(self._state.copy())
                if move is None or not self._state.make_move(*move):
                    raise ValueError("MCTS did not return a legal move.")
            except (RuntimeError, ValueError, TypeError, TimeoutError, OSError) as error:
                return self._response(False, f"AI search failed: {error}. Retry the AI move or undo.")
            return self._response(True)

    def undo(self) -> dict[str, Any]:
        """Undo a human turn and its AI reply, keeping a black player's opening."""
        with self._lock:
            if not self._can_undo():
                return self._response(False, "There is no human move to undo.")
            self._state.undo_move()
            if self._player and self._state.is_red_turn != self._human_is_red:
                self._state.undo_move()
            return self._response(True)

    def snapshot(self) -> dict[str, Any]:
        """Return the current UI state without mutating the game."""
        with self._lock:
            return self._serialize_state()

    def _is_ai_turn(self) -> bool:
        return self._player is not None and self._state.is_red_turn != self._human_is_red

    def _can_undo(self) -> bool:
        opening_plies = int(self._player is not None and not self._human_is_red)
        return self._state.move_count > opening_plies

    def _response(self, ok: bool, error: str | None = None) -> dict[str, Any]:
        response = {"ok": ok, "state": self._serialize_state()}
        if error is not None:
            response["error"] = error
        return response

    def _serialize_board(self) -> list[list[dict[str, Any] | None]]:
        board = []
        for row in range(self._state.board.rows):
            board_row = []
            for col in range(self._state.board.cols):
                piece = self._state.board.get_piece(row, col)
                board_row.append(None if piece.piece_type == PieceType.EMPTY else {
                    "type": piece.piece_type.name,
                    "isRed": piece.is_red,
                    "symbol": str(piece),
                })
            board.append(board_row)
        return board

    def _serialize_state(self) -> dict[str, Any]:
        is_terminal = self._state.is_terminal()
        winner = self._state.get_winner() if is_terminal else None
        history = self._state.move_history
        return {
            "board": self._serialize_board(),
            "isRedTurn": self._state.is_red_turn,
            "legalMoves": [list(move) for move in self._state.get_legal_moves()],
            "lastMove": list(history[-1][:4]) if history else None,
            "moveCount": self._state.move_count,
            "isTerminal": is_terminal,
            "winner": "red" if winner is True else "black" if winner is False else None,
            "terminationReason": self._state.get_termination_reason() if is_terminal else None,
            "mode": self._mode,
            "humanColor": "red" if self._human_is_red else "black",
            "model": self._player.name if self._player else None,
            "device": self._player.device if self._player else None,
            "simulations": self._simulations,
            "isAiTurn": self._is_ai_turn() and not is_terminal,
            "canUndo": self._can_undo(),
        }
