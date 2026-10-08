"""Run a local, human-versus-human Xiangqi web game."""
from __future__ import annotations

import argparse
import json
import sys
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from urllib.parse import urlparse


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
WEB_ROOT = REPOSITORY_ROOT / "web"
MAX_REQUEST_BYTES = 4_096
STATIC_FILES = {
    "index.html": "text/html; charset=utf-8",
    "styles.css": "text/css; charset=utf-8",
    "app.js": "application/javascript; charset=utf-8",
    "assets/xiangqi-board.svg": "image/svg+xml",
    "receptive_field.html": "text/html; charset=utf-8",
    "receptive_field.css": "text/css; charset=utf-8",
    "receptive_field.js": "application/javascript; charset=utf-8",
}

sys.path.insert(0, str(REPOSITORY_ROOT))

from src.game.game_state import GameState
from src.game.pieces import PieceType


Move = tuple[int, int, int, int]


class LocalGame:
    """Thread-safe game service backed by the canonical GameState engine."""

    def __init__(self) -> None:
        self._state = GameState()
        self._lock = Lock()

    def new_game(self) -> dict[str, Any]:
        """Replace the active game with the standard initial position."""
        with self._lock:
            self._state = GameState()
            return self._response(True)

    def move(self, move: Move) -> dict[str, Any]:
        """Apply a player move only when the engine accepts it as legal."""
        with self._lock:
            if self._state.is_terminal():
                return self._response(False, "The game has already ended.")

            if not self._state.make_move(*move):
                return self._response(False, "That move is not legal.")

            return self._response(True)

    def undo(self) -> dict[str, Any]:
        """Undo the most recent move, when one exists."""
        with self._lock:
            if not self._state.undo_move():
                return self._response(False, "There is no move to undo.")

            return self._response(True)

    def snapshot(self) -> dict[str, Any]:
        """Return the current UI state without mutating the game."""
        with self._lock:
            return self._serialize_state()

    def _response(self, ok: bool, error: Optional[str] = None) -> dict[str, Any]:
        response = {"ok": ok, "state": self._serialize_state()}
        if error is not None:
            response["error"] = error
        return response

    def _serialize_state(self) -> dict[str, Any]:
        board = []
        for row in range(self._state.board.rows):
            board_row = []
            for col in range(self._state.board.cols):
                piece = self._state.board.get_piece(row, col)
                if piece.piece_type == PieceType.EMPTY:
                    board_row.append(None)
                else:
                    board_row.append(
                        {
                            "type": piece.piece_type.name,
                            "isRed": piece.is_red,
                            "symbol": str(piece),
                        }
                    )
            board.append(board_row)

        is_terminal = self._state.is_terminal()
        winner = self._state.get_winner() if is_terminal else None
        history = self._state.move_history
        last_move = list(history[-1][:4]) if history else None

        return {
            "board": board,
            "isRedTurn": self._state.is_red_turn,
            "legalMoves": [list(move) for move in self._state.get_legal_moves()],
            "lastMove": last_move,
            "moveCount": self._state.move_count,
            "isTerminal": is_terminal,
            "winner": "red" if winner is True else "black" if winner is False else None,
            "terminationReason": (
                self._state.get_termination_reason() if is_terminal else None
            ),
        }


class GameRequestHandler(BaseHTTPRequestHandler):
    """Serve the local web client and its JSON game API."""

    server: "GameWebServer"

    def do_GET(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/state":
            self._write_json(HTTPStatus.OK, self.server.game.snapshot())
            return

        filename = "index.html" if path in {"/", "/index.html"} else path.lstrip("/")
        content_type = STATIC_FILES.get(filename)
        if content_type is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self._write_static_file(filename, content_type)

    def do_POST(self) -> None:  # noqa: N802
        path = urlparse(self.path).path
        if path == "/api/new-game":
            self._write_json(HTTPStatus.OK, self.server.game.new_game())
            return

        if path == "/api/undo":
            self._write_game_response(self.server.game.undo())
            return

        if path != "/api/move":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        payload = self._read_json_body()
        move = self._parse_move(payload)
        if move is None:
            self._write_json(
                HTTPStatus.BAD_REQUEST,
                {"ok": False, "error": "Move coordinates must be board integers."},
            )
            return

        self._write_game_response(self.server.game.move(move))

    def _write_game_response(self, response: dict[str, Any]) -> None:
        status = HTTPStatus.OK if response["ok"] else HTTPStatus.BAD_REQUEST
        self._write_json(status, response)

    def _read_json_body(self) -> Optional[dict[str, Any]]:
        try:
            content_length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            return None

        if not 0 < content_length <= MAX_REQUEST_BYTES:
            return None

        try:
            payload = json.loads(self.rfile.read(content_length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return None

        return payload if isinstance(payload, dict) else None

    @staticmethod
    def _parse_move(payload: Optional[dict[str, Any]]) -> Optional[Move]:
        if payload is None:
            return None

        coordinate_names = ("fromRow", "fromCol", "toRow", "toCol")
        coordinates = tuple(payload.get(name) for name in coordinate_names)
        if not all(type(coordinate) is int for coordinate in coordinates):
            return None

        from_row, from_col, to_row, to_col = coordinates
        if not (0 <= from_row < 10 and 0 <= to_row < 10):
            return None
        if not (0 <= from_col < 9 and 0 <= to_col < 9):
            return None

        return from_row, from_col, to_row, to_col

    def _write_static_file(self, filename: str, content_type: str) -> None:
        try:
            content = (WEB_ROOT / filename).read_bytes()
        except FileNotFoundError:
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _write_json(self, status: HTTPStatus, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format: str, *args: object) -> None:
        """Keep the local server output focused on its startup URL."""


class GameWebServer(ThreadingHTTPServer):
    """HTTP server that owns one local, human-versus-human game."""

    def __init__(self, server_address: tuple[str, int]) -> None:
        super().__init__(server_address, GameRequestHandler)
        self.game = LocalGame()


def parse_args() -> argparse.Namespace:
    """Parse local server configuration."""
    parser = argparse.ArgumentParser(description="Run the local Xiangqi web game.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    """Start the local web server until interrupted."""
    args = parse_args()
    server = GameWebServer((args.host, args.port))
    print(f"Xiangqi is available at http://{args.host}:{server.server_port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
