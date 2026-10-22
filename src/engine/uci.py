"""Minimal synchronous UCI client for the bundled Pikafish binary."""

from __future__ import annotations

from pathlib import Path
from queue import Empty, Queue
import subprocess
import sys
import threading
import time

from .search_score import parse_uci_score

ENGINE_DIR = Path(__file__).resolve().parents[2] / "pikafish"
NNUE_NAME = "pikafish.nnue"
_WINDOWS_BINARY = "Pikafish-Windows-x86-64-universal.exe"
_MAC_BINARY = "Pikafish-MacOS-universal"
_LINUX_BINARY = "Pikafish-Linux-x86-64-universal"


def find_engine_binary() -> Path:
    """Locate the Pikafish executable for the current platform."""
    if sys.platform.startswith("win"):
        name = _WINDOWS_BINARY
    elif sys.platform == "darwin":
        name = _MAC_BINARY
    else:
        name = _LINUX_BINARY
    path = ENGINE_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"Pikafish executable not found: {path}")
    return path


class PikafishEngine:
    """One persistent UCI subprocess; callers must hold the game lock."""

    def __init__(self, path: Path | None = None, threads: int = 2,
                 hash_mb: int = 64) -> None:
        self.path = Path(path) if path else find_engine_binary()
        if not self.path.is_file():
            raise FileNotFoundError(f"Pikafish binary not found: {self.path}")
        self._queue: Queue[str | None] = Queue()
        kwargs: dict = {}
        if sys.platform == "win32":
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        self.process = subprocess.Popen(
            [str(self.path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, bufsize=1,
            encoding="utf-8", errors="replace", cwd=str(ENGINE_DIR), **kwargs)
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        try:
            self._reader.start()
            self._send("uci")
            self._wait_for("uciok", timeout=20.0)
            nnue = ENGINE_DIR / NNUE_NAME
            if nnue.is_file():
                self._send(f"setoption name EvalFile value {nnue.name}")
            self._send(f"setoption name Threads value {threads}")
            self._send(f"setoption name Hash value {hash_mb}")
            self._send("isready")
            self._wait_for("readyok", timeout=30.0)
        except Exception:
            self.close()
            raise

    def _read_loop(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._queue.put(line.rstrip("\r\n"))
        self._queue.put(None)

    def _send(self, command: str) -> None:
        assert self.process.stdin is not None
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _drain(self) -> None:
        while True:
            try:
                self._queue.get_nowait()
            except Empty:
                return

    def _wait_for(self, token: str, timeout: float, *, prefix: bool = False) -> str:
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"Pikafish did not answer '{token}' in time.")
            try:
                line = self._queue.get(timeout=remaining)
            except Empty:
                raise TimeoutError(f"Pikafish did not answer '{token}' in time.")
            if line is None:
                raise RuntimeError("Pikafish exited before answering.")
            if line.startswith(token) if prefix else token in line:
                return line

    def _setup_position(self, fen: str, moves: list[str] | None, timeout: float) -> None:
        self._drain()
        self._send("ucinewgame")
        position = f"position fen {fen}"
        if moves:
            position += " moves " + " ".join(moves)
        self._send(position)
        self._send("isready")
        self._wait_for("readyok", timeout=min(timeout, 15.0))

    @staticmethod
    def _parse_bestmove_token(line: str) -> str:
        parts = line.split()
        if len(parts) < 2 or parts[1] in ("(none)", "0000"):
            raise ValueError("Pikafish returned no move.")
        return parts[1].strip()

    def bestmove(self, fen: str, moves: list[str] | None = None,
                 depth: int = 12, timeout: float = 60.0) -> str:
        """Search one position and return the raw ``bestmove`` token."""
        self._setup_position(fen, moves, timeout)
        self._send(f"go depth {depth}")
        line = self._wait_for("bestmove", timeout=timeout, prefix=True)
        return self._parse_bestmove_token(line)

    def search_with_eval(self, fen: str, moves: list[str] | None = None,
                         depth: int = 12, timeout: float = 60.0) -> tuple[str, int]:
        """Search one position; return ``(bestmove_token, score_cp)``.

        Collects UCI ``info`` lines, then parses the last score with
        ``parse_uci_score``. Does not change ``bestmove()``.
        """
        self._setup_position(fen, moves, timeout)
        self._send(f"go depth {depth}")
        info_lines: list[str] = []
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Pikafish did not answer 'bestmove' in time.")
            try:
                line = self._queue.get(timeout=remaining)
            except Empty:
                raise TimeoutError("Pikafish did not answer 'bestmove' in time.")
            if line is None:
                raise RuntimeError("Pikafish exited before answering.")
            if line.startswith("bestmove"):
                return self._parse_bestmove_token(line), parse_uci_score(info_lines)
            info_lines.append(line)

    def close(self) -> None:
        if self.process.poll() is not None:
            return
        try:
            self._send("quit")
        except (BrokenPipeError, ValueError, OSError):
            pass
        try:
            self.process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            self.process.kill()

    def __enter__(self) -> PikafishEngine:
        return self

    def __exit__(self, *args: object) -> None:
        self.close()
