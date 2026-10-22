"""Discover local model artifacts and lazily load an MCTS opponent."""
from pathlib import Path
from typing import Any

from ..game.game_state import GameState
from ..game.rules import Move


MODELS_ROOT = Path(__file__).resolve().parents[2] / "models"
MODEL_SUFFIXES = {".pt", ".pth"}
DEFAULT_SIMULATIONS = 16
MAX_SIMULATIONS = 400


class AIPlayer:
    """Run deterministic MCTS with a versioned, inference-only model artifact."""

    def __init__(self, name: str, model: Any, search: Any) -> None:
        self.name = name
        self.device = str(model.get_device())
        self.max_game_plies = model.config.max_game_plies
        self.no_capture_limit = model.config.no_capture_limit
        self._search = search

    def choose_move(self, state: GameState) -> Move | None:
        """Choose a move without changing the supplied game state."""
        move, _ = self._search.search(state)
        return move


class ModelRepository:
    """Restrict model selection to weight artifacts inside the models directory."""

    def __init__(self, root: Path = MODELS_ROOT) -> None:
        self.root = root.resolve()

    def list_models(self) -> list[str]:
        """List relative filenames without importing PyTorch or loading weights."""
        if not self.root.is_dir():
            return []
        names = []
        for path in self.root.rglob("*"):
            if path.suffix.lower() not in MODEL_SUFFIXES or not path.is_file():
                continue
            if not path.resolve().is_relative_to(self.root):
                continue
            names.append(path.relative_to(self.root).as_posix())
        return sorted(names, key=str.casefold)

    def resolve_model(self, name: str | None) -> Path:
        """Reject absolute paths, traversal, and links outside the model directory."""
        if not isinstance(name, str) or not name or "\x00" in name:
            raise ValueError("Select a model from the models folder.")
        relative = Path(name)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Model paths must stay inside the models folder.")
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path.suffix.lower() not in MODEL_SUFFIXES:
            raise ValueError("Select a .pt or .pth file inside the models folder.")
        if not path.is_file():
            raise ValueError("The selected model no longer exists. Refresh the model list.")
        return path

    def load_player(self, name: str | None, simulations: int) -> AIPlayer:
        """Load current versioned weights on CUDA when available, otherwise CPU."""
        path = self.resolve_model(name)
        model_name = path.relative_to(self.root).as_posix()
        import torch

        from ..model.mcts import MCTS
        from ..utils.model_io import load_model_artifact

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = load_model_artifact(path, device=device)
        search = MCTS(model, num_simulations=simulations,
                      temperature=0.0, dirichlet_epsilon=0.0)
        return AIPlayer(model_name, model, search)
