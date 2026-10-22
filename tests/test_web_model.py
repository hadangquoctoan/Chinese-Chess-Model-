"""A tiny real NN/MCTS artifact roundtrip through the web-game service."""
import numpy as np
import pytest

from src.web.ai_player import ModelRepository
from src.web.local_game import LocalGame


def test_web_loads_artifact_and_plays_real_mcts_on_cpu(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    from src.model.model_config import ModelConfig
    from src.model.neural_network import ChineseChessNet
    from src.utils.model_io import atomic_torch_save, build_model_artifact

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    config = ModelConfig(num_residual_blocks=0, num_channels=4,
                         policy_channels=1, value_channels=1)
    model = ChineseChessNet(config).eval()
    atomic_torch_save(build_model_artifact(model), tmp_path / "tiny.pt")
    game = LocalGame(ModelRepository(tmp_path))
    response = game.new_game(mode="ai", model="tiny.pt", human_color="black", simulations=1)
    assert response["ok"], response.get("error")
    assert response["state"]["device"] == "cpu"
    initial = game._state.to_tensor().copy()
    legal_moves = game.snapshot()["legalMoves"]
    response = game.ai_move()
    assert response["ok"], response.get("error")
    assert response["state"]["lastMove"] in legal_moves
    assert response["state"]["moveCount"] == 1
    assert not np.array_equal(game._state.to_tensor(), initial)


def test_web_rejects_unversioned_weights_without_resetting_game(tmp_path):
    torch = pytest.importorskip("torch")
    torch.save({"weight": torch.zeros(1)}, tmp_path / "legacy.pt")
    game = LocalGame(ModelRepository(tmp_path))
    before = game.snapshot()
    response = game.new_game(mode="ai", model="legacy.pt")
    assert not response["ok"]
    assert "Legacy" in response["error"]
    assert response["state"] == before


def test_web_reports_truncated_model_without_resetting_game(tmp_path):
    pytest.importorskip("torch")
    (tmp_path / "empty.pt").touch()
    game = LocalGame(ModelRepository(tmp_path))
    before = game.snapshot()
    response = game.new_game(mode="ai", model="empty.pt")
    assert not response["ok"]
    assert response["state"] == before
