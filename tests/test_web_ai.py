"""Model discovery and web-game turn management without requiring PyTorch."""
import pytest

from src.web.ai_player import MAX_SIMULATIONS, ModelRepository
from src.web.local_game import LocalGame


class LegalPlayer:
    """A deterministic opponent for game-service contract tests."""

    name = "test_model.pt"
    device = "cpu"
    max_game_plies = 200
    no_capture_limit = 100

    def choose_move(self, state):
        return state.get_legal_moves()[0]


def ai_game(monkeypatch, human_color="red", player=None):
    models = ModelRepository()
    monkeypatch.setattr(models, "load_player", lambda *args: player or LegalPlayer())
    game = LocalGame(models)
    response = game.new_game(mode="ai", model="test_model.pt", human_color=human_color)
    assert response["ok"]
    return game


def test_discovery_lists_only_model_paths_and_rejects_traversal(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    (root / "test_model.pt").touch()
    (root / "note.txt").touch()
    (root / "archive").mkdir()
    (root / "archive" / "other.pth").touch()
    models = ModelRepository(root)
    assert models.list_models() == ["archive/other.pth", "test_model.pt"]
    assert models.resolve_model("test_model.pt") == root / "test_model.pt"
    for name in ("../outside.pt", str(tmp_path / "outside.pt"), "note.txt", "missing.pt", None, "bad\x00.pt"):
        with pytest.raises(ValueError):
            models.resolve_model(name)
    assert ModelRepository(tmp_path / "absent").list_models() == []


def test_model_links_cannot_escape_models_directory(tmp_path):
    root = tmp_path / "models"
    root.mkdir()
    outside = tmp_path / "outside.pt"
    outside.touch()
    try:
        (root / "linked.pt").symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("Creating symlinks is not available on this platform")
    models = ModelRepository(root)
    assert models.list_models() == []
    with pytest.raises(ValueError, match="inside"):
        models.resolve_model("linked.pt")


def test_ai_reply_and_undo_restore_the_human_turn(monkeypatch):
    game = ai_game(monkeypatch)
    initial = game.snapshot()
    assert not initial["isAiTurn"]
    assert game.move(tuple(initial["legalMoves"][0]))["ok"]
    assert game.snapshot()["isAiTurn"]
    assert not game.move(tuple(game.snapshot()["legalMoves"][0]))["ok"]
    response = game.ai_move()
    assert response["ok"] and response["state"]["moveCount"] == 2
    assert not response["state"]["isAiTurn"]
    assert not game.ai_move()["ok"]
    response = game.undo()
    assert response["ok"]
    assert response["state"]["board"] == initial["board"]
    assert response["state"]["moveCount"] == 0


def test_black_human_gets_ai_opening_and_undo_keeps_it(monkeypatch):
    game = ai_game(monkeypatch, human_color="black")
    assert game.snapshot()["isAiTurn"]
    assert not game.undo()["ok"]
    opening = game.ai_move()["state"]
    assert opening["moveCount"] == 1 and not opening["canUndo"]
    assert not game.undo()["ok"]
    assert game.move(tuple(opening["legalMoves"][0]))["ok"]
    assert game.ai_move()["ok"]
    restored = game.undo()["state"]
    assert restored["moveCount"] == 1
    assert restored["board"] == opening["board"]
    assert not restored["isAiTurn"]


def test_pending_human_move_can_be_undone_without_ai_reply(monkeypatch):
    game = ai_game(monkeypatch)
    assert game.move(tuple(game.snapshot()["legalMoves"][0]))["ok"]
    response = game.undo()
    assert response["ok"] and response["state"]["moveCount"] == 0
    assert not response["state"]["isAiTurn"]


def test_bad_model_or_missing_torch_preserves_the_active_game(monkeypatch):
    models = ModelRepository()
    game = LocalGame(models)
    assert game.move(tuple(game.snapshot()["legalMoves"][0]))["ok"]
    before = game.snapshot()
    for error in (ValueError("Incompatible encoding"), ImportError("No torch")):
        def fail(*args):
            raise error
        monkeypatch.setattr(models, "load_player", fail)
        response = game.new_game(mode="ai", model="bad.pt")
        assert not response["ok"]
        assert response["state"] == before


def test_failed_ai_search_preserves_board_and_allows_retry(monkeypatch):
    player = LegalPlayer()
    game = ai_game(monkeypatch, human_color="black", player=player)
    before = game.snapshot()
    def fail(state):
        state.make_move(*state.get_legal_moves()[0])
        raise RuntimeError("Inference failed")
    monkeypatch.setattr(player, "choose_move", fail)
    response = game.ai_move()
    assert not response["ok"] and response["state"] == before
    monkeypatch.setattr(player, "choose_move", lambda state: state.get_legal_moves()[0])
    assert game.ai_move()["ok"]


def test_terminal_human_move_does_not_request_ai_and_undo_restores_turn(monkeypatch):
    player = LegalPlayer()
    player.max_game_plies = 1
    game = ai_game(monkeypatch, player=player)
    response = game.move(tuple(game.snapshot()["legalMoves"][0]))
    assert response["state"]["isTerminal"]
    assert not response["state"]["isAiTurn"]
    assert not game.ai_move()["ok"]
    assert game.undo()["state"]["isRedTurn"]


def test_terminal_ai_reply_can_be_undone_as_a_full_turn(monkeypatch):
    player = LegalPlayer()
    player.max_game_plies = 2
    game = ai_game(monkeypatch, player=player)
    before = game.snapshot()
    assert game.move(tuple(before["legalMoves"][0]))["ok"]
    response = game.ai_move()
    assert response["ok"] and response["state"]["isTerminal"]
    assert not response["state"]["isAiTurn"]
    restored = game.undo()["state"]
    assert restored["moveCount"] == 0 and not restored["isTerminal"]
    assert restored["board"] == before["board"]


def test_switching_back_to_humans_and_model_rule_limits(monkeypatch):
    player = LegalPlayer()
    player.max_game_plies, player.no_capture_limit = 12, 7
    game = ai_game(monkeypatch, player=player)
    assert game._state.max_plies == 12
    assert game._state.no_capture_limit == 7
    response = game.new_game()
    assert response["ok"] and response["state"]["mode"] == "human"
    assert response["state"]["model"] is None
    assert game.move(tuple(game.snapshot()["legalMoves"][0]))["ok"]
    assert game.move(tuple(game.snapshot()["legalMoves"][0]))["ok"]
    assert not game.ai_move()["ok"]


@pytest.mark.parametrize("options", [
    {"mode": "invalid"}, {"human_color": "green"}, {"simulations": True},
    {"simulations": 0}, {"simulations": MAX_SIMULATIONS + 1}, {"simulations": 1.5},
])
def test_invalid_game_settings_do_not_reset_game(options):
    game = LocalGame()
    before = game.snapshot()
    response = game.new_game(**options)
    assert not response["ok"] and response["state"] == before


def test_local_human_game_does_not_import_pytorch():
    import subprocess
    import sys

    code = ("import sys; from scripts.play_web import LocalGame; "
            "LocalGame().snapshot(); "
            "assert not any(name == 'torch' or name.startswith('torch.') for name in sys.modules)")
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
