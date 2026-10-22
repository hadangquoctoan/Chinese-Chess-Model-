"""Web-game Pikafish opponent still goes through project Rules."""
from src.engine.player import PikafishPlayer
from src.web.ai_player import ModelRepository
from src.web.local_game import LocalGame


class FakeEngine:
    def __init__(self, token: str = "h2e2") -> None:
        self.token = token
        self.closed = False

    def bestmove(self, fen, moves=None, depth=8, timeout=60.0):
        return self.token

    def close(self) -> None:
        self.closed = True


def test_pikafish_web_mode_applies_converted_move_through_rules(monkeypatch):
    engine = FakeEngine("h2e2")

    def load(depth=8, path=None):
        return PikafishPlayer(engine, depth=depth)

    monkeypatch.setattr("src.engine.player.load_pikafish_player", load)
    game = LocalGame(ModelRepository())
    response = game.new_game(mode="pikafish", human_color="black", simulations=4)
    assert response["ok"]
    assert response["state"]["mode"] == "pikafish"
    assert response["state"]["isAiTurn"]
    assert response["state"]["model"] == "Pikafish"
    reply = game.ai_move()
    assert reply["ok"]
    assert reply["state"]["lastMove"] == [2, 7, 2, 4]
    assert reply["state"]["moveCount"] == 1
    assert not reply["state"]["isAiTurn"]


def test_illegal_pikafish_token_does_not_mutate_the_board(monkeypatch):
    engine = FakeEngine("e0e2")
    monkeypatch.setattr(
        "src.engine.player.load_pikafish_player",
        lambda depth=8, path=None: PikafishPlayer(engine, depth=depth),
    )
    game = LocalGame()
    assert game.new_game(mode="pikafish", human_color="black", simulations=4)["ok"]
    before = game.snapshot()
    response = game.ai_move()
    assert not response["ok"]
    assert "not legal" in response["error"]
    assert response["state"] == before


def test_invalid_pikafish_depth_preserves_the_active_game():
    game = LocalGame()
    assert game.move(tuple(game.snapshot()["legalMoves"][0]))["ok"]
    before = game.snapshot()
    response = game.new_game(mode="pikafish", simulations=99)
    assert not response["ok"]
    assert response["state"] == before
