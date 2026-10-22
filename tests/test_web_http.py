"""HTTP contracts for local model discovery, AI moves and static module assets."""
from http import HTTPStatus
from http.client import HTTPConnection
import json
from threading import Thread

import pytest

from scripts.play_web import GameWebServer
from src.web.ai_player import ModelRepository
from src.web.local_game import LocalGame


class LegalPlayer:
    name = "test_model.pt"
    device = "cpu"
    max_game_plies = 200
    no_capture_limit = 100

    def choose_move(self, state):
        return state.get_legal_moves()[0]


@pytest.fixture
def server(tmp_path, monkeypatch):
    models = ModelRepository(tmp_path)
    (tmp_path / "test_model.pt").touch()
    monkeypatch.setattr(models, "load_player", lambda *args: LegalPlayer())
    server = GameWebServer(("127.0.0.1", 0))
    server.game = LocalGame(models)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def send(server, method, path, payload=None):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        body = json.dumps(payload) if payload is not None else None
        connection.request(method, path, body=body, headers={"Content-Type": "application/json"})
        response = connection.getresponse()
        return response.status, json.loads(response.read())
    finally:
        connection.close()


def test_http_models_new_game_ai_reply_and_undo(server):
    status, response = send(server, "GET", "/api/models")
    assert status == HTTPStatus.OK
    assert response["models"] == ["test_model.pt"]
    status, response = send(server, "POST", "/api/new-game", {
        "mode": "ai", "model": "test_model.pt", "humanColor": "black", "simulations": 16,
    })
    assert status == HTTPStatus.OK and response["state"]["isAiTurn"]
    status, response = send(server, "POST", "/api/ai-move")
    assert status == HTTPStatus.OK and response["state"]["moveCount"] == 1
    move = response["state"]["legalMoves"][0]
    payload = dict(zip(("fromRow", "fromCol", "toRow", "toCol"), move))
    status, response = send(server, "POST", "/api/move", payload)
    assert status == HTTPStatus.OK and response["state"]["isAiTurn"]
    assert send(server, "POST", "/api/move", payload)[0] == HTTPStatus.BAD_REQUEST
    assert send(server, "POST", "/api/ai-move")[0] == HTTPStatus.OK
    status, response = send(server, "POST", "/api/undo")
    assert status == HTTPStatus.OK and response["state"]["moveCount"] == 1
    status, response = send(server, "POST", "/api/new-game")
    assert status == HTTPStatus.OK and response["state"]["mode"] == "human"


@pytest.mark.parametrize("payload", [[], "invalid", {"simulations": 0}, {"mode": "bad"}])
def test_http_invalid_settings_preserve_active_game(server, payload):
    _, before = send(server, "GET", "/api/state")
    status, response = send(server, "POST", "/api/new-game", payload)
    assert status == HTTPStatus.BAD_REQUEST and not response["ok"]
    assert send(server, "GET", "/api/state")[1] == before


@pytest.mark.parametrize("path,content_type", [
    ("/", "text/html"), ("/app.js", "application/javascript"),
    ("/api.js", "application/javascript"), ("/board-view.js", "application/javascript"),
    ("/game-controls.js", "application/javascript"), ("/game-controls.css", "text/css"),
])
def test_new_static_assets_are_served(server, path, content_type):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.request("GET", path)
        response = connection.getresponse()
        assert response.status == HTTPStatus.OK
        assert response.getheader("Content-Type").startswith(content_type)
        assert response.read()
    finally:
        connection.close()


def test_model_files_are_not_downloadable_static_assets(server):
    connection = HTTPConnection("127.0.0.1", server.server_port, timeout=10)
    try:
        connection.request("GET", "/models/test_model.pt")
        response = connection.getresponse()
        assert response.status == HTTPStatus.NOT_FOUND
        response.read()
    finally:
        connection.close()
