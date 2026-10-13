"""Regression tests for model-consuming command entrypoints."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import build_models, evaluate, play_cli
from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet


def test_train_cli_exposes_explicit_checkpoint_fork(monkeypatch):
    from scripts import train
    calls = []
    class FakeTrainer:
        def close(self):
            calls.append("close")
    class FakeSession:
        trainer = FakeTrainer()
        iteration = 1
        best_score = 0.5
        def start(self, **options):
            calls.append(options)
        def run(self, **options):
            calls.append("run")
    monkeypatch.setattr(train, "load_config", lambda path: {"training_loop": {}, "paths": {"log_dir": "unused"}})
    monkeypatch.setattr(train, "create_session", lambda *args: FakeSession())
    import logging
    monkeypatch.setattr(train, "setup_logger", lambda **options: logging.getLogger("cli-test"))
    train.main(["--resume", "checkpoint.pt", "--publish-missing-best"])
    assert calls[0] == {"resume": "checkpoint.pt", "publish_missing_best": True}
    assert calls[-1] == "close"


def test_move_parser_rejects_unicode_digits():
    assert play_cli.parse_move("a²-a1") is None


def test_build_models_does_not_overwrite_existing_output(tmp_path):
    target = tmp_path / "untrained_small.pt"
    target.write_bytes(b"keep-existing-model")
    with pytest.raises(FileExistsError):
        build_models.build_one("small", tmp_path)
    assert target.read_bytes() == b"keep-existing-model"


def test_evaluate_uses_artifact_architecture_not_medium(monkeypatch, capsys):
    model = ChineseChessNet(ModelConfig(num_channels=4, num_residual_blocks=0,
                                       policy_channels=1, value_channels=1))
    calls = []
    def load(path, device="cpu"):
        calls.append(path)
        return model
    monkeypatch.setattr(evaluate, "load_model_artifact", load, raising=False)
    monkeypatch.setattr(evaluate.Evaluator, "evaluate", lambda *a, **k: (0.5, 0, 0, 2))
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--model", "tiny.pt", "--games", "2"])
    evaluate.main()
    assert calls == ["tiny.pt"]
    output = capsys.readouterr().out
    assert "self-play" in output.lower()
    assert "Red wins:" not in output


def test_cli_play_loads_versioned_artifact(monkeypatch):
    model = ChineseChessNet(ModelConfig(num_channels=4, num_residual_blocks=0,
                                       policy_channels=1, value_channels=1))
    calls = []
    def load(path, device="cpu"):
        calls.append(path)
        return model
    monkeypatch.setattr("src.utils.model_io.load_model_artifact", load, raising=False)
    monkeypatch.setattr(sys, "argv", ["play_cli.py", "--model", "tiny.pt"])
    monkeypatch.setattr("builtins.input", lambda *a: "quit")
    play_cli.main()
    assert calls == ["tiny.pt"]
