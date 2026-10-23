"""Tests for worker_select.choose_worker_for_game and sparring_config validation."""
import random

import pytest

from src.training.sparring_config import validate_sparring_config
from src.training.worker_select import choose_worker_for_game


def test_ratio_03_gives_about_300_sparring_per_1000():
    rng = random.Random(0)
    count = sum(1 for _ in range(1000) if choose_worker_for_game(rng, 0.3, True) == "sparring")
    assert 250 <= count <= 350


def test_disabled_always_self_even_with_ratio_one():
    rng = random.Random(0)
    for _ in range(50):
        assert choose_worker_for_game(rng, 1, False) == "self"


def test_ratio_zero_always_self():
    rng = random.Random(1)
    for _ in range(50):
        assert choose_worker_for_game(rng, 0, True) == "self"


def test_ratio_one_always_sparring_when_enabled():
    rng = random.Random(1)
    for _ in range(50):
        assert choose_worker_for_game(rng, 1, True) == "sparring"


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf"), "0.3", None])
def test_invalid_ratio_raises(bad):
    with pytest.raises(ValueError):
        choose_worker_for_game(random.Random(0), bad, True)


def test_missing_section_returns_disabled_defaults():
    cfg = validate_sparring_config(None)
    assert cfg == {
        "enabled": False,
        "ratio": 0.3,
        "depth": 1,
        "threads": 8,
        "hash_mb": 1024,
        "engine_path": None,
    }


@pytest.mark.parametrize("section", [
    {"enabled": True, "ratio": -0.1, "engine_path": "whatever"},
    {"enabled": True, "ratio": 1.5, "engine_path": "whatever"},
    {"enabled": True, "ratio": float("nan"), "engine_path": "whatever"},
])
def test_bad_ratio_rejected(section):
    with pytest.raises(ValueError):
        validate_sparring_config(section)


@pytest.mark.parametrize("section", [
    {"enabled": True, "depth": 0, "engine_path": "whatever"},
    {"enabled": True, "depth": 31, "engine_path": "whatever"},
    {"enabled": True, "depth": 2.5, "engine_path": "whatever"},
])
def test_bad_depth_rejected(section):
    with pytest.raises(ValueError):
        validate_sparring_config(section)


@pytest.mark.parametrize("section", [
    {"enabled": True, "threads": 0, "engine_path": "whatever"},
    {"enabled": True, "threads": -4, "engine_path": "whatever"},
    {"enabled": True, "hash_mb": 0, "engine_path": "whatever"},
])
def test_bad_threads_or_hash_rejected(section):
    with pytest.raises(ValueError):
        validate_sparring_config(section)


def test_enabled_with_missing_engine_file_rejected():
    with pytest.raises(ValueError):
        validate_sparring_config({"enabled": True, "engine_path": "no/such/pikafish.exe"})


def test_disabled_with_missing_engine_accepted():
    cfg = validate_sparring_config({"enabled": False, "engine_path": "no/such/pikafish.exe"})
    assert cfg["enabled"] is False


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        validate_sparring_config({"enabled": False, "bogus": 1})


def test_choose_sparring_depth_starts_at_one():
    from src.training.sparring_depth import choose_sparring_depth

    rng = random.Random(0)
    assert choose_sparring_depth(rng, 0, 0, 0, 0) == 1
    assert choose_sparring_depth(rng, 0, 10, 0, 0) == 1
    assert choose_sparring_depth(rng, 4, 10, 0, 0) == 1


def test_choose_sparring_depth_mixes_after_depth1_wins():
    from src.training.sparring_depth import choose_sparring_depth

    rng = random.Random(0)
    picks = [choose_sparring_depth(rng, 7, 10, 0, 0) for _ in range(300)]
    assert 1 in picks and 2 in picks
    assert 70 <= picks.count(2) <= 130


def test_choose_sparring_depth_half_when_both_thresholds_pass():
    from src.training.sparring_depth import choose_sparring_depth

    rng = random.Random(0)
    picks = [choose_sparring_depth(rng, 8, 10, 5, 10) for _ in range(400)]
    assert 160 <= picks.count(2) <= 240


def test_train_depth2_yaml_has_sparring_section():
    from pathlib import Path

    import yaml

    from src.training.pipeline import CONFIG_FIELDS

    path = Path("configs/train_depth2_16h.yaml")
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert set(config) <= set(CONFIG_FIELDS)
    assert config["model"]["size"] == "small"
    assert config["training"]["batch_size"] == 2048
    assert config["self_play"]["num_games_per_iteration"] == 48
    assert config["self_play"]["mcts_simulations"] == 128
    assert config["self_play"]["parallel_workers"] == 1
    assert config["training_loop"]["evaluation_games"] == 8
    assert config["training_loop"]["samples_per_iteration"] == 16000
    assert config["sparring"]["enabled"] is True
    assert config["sparring"]["ratio"] == 0.3
    assert config["sparring"]["depth"] == 1
    assert set(config["sparring"]) <= CONFIG_FIELDS["sparring"]
    from src.training.sparring_config import validate_sparring_config
    if Path(config["sparring"]["engine_path"]).is_file() or (
            Path("pikafish") / "Pikafish-Linux-x86-64-universal").is_file():
        validated = validate_sparring_config(config["sparring"])
        assert validated["enabled"] is True


def test_train_laptop_8h_yaml_fits_rtx3000_and_windows_engine():
    from pathlib import Path

    import yaml

    from src.training.pipeline import CONFIG_FIELDS, validate_session_config

    rented = yaml.safe_load(Path("configs/train_depth2_16h.yaml").read_text(encoding="utf-8"))
    config = yaml.safe_load(Path("configs/train_laptop_8h.yaml").read_text(encoding="utf-8"))
    assert set(config) <= set(CONFIG_FIELDS)
    assert config["model"]["size"] == "small"
    assert config["training"]["batch_size"] == 64
    assert config["training"]["use_amp"] is True
    assert config["self_play"]["mcts_simulations"] == 64
    assert config["self_play"]["num_games_per_iteration"] == 8
    assert config["optimization"]["num_workers"] == 0
    assert config["optimization"]["max_batch_size"] == 32
    assert config["sparring"]["threads"] == 4
    assert config["sparring"]["hash_mb"] == 256
    assert "Windows" in config["sparring"]["engine_path"]
    assert config["paths"]["best_model_path"] != rented["paths"]["best_model_path"]
    assert config["paths"]["checkpoint_dir"] != rented["paths"]["checkpoint_dir"]
    engine = Path(config["sparring"]["engine_path"])
    if not engine.is_file():
        engine = Path("pikafish") / "Pikafish-Windows-x86-64-universal.exe"
    if engine.is_file():
        validate_session_config(config)


def test_train_depth2_16h_windows_config_uses_windows_engine_and_parallel_workers():
    from pathlib import Path

    import yaml

    from src.training.sparring_config import validate_sparring_config

    config = yaml.safe_load(
        Path("configs/train_depth2_16h_windows.yaml").read_text(encoding="utf-8"))
    assert config["self_play"]["parallel_workers"] == 8
    assert config["training_loop"]["checkpoint_every"] == 5
    assert config["training_loop"]["auto_checkpoint_interval"] == 1800
    engine = Path(config["sparring"]["engine_path"])
    assert engine.name == "Pikafish-Windows-x86-64-universal.exe"
    if engine.is_file():
        validated = validate_sparring_config(config["sparring"])
        assert validated["engine_path"].endswith(engine.name)


def test_train_server_fast_config_uses_light_cpu_bound_budget():
    from pathlib import Path

    import yaml

    from src.training.sparring_config import validate_sparring_config

    config = yaml.safe_load(
        Path("configs/train_server_fast.yaml").read_text(encoding="utf-8"))
    assert config["self_play"]["parallel_workers"] == 12
    assert config["self_play"]["num_games_per_iteration"] == 16
    assert config["self_play"]["mcts_simulations"] == 64
    assert config["training_loop"]["evaluation_games"] == 4
    assert config["training_loop"]["evaluation_mcts_simulations"] == 24
    assert config["sparring"]["threads"] == 4
    assert config["sparring"]["hash_mb"] == 256
    engine = Path(config["sparring"]["engine_path"])
    assert engine.name == "Pikafish-Windows-x86-64-universal.exe"
    if engine.is_file():
        validate_sparring_config(config["sparring"])


def test_enabled_with_existing_engine_accepted(tmp_path):
    binary = tmp_path / "pikafish"
    binary.write_bytes(b"x")
    cfg = validate_sparring_config({
        "enabled": True, "ratio": 0.3, "depth": 1, "threads": 8,
        "hash_mb": 64, "engine_path": str(binary),
    })
    assert cfg["enabled"] is True
    assert cfg["engine_path"] == str(binary)
