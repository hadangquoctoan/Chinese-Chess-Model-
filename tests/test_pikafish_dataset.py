"""Tests for game-level Pikafish dataset generation and splitting."""

import numpy as np
import pytest

from src.training.pikafish_dataset import (
    LabeledGame, flatten_games, generate_pikafish_games, split_train_validation,
)


def _fake_search(engine, state, depth, timeout):
    move = state.get_legal_moves()[0]
    return state.to_tensor(), np.full(1800, 1 / 1800, dtype=np.float32), 0.0, move


def test_generation_and_validation_split_are_by_game(monkeypatch):
    monkeypatch.setattr('src.training.pikafish_dataset._search_label', _fake_search)
    games = generate_pikafish_games(
        object(), num_games=3, depth=8, seed=7,
        opening_plies=2, max_game_plies=6, timeout=1.0)
    assert [game.game_id for game in games] == [0, 1, 2]
    assert all(len(game.states) == 6 for game in games)
    train, validation = split_train_validation(games, validation_games=1)
    assert {game.game_id for game in train}.isdisjoint(
        game.game_id for game in validation)
    assert [game.game_id for game in validation] == [2]


def test_invalid_validation_split_is_rejected():
    with pytest.raises(ValueError, match='validation_games'):
        split_train_validation([], validation_games=1)


def test_progress_callback_reports_each_completed_game(monkeypatch):
    monkeypatch.setattr('src.training.pikafish_dataset._search_label', _fake_search)
    reports = []

    generate_pikafish_games(
        object(), num_games=3, depth=8, seed=7,
        opening_plies=0, max_game_plies=4, timeout=1.0,
        progress=lambda completed, game: reports.append(
            (completed, game.game_id, len(game.states))))

    assert reports == [(1, 0, 4), (2, 1, 4), (3, 2, 4)]


def test_flatten_games_aligns_game_ids_with_positions():
    state = np.zeros((19, 10, 9), dtype=np.float32)
    policy = np.full(1800, 1 / 1800, dtype=np.float32)
    games = [
        LabeledGame(4, [state, state], [policy, policy], [0.0, 0.5]),
        LabeledGame(9, [state], [policy], [-0.5]),
    ]

    states, policies, values, game_ids = flatten_games(games)

    assert len(states) == len(policies) == len(values) == 3
    assert game_ids == [4, 4, 9]
    assert values == [0.0, 0.5, -0.5]


def test_flatten_games_rejects_empty_dataset():
    with pytest.raises(ValueError, match='empty'):
        flatten_games([])


def test_flatten_games_rejects_misaligned_game_lists():
    state = np.zeros((19, 10, 9), dtype=np.float32)
    policy = np.full(1800, 1 / 1800, dtype=np.float32)
    game = LabeledGame(5, [state, state], [policy], [0.0, 0.0])

    with pytest.raises(ValueError, match='game 5'):
        flatten_games([game])


def test_random_openings_change_generated_state_sequence(monkeypatch):
    monkeypatch.setattr('src.training.pikafish_dataset._search_label', _fake_search)

    def states(seed):
        game = generate_pikafish_games(
            object(), num_games=1, depth=8, seed=seed,
            opening_plies=4, max_game_plies=6, timeout=1.0)[0]
        return np.stack(game.states)

    first = states(11)
    assert np.array_equal(first, states(11))
    assert not np.array_equal(first, states(12))


def test_dataset_archives_record_state_limit_metadata(tmp_path):
    from pathlib import Path

    from scripts.build_pikafish_dataset import _save_games
    from scripts.build_teacher_labels import load_teacher_npz
    from src.game.game_state import NO_CAPTURE_DRAW_PLY_LIMIT

    state = np.zeros((19, 10, 9), dtype=np.float32)
    policy = np.full(1800, 1 / 1800, dtype=np.float32)
    path = tmp_path / 'train.npz'

    count = _save_games(path, [LabeledGame(7, [state], [policy], [0.0])],
                        depth=8, engine_path=Path('pikafish'),
                        max_game_plies=24)

    loaded = load_teacher_npz(path)
    assert count == 1
    assert loaded['game_ids'].tolist() == [7]
    assert loaded['max_game_plies'] == 24
    assert loaded['no_capture_limit'] == NO_CAPTURE_DRAW_PLY_LIMIT
