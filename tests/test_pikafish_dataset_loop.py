"""Tests for continuous Pikafish shard generation and merging."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.build_teacher_labels import save_teacher_npz
from scripts.run_pikafish_dataset_loop import (
    _check_metadata, _pending_shards, _remove_partial_files,
    merge_shards, next_game_id,
)
from src.game.game_state import NO_CAPTURE_DRAW_PLY_LIMIT


def _write_dataset(path: Path, game_ids: list[int], depth: int = 8,
                   max_game_plies: int = 160) -> None:
    count = len(game_ids)
    states = np.zeros((count, 19, 10, 9), dtype=np.float32)
    policies = np.full((count, 1800), 1 / 1800, dtype=np.float32)
    values = np.zeros(count, dtype=np.float32)
    save_teacher_npz(path, states, policies, values, depth=depth,
                     engine_path=Path('pikafish'), game_ids=game_ids,
                     max_game_plies=max_game_plies,
                     no_capture_limit=NO_CAPTURE_DRAW_PLY_LIMIT)


def test_next_game_id_uses_train_and_validation(tmp_path: Path):
    train = tmp_path / 'train.npz'
    validation = tmp_path / 'validation.npz'
    _write_dataset(train, [0, 1, 2])
    _write_dataset(validation, [3, 4])
    assert next_game_id(train, validation) == 5


def test_merge_shards_appends_positions_and_game_ids(tmp_path: Path):
    train = tmp_path / 'train.npz'
    shard = tmp_path / 'shard.npz'
    _write_dataset(train, [0, 1])
    _write_dataset(shard, [2, 3])

    positions, games = merge_shards(
        train, [shard], depth=8, engine_path=Path('pikafish'),
        max_game_plies=160)

    with np.load(train, allow_pickle=False) as archive:
        assert positions == 4
        assert games == 4
        assert archive['states'].shape == (4, 19, 10, 9)
        assert archive['policies'].shape == (4, 1800)
        assert archive['game_ids'].tolist() == [0, 1, 2, 3]


def test_merge_shards_allows_repeated_positions_within_game(
        tmp_path: Path):
    train = tmp_path / 'train.npz'
    shard = tmp_path / 'shard.npz'
    _write_dataset(train, [0, 0, 1, 1])
    _write_dataset(shard, [2, 2])

    positions, games = merge_shards(
        train, [shard], depth=8, engine_path=Path('pikafish'),
        max_game_plies=160)

    assert positions == 6
    assert games == 3


def test_merge_shards_rejects_overlapping_game_ids(tmp_path: Path):
    train = tmp_path / 'train.npz'
    shard = tmp_path / 'shard.npz'
    _write_dataset(train, [0, 1])
    _write_dataset(shard, [1, 2])

    with pytest.raises(ValueError, match='overlap'):
        merge_shards(train, [shard], depth=8,
                     engine_path=Path('pikafish'), max_game_plies=160)


def test_check_metadata_rejects_depth_mismatch(tmp_path: Path):
    archive = tmp_path / 'depth2.npz'
    _write_dataset(archive, [0], depth=2)

    with pytest.raises(ValueError, match='engine_depth'):
        _check_metadata(archive, depth=8, max_game_plies=160)


def test_remove_partial_files_keeps_complete_shards(tmp_path: Path):
    train = tmp_path / 'train.npz'
    shards_dir = tmp_path / 'shards'
    shards_dir.mkdir()
    train_partial = train.with_name('train.partial.npz')
    shard_partial = shards_dir / 'shard.partial.npz'
    complete = shards_dir / 'shard.npz'
    for path in (train_partial, shard_partial, complete):
        path.write_bytes(b'x')
    args = SimpleNamespace(train_output=train, shards_dir=shards_dir)

    _remove_partial_files(args)

    assert not train_partial.exists()
    assert not shard_partial.exists()
    assert complete.exists()


def test_pending_shards_skip_already_merged_and_return_new(tmp_path: Path):
    train = tmp_path / 'train.npz'
    shards_dir = tmp_path / 'shards'
    shards_dir.mkdir()
    _write_dataset(train, [0, 1])
    merged = shards_dir / 'merged.npz'
    pending = shards_dir / 'pending.npz'
    _write_dataset(merged, [0, 1])
    _write_dataset(pending, [2, 3])

    result = _pending_shards(train, shards_dir)

    assert result == [pending]
    assert not merged.exists()
