"""Tests for merging per-depth Pikafish farm shards."""

from pathlib import Path

import numpy as np
import pytest

import scripts.merge_pikafish_farm as merge_farm
from scripts.build_teacher_labels import save_teacher_npz
from src.game.game_state import NO_CAPTURE_DRAW_PLY_LIMIT
from src.training.shard_plan import depth_dir, shard_filename


def _write_shard(path: Path, game_ids: list[int], depth: int = 3,
                 max_game_plies: int = 160) -> None:
    """Write one synthetic shard with one labeled position per game id."""
    count = len(game_ids)
    states = np.zeros((count, 19, 10, 9), dtype=np.float32)
    policies = np.full((count, 1800), 1 / 1800, dtype=np.float32)
    values = np.zeros(count, dtype=np.float32)
    save_teacher_npz(path, states, policies, values, depth=depth,
                     engine_path=Path('pikafish'), game_ids=game_ids,
                     max_game_plies=max_game_plies,
                     no_capture_limit=NO_CAPTURE_DRAW_PLY_LIMIT)


def _merge_argv(out_root: Path, output: Path) -> list[str]:
    """Build a depth-3 merge command line writing to ``output``."""
    return ['--out-root', str(out_root), '--depth', '3',
            '--output', str(output)]


@pytest.fixture(autouse=True)
def _stub_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep tests independent of the platform-specific Pikafish binary."""
    monkeypatch.setattr(merge_farm, 'find_engine_binary',
                        lambda: Path('pikafish'))


def test_merge_two_shards(tmp_path: Path) -> None:
    out_root = tmp_path / 'farm'
    shard_dir = depth_dir(out_root, 3)
    first = shard_dir / shard_filename(3, 30_000_000, 1)
    second = shard_dir / shard_filename(3, 30_000_001, 2)
    _write_shard(first, [30_000_000])
    _write_shard(second, [30_000_001, 30_000_002])
    output = tmp_path / 'train_depth3.npz'

    assert merge_farm.main(_merge_argv(out_root, output)) == 0

    with np.load(output, allow_pickle=False) as archive:
        assert archive['states'].shape == (3, 19, 10, 9)
        assert archive['game_ids'].tolist() == [
            30_000_000, 30_000_001, 30_000_002]
        assert int(archive['engine_depth']) == 3
    assert first.is_file()
    assert second.is_file()


def test_merge_is_idempotent(
        tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out_root = tmp_path / 'farm'
    shard_dir = depth_dir(out_root, 3)
    _write_shard(shard_dir / shard_filename(3, 30_000_000, 1), [30_000_000])
    _write_shard(shard_dir / shard_filename(3, 30_000_001, 2),
                 [30_000_001, 30_000_002])
    output = tmp_path / 'train_depth3.npz'
    assert merge_farm.main(_merge_argv(out_root, output)) == 0
    original = output.read_bytes()
    original_mtime = output.stat().st_mtime_ns
    capsys.readouterr()

    assert merge_farm.main(_merge_argv(out_root, output)) == 0

    captured = capsys.readouterr()
    assert 'nothing to merge' in captured.out
    assert output.read_bytes() == original
    assert output.stat().st_mtime_ns == original_mtime
    assert merge_farm.select_new_shards(output, shard_dir) == []


def test_merge_appends_new_shard_later(tmp_path: Path) -> None:
    out_root = tmp_path / 'farm'
    shard_dir = depth_dir(out_root, 3)
    _write_shard(shard_dir / shard_filename(3, 30_000_000, 1), [30_000_000])
    output = tmp_path / 'train_depth3.npz'
    assert merge_farm.main(_merge_argv(out_root, output)) == 0

    _write_shard(shard_dir / shard_filename(3, 30_000_001, 2),
                 [30_000_001, 30_000_002])
    assert merge_farm.main(_merge_argv(out_root, output)) == 0

    with np.load(output, allow_pickle=False) as archive:
        assert archive['states'].shape[0] == 3
        assert archive['game_ids'].tolist() == [
            30_000_000, 30_000_001, 30_000_002]


def test_partial_overlap_rejected(tmp_path: Path) -> None:
    shard_dir = depth_dir(tmp_path / 'farm', 3)
    shard = shard_dir / shard_filename(3, 30_000_000, 2)
    _write_shard(shard, [30_000_000, 30_000_001])
    output = tmp_path / 'train_depth3.npz'
    _write_shard(output, [30_000_001, 30_000_002])

    with pytest.raises(ValueError, match='overlap'):
        merge_farm.select_new_shards(output, shard_dir)


def test_metadata_mismatch_rejected(tmp_path: Path) -> None:
    out_root = tmp_path / 'farm'
    shard = depth_dir(out_root, 3) / shard_filename(3, 30_000_000, 1)
    _write_shard(shard, [30_000_000], max_game_plies=100)
    output = tmp_path / 'train_depth3.npz'

    with pytest.raises(ValueError, match='max_game_plies'):
        merge_farm.main(_merge_argv(out_root, output))
    assert not output.exists()
