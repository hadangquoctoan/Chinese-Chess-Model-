"""Tests for shard naming, durable game-id progress and target parsing."""

import pytest

from src.training.shard_plan import (
    DepthProgress,
    DepthTarget,
    game_id_base,
    load_progress,
    parse_shard_filename,
    parse_targets,
    remove_partial_files,
    save_progress,
    shard_filename,
)


def test_parse_targets_ok():
    assert parse_targets('3:200,4:150,9:inf') == [
        DepthTarget(3, 200), DepthTarget(4, 150), DepthTarget(9, None)]


@pytest.mark.parametrize('text', [
    '', '3', '3:', '0:5', '3:0', '3:-1', '3:abc', '3:10,3:20', '9:inf,3:10'])
def test_parse_targets_rejects(text):
    with pytest.raises(ValueError):
        parse_targets(text)


def test_game_id_base():
    assert game_id_base(8) == 80_000_000


def test_shard_filename_roundtrip():
    name = shard_filename(8, 80_000_040, 8)
    assert name == 'd8_g80000040_n8.npz'
    assert parse_shard_filename(name) == (8, 80_000_040, 8)


def test_parse_shard_filename_ignores_others():
    assert parse_shard_filename('d8_g1_n1.partial.npz') is None
    assert parse_shard_filename('x.npz') is None
    assert parse_shard_filename('progress.json') is None


def test_load_progress_empty_dir(tmp_path):
    assert load_progress(tmp_path, 3) == DepthProgress(30_000_000, 0)


def test_load_progress_from_shard_names(tmp_path):
    for name in ('d3_g30000000_n20.npz', 'd3_g30000020_n20.npz',
                 'd3_g30000040_n5.partial.npz'):
        (tmp_path / name).touch()
    assert load_progress(tmp_path, 3) == DepthProgress(30_000_040, 40)


def test_progress_json_survives_shard_deletion(tmp_path):
    save_progress(tmp_path, DepthProgress(30_000_100, 80))
    assert load_progress(tmp_path, 3) == DepthProgress(30_000_100, 80)

    (tmp_path / 'd3_g30000090_n20.npz').touch()
    assert load_progress(tmp_path, 3) == DepthProgress(30_000_110, 80)


def test_corrupt_progress_json_is_ignored(tmp_path):
    (tmp_path / 'progress.json').write_text('{not json', encoding='utf-8')
    assert load_progress(tmp_path, 3) == DepthProgress(30_000_000, 0)


def test_remove_partial_files(tmp_path):
    partial_a = tmp_path / 'd3_g30000000_n20.partial.npz'
    partial_b = tmp_path / 'd3_g30000020_n20.partial.npz'
    shard = tmp_path / 'd3_g30000000_n20.npz'
    for path in (partial_a, partial_b, shard):
        path.touch()

    assert remove_partial_files(tmp_path) == 2
    assert shard.exists()
    assert not partial_a.exists()
    assert not partial_b.exists()
