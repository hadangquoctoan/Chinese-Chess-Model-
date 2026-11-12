"""Tests for the multi-depth Pikafish farm CLI."""

import argparse
from pathlib import Path

import numpy as np
import pytest

import scripts.run_pikafish_farm as farm
from src.training.job_queue import ShardJob


def _pikafish_available() -> bool:
    """Return whether the bundled binary and NNUE weights are present."""
    from src.engine.uci import ENGINE_DIR, NNUE_NAME, find_engine_binary

    try:
        binary = find_engine_binary()
    except FileNotFoundError:
        return False
    return all((ENGINE_DIR / name).is_file()
               for name in (binary.name, NNUE_NAME))


def _fixed_options() -> argparse.Namespace:
    """Return a deterministic options namespace for command building."""
    return argparse.Namespace(
        python='python.exe', seed=1000, opening_plies=12,
        max_game_plies=160, threads=1, hash_mb=64, timeout=60.0)


def test_default_workers():
    assert farm.default_workers(56) == 54
    assert farm.default_workers(2) == 1
    assert farm.default_workers(None) == 1


def test_build_shard_command(tmp_path: Path):
    job = ShardJob(
        depth=3, game_id_start=30_000_005, games=20,
        path=tmp_path / 'depth3' / 'd3_g30000005_n20.npz')
    options = _fixed_options()
    expected = [
        'python.exe',
        str(tmp_path / 'scripts' / 'generate_pikafish_shard.py'),
        '--output', str(job.path),
        '--games', '20',
        '--game-id-start', '30000005',
        '--seed', '30001005',
        '--depth', '3',
        '--opening-plies', '12',
        '--max-game-plies', '160',
        '--threads', '1',
        '--hash-mb', '64',
        '--timeout', '60.0',
    ]
    assert farm.build_shard_command(job, options, tmp_path) == expected


def test_prepare_stop_file_removes_stale(tmp_path: Path):
    stop_file = tmp_path / 'STOP'
    stop_file.write_text('stop', encoding='utf-8')

    assert farm.prepare_stop_file(stop_file) is True
    assert not stop_file.exists()
    assert farm.prepare_stop_file(stop_file) is False


@pytest.mark.parametrize('argv', [
    ['--target', '3:10', '--workers', '0'],
    ['--target', '3:10', '--hours', '0'],
    ['--target', '3:10,3:20'],
])
def test_parser_rejects_bad_args(argv: list[str]):
    with pytest.raises(SystemExit):
        farm.main(argv)


def test_missing_binary_message(monkeypatch, tmp_path: Path):
    def missing_binary() -> Path:
        raise FileNotFoundError('Pikafish executable not found')

    monkeypatch.setattr(farm, 'find_engine_binary', missing_binary)
    with pytest.raises(SystemExit) as info:
        farm.main([
            '--target', '1:1',
            '--out-root', str(tmp_path / 'farm'),
            '--log-dir', str(tmp_path / 'logs'),
        ])
    assert 'Pikafish' in str(info.value)


@pytest.mark.skipif(
    not _pikafish_available(), reason='bundled Pikafish is unavailable')
def test_smoke_two_shards(tmp_path: Path):
    out_root = tmp_path / 'farm'
    log_dir = tmp_path / 'logs'
    argv = [
        '--target', '1:2', '--workers', '2', '--games-per-shard', '1',
        '--out-root', str(out_root), '--log-dir', str(log_dir),
        '--max-game-plies', '12', '--opening-plies', '2',
    ]

    assert farm.main(argv) == 0
    depth_one = out_root / 'depth1'
    names = sorted(path.name for path in depth_one.glob('*.npz'))
    assert names == ['d1_g10000000_n1.npz', 'd1_g10000001_n1.npz']
    with np.load(depth_one / names[0], allow_pickle=False) as archive:
        assert int(archive['engine_depth']) == 1

    assert farm.main(argv) == 0
    assert sorted(path.name for path in depth_one.glob('*.npz')) == names
