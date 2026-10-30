"""Supervised pipeline: Pikafish dataset, pretrain, then measurement.

The full profile generates depth-8 teacher labels and can run for hours;
``--smoke`` exercises every stage with tiny budgets and separate artifacts.
ReplayBuffer is only the supervised data container here: no model self-play,
MCTS data generation, or AlphaZero training is involved.
"""
import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

SMOKE_CONFIG_PATH = Path('configs') / 'supervised_smoke.yaml'
FULL_CONFIG_PATH = Path('configs') / 'supervised.yaml'
DATASET_DEPTH = 8
DATASET_OPENING_PLIES = 8
DATASET_THREADS = 4
DATASET_HASH_MB = 256
PRETRAIN_PATIENCE = 2
MCTS_SIMULATIONS = 64
MEASUREMENT_OPENING_PLIES = 4
MEASUREMENT_OPENING_SEED = 0
MEASUREMENT_THREADS = 4
MEASUREMENT_HASH_MB = 256
MEASUREMENT_MODES = ('raw', 'mcts')


@dataclass(frozen=True)
class RunProfile:
    """Budgets and artifact stem for one smoke or full supervised run."""

    games: int
    validation_games: int
    max_game_plies: int
    epochs: int
    evaluation_games: int
    measurement_depths: tuple[int, ...]
    stem: str
    dataset_prefix: str
    config_path: Path


SMOKE_PROFILE = RunProfile(
    games=3, validation_games=1, max_game_plies=24, epochs=1,
    evaluation_games=2, measurement_depths=(1,), stem='supervised_smoke',
    dataset_prefix='supervised_smoke_', config_path=SMOKE_CONFIG_PATH)
FULL_PROFILE = RunProfile(
    games=110, validation_games=10, max_game_plies=160, epochs=12,
    evaluation_games=8, measurement_depths=(1, 2), stem='supervised',
    dataset_prefix='', config_path=FULL_CONFIG_PATH)


def _artifact_paths(root: Path, profile: RunProfile) -> dict[str, Path]:
    """Return per-profile artifact paths, isolated by the profile stem."""
    labels = root / 'data' / 'teacher_labels'
    return {
        'train': labels / f'{profile.dataset_prefix}train.npz',
        'validation': labels / f'{profile.dataset_prefix}validation.npz',
        'model': root / 'models' / profile.stem / 'best_model.pt',
        'log_dir': root / 'logs' / profile.stem / 'pretrain',
        'checkpoint_dir': root / 'models' / profile.stem / 'pretrain_checkpoints',
    }


def _dataset_command(python: str, root: Path, profile: RunProfile,
                     paths: dict[str, Path]) -> list[str]:
    return [
        python, str(root / 'scripts' / 'build_pikafish_dataset.py'),
        '--train-output', str(paths['train']),
        '--validation-output', str(paths['validation']),
        '--games', str(profile.games),
        '--validation-games', str(profile.validation_games),
        '--depth', str(DATASET_DEPTH),
        '--opening-plies', str(DATASET_OPENING_PLIES),
        '--max-game-plies', str(profile.max_game_plies),
        '--threads', str(DATASET_THREADS),
        '--hash-mb', str(DATASET_HASH_MB),
    ]


def _pretrain_command(python: str, root: Path, profile: RunProfile,
                      paths: dict[str, Path]) -> list[str]:
    return [
        python, str(root / 'scripts' / 'pretrain_supervised.py'),
        '--data', str(paths['train']),
        '--val-data', str(paths['validation']),
        '--config', str(root / profile.config_path),
        '--output', str(paths['model']),
        '--epochs', str(profile.epochs),
        '--patience', str(PRETRAIN_PATIENCE),
        '--log-dir', str(paths['log_dir']),
        '--checkpoint-dir', str(paths['checkpoint_dir']),
    ]


def _measurement_command(python: str, root: Path, profile: RunProfile,
                         paths: dict[str, Path], mode: str,
                         depth: int) -> list[str]:
    command = [
        python, str(root / 'scripts' / 'measure_vs_pikafish.py'),
        '--model', str(paths['model']),
        '--mode', mode,
        '--games', str(profile.evaluation_games),
        '--depth', str(depth),
        '--max-game-plies', str(profile.max_game_plies),
        '--opening-plies', str(MEASUREMENT_OPENING_PLIES),
        '--opening-seed', str(MEASUREMENT_OPENING_SEED),
        '--threads', str(MEASUREMENT_THREADS),
        '--hash-mb', str(MEASUREMENT_HASH_MB),
    ]
    if mode == 'mcts':
        command += ['--mcts-sims', str(MCTS_SIMULATIONS)]
    return command


def supervised_steps(python: str, root: Path,
                     smoke: bool) -> list[list[str]]:
    """Return supervised commands in execution order for ``python`` at ``root``."""
    profile = SMOKE_PROFILE if smoke else FULL_PROFILE
    paths = _artifact_paths(root, profile)
    steps = [
        _dataset_command(python, root, profile, paths),
        _pretrain_command(python, root, profile, paths),
    ]
    for mode in MEASUREMENT_MODES:
        for depth in profile.measurement_depths:
            steps.append(_measurement_command(
                python, root, profile, paths, mode, depth))
    return steps


def main(argv: list[str] | None = None) -> None:
    """Run the selected profile, stopping at the first failing stage."""
    parser = argparse.ArgumentParser(
        description='Supervised pipeline: dataset, pretrain, Pikafish match')
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny budgets and separate supervised_smoke artifacts')
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    for command in supervised_steps(sys.executable, root, args.smoke):
        print('+ ' + ' '.join(command), flush=True)
        subprocess.run(command, check=True)


if __name__ == '__main__':
    main()
