"""Orchestrate labels, pretrain, 9h snapshot, then train until hour 15."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HOUR = 3600
SNAPSHOT_HOURS = 9
HARD_LIMIT_HOURS = 15
FINAL_MEASURE_SECONDS = 30 * 60
DEFAULT_CONFIG = "configs/train_depth2_16h.yaml"
TEACHER_NPZ = Path("data") / "teacher_labels" / "teacher_depth2.npz"
PRETRAIN_PT = Path("models") / "pretrain_depth2.pt"
BEST_PT = Path("models") / "best_model.pt"
SNAPSHOT_PT = Path("models") / "snapshot_9h.pt"
PROGRESS_PT = Path("models") / "checkpoints" / "checkpoint_progress.pt"


def _run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    completed = subprocess.run(args)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _copy_snapshot(source: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"No playable weights to snapshot: {source}")
    SNAPSHOT_PT.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, SNAPSHOT_PT)
    print(f"Wrote {SNAPSHOT_PT} from {source}", flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="16-hour Pikafish sparring run")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--num-positions", type=int, default=4000)
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--mcts-sims", type=int, default=128)
    parser.add_argument("--measure-games", type=int, default=20)
    args = parser.parse_args(argv)
    python = sys.executable
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    start = time.monotonic()
    deadline_9h = start + SNAPSHOT_HOURS * HOUR
    deadline_end = start + HARD_LIMIT_HOURS * HOUR

    _run([python, str(root / "scripts" / "build_teacher_labels.py"),
          "--output", str(TEACHER_NPZ), "--num-positions", str(args.num_positions),
          "--depth", "2"])
    _run([python, str(root / "scripts" / "pretrain_supervised.py"),
          "--data", str(TEACHER_NPZ), "--config", args.config,
          "--output", str(PRETRAIN_PT), "--epochs", str(args.pretrain_epochs)])

    first_budget = max(0.0, _remaining(deadline_9h))
    if first_budget > 0:
        train_cmd = [python, str(root / "scripts" / "train.py"),
                     "--config", args.config, "--init-model", str(PRETRAIN_PT),
                     "--max-seconds", str(first_budget)]
        _run(train_cmd)
    source = BEST_PT if BEST_PT.is_file() else PRETRAIN_PT
    _copy_snapshot(source)

    measure = [python, str(root / "scripts" / "measure_vs_pikafish.py"),
               "--model", str(SNAPSHOT_PT), "--games", str(args.measure_games),
               "--mcts-sims", str(args.mcts_sims)]
    if _remaining(deadline_end) > 0:
        _run(measure + ["--depth", "1"])
    if _remaining(deadline_9h) > 15 * 60 and _remaining(deadline_end) > 0:
        _run(measure + ["--depth", "2"])

    train_until = deadline_end - FINAL_MEASURE_SECONDS
    second_budget = max(0.0, train_until - time.monotonic())
    if second_budget > 0 and PROGRESS_PT.is_file():
        _run([python, str(root / "scripts" / "train.py"), "--config", args.config,
              "--resume", str(PROGRESS_PT), "--max-seconds", str(second_budget)])
    elif second_budget > 0 and not PROGRESS_PT.is_file() and PRETRAIN_PT.is_file():
        _run([python, str(root / "scripts" / "train.py"), "--config", args.config,
              "--init-model", str(PRETRAIN_PT), "--max-seconds", str(second_budget)])

    if BEST_PT.is_file():
        final_model = str(BEST_PT)
        _run([python, str(root / "scripts" / "measure_vs_pikafish.py"),
              "--model", final_model, "--games", str(args.measure_games),
              "--depth", "1", "--mcts-sims", str(args.mcts_sims)])
        _run([python, str(root / "scripts" / "measure_vs_pikafish.py"),
              "--model", final_model, "--games", str(args.measure_games),
              "--depth", "2", "--mcts-sims", str(args.mcts_sims)])
    print(f"{HARD_LIMIT_HOURS}h orchestration finished", flush=True)


if __name__ == "__main__":
    main()
