"""8-hour laptop overnight: labels, pretrain, sparring, 6h snapshot, stop at 8h.

Sized for Quadro RTX 3000 6GB / 32GB RAM / i7-9850H. Does not write the
rented-server paths under models/best_model.pt.
"""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

HOUR = 3600
DEFAULT_HOURS = 8
SNAPSHOT_HOURS = 6
FINAL_MEASURE_SECONDS = 20 * 60
DEFAULT_CONFIG = "configs/train_laptop_8h.yaml"
TEACHER_NPZ = Path("data") / "teacher_labels" / "laptop_depth2.npz"
PRETRAIN_PT = Path("models") / "laptop_8h" / "pretrain_depth2.pt"
BEST_PT = Path("models") / "laptop_8h" / "best_model.pt"
SNAPSHOT_PT = Path("models") / "laptop_8h" / "snapshot_6h.pt"
PROGRESS_PT = Path("models") / "laptop_8h" / "checkpoints" / "checkpoint_progress.pt"


USER_STOP = 130


def _run(args: list[str], *, allow_stop: bool = False) -> str:
    print("+", " ".join(args), flush=True)
    completed = subprocess.run(args)
    if completed.returncode == USER_STOP:
        if allow_stop:
            return "stopped"
        raise SystemExit(USER_STOP)
    if completed.returncode != 0:
        raise SystemExit(completed.returncode)
    return "ok"


def _remaining(deadline: float) -> float:
    return deadline - time.monotonic()


def _copy_snapshot(source: Path, dest: Path) -> None:
    if not source.is_file():
        raise SystemExit(f"No playable weights to snapshot: {source}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, dest)
    print(f"Wrote {dest} from {source}", flush=True)


def _require_cuda() -> None:
    import torch
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for the laptop overnight run; refusing CPU.")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / 1024 ** 3
    print(f"gpu: {props.name} | vram: {vram_gb:.1f}GB", flush=True)
    if vram_gb < 5.0:
        raise SystemExit(f"Need at least 6GB-class GPU, found {vram_gb:.1f}GB")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="8-hour laptop Pikafish sparring run")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--hours", type=float, default=DEFAULT_HOURS)
    parser.add_argument("--num-positions", type=int, default=1500)
    parser.add_argument("--pretrain-epochs", type=int, default=2)
    parser.add_argument("--mcts-sims", type=int, default=64)
    parser.add_argument("--measure-games", type=int, default=8)
    args = parser.parse_args(argv)
    if args.hours <= 0:
        parser.error("--hours must be positive")
    python = sys.executable
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    _require_cuda()

    start = time.monotonic()
    deadline_snap = start + SNAPSHOT_HOURS * HOUR * (args.hours / DEFAULT_HOURS)
    deadline_end = start + args.hours * HOUR
    print(f"laptop run {args.hours:.1f}h; snapshot at {SNAPSHOT_HOURS * args.hours / DEFAULT_HOURS:.1f}h",
          flush=True)

    _run([python, str(root / "scripts" / "build_teacher_labels.py"),
          "--output", str(TEACHER_NPZ), "--num-positions", str(args.num_positions),
          "--depth", "2", "--threads", "4", "--hash-mb", "256"])
    _run([python, str(root / "scripts" / "pretrain_supervised.py"),
          "--data", str(TEACHER_NPZ), "--config", args.config,
          "--output", str(PRETRAIN_PT), "--epochs", str(args.pretrain_epochs),
          "--log-dir", "logs/laptop_8h_pretrain",
          "--checkpoint-dir", "models/laptop_8h/pretrain_checkpoints"])

    first_budget = max(0.0, _remaining(deadline_snap))
    first_status = "ok"
    if first_budget > 0:
        first_status = _run([python, str(root / "scripts" / "train.py"),
                             "--config", args.config, "--init-model", str(PRETRAIN_PT),
                             "--max-seconds", str(first_budget)], allow_stop=True)
    source = BEST_PT if BEST_PT.is_file() else PRETRAIN_PT
    _copy_snapshot(source, SNAPSHOT_PT)
    if first_status == "stopped":
        print("Stopped after the first train window; skipping remaining hours.", flush=True)
        print(f"play: {SNAPSHOT_PT}", flush=True)
        return

    measure = [python, str(root / "scripts" / "measure_vs_pikafish.py"),
               "--model", str(SNAPSHOT_PT), "--games", str(args.measure_games),
               "--mcts-sims", str(args.mcts_sims), "--threads", "4", "--hash-mb", "256"]
    if _remaining(deadline_end) > 0:
        _run(measure + ["--depth", "1"])

    train_until = deadline_end - FINAL_MEASURE_SECONDS
    second_budget = max(0.0, train_until - time.monotonic())
    if second_budget > 0 and PROGRESS_PT.is_file():
        _run([python, str(root / "scripts" / "train.py"), "--config", args.config,
              "--resume", str(PROGRESS_PT), "--max-seconds", str(second_budget)],
             allow_stop=True)
    if BEST_PT.is_file() and _remaining(deadline_end) > 0:
        _run([python, str(root / "scripts" / "measure_vs_pikafish.py"),
              "--model", str(BEST_PT), "--games", str(args.measure_games),
              "--depth", "1", "--mcts-sims", str(args.mcts_sims),
              "--threads", "4", "--hash-mb", "256"])
        if _remaining(deadline_end) > 8 * 60:
            _run([python, str(root / "scripts" / "measure_vs_pikafish.py"),
                  "--model", str(BEST_PT), "--games", str(args.measure_games),
                  "--depth", "2", "--mcts-sims", str(args.mcts_sims),
                  "--threads", "4", "--hash-mb", "256"])
    print("laptop 8h orchestration finished", flush=True)
    print(f"play: {SNAPSHOT_PT} (6h) and {BEST_PT} (final)", flush=True)


if __name__ == "__main__":
    main()
