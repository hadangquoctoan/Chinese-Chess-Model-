"""One-command server training: optional labels/pretrain, then a timed train run."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LABELS = Path("data") / "teacher_labels" / "teacher_depth2.npz"
PRETRAIN = Path("models") / "pretrain_depth2.pt"
WINDOWS_CONFIG = "configs/train_server_fast.yaml"
LINUX_CONFIG = "configs/train_server_fast_linux.yaml"
TORCH_PROBE = """
import sys
import torch
version = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])
ready = bool(torch.version.cuda) and torch.cuda.is_available() and version >= (2, 6)
sys.exit(0 if ready else 1)
"""


def default_config() -> str:
    """Pick the fast server config for the current operating system."""
    return WINDOWS_CONFIG if os.name == "nt" else LINUX_CONFIG


def cuda_ready() -> bool:
    """Return whether the current interpreter has CUDA torch >= 2.6."""
    return subprocess.run(
        [sys.executable, "-c", TORCH_PROBE],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def run(command: list[str], dry_run: bool) -> None:
    """Print and optionally execute one child command."""
    print("+", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Set up labels/pretrain, then train on a rented server.")
    parser.add_argument("--config", default=None, help="YAML config (default: platform fast config)")
    parser.add_argument("--hours", type=float, default=8.0,
                        help="Timed training budget; labels/pretrain are additional")
    parser.add_argument("--init-model", default=None,
                        help="Weights to seed a fresh run, e.g. models/best_model.pt")
    parser.add_argument("--resume", default=None,
                        help="Full training_session checkpoint to resume")
    parser.add_argument("--num-positions", type=int, default=4000,
                        help="Teacher positions when labels are generated")
    parser.add_argument("--depth", type=int, default=2,
                        help="Pikafish teacher depth when labels are generated")
    parser.add_argument("--pretrain-epochs", type=int, default=3)
    parser.add_argument("--teacher-threads", type=int, default=8)
    parser.add_argument("--teacher-hash-mb", type=int, default=128)
    parser.add_argument("--skip-labels", action="store_true")
    parser.add_argument("--skip-pretrain", action="store_true")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print commands without executing them")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.chdir(ROOT)
    if args.hours <= 0:
        raise SystemExit("--hours must be positive")
    if args.init_model and args.resume:
        raise SystemExit("--init-model and --resume are mutually exclusive")

    config = args.config or default_config()
    if not (ROOT / config).is_file():
        raise SystemExit(f"Config not found: {config}")
    if not args.dry_run and not cuda_ready():
        raise SystemExit(
            "CUDA torch >= 2.6 is not available in this interpreter. "
            "Run the venv Python from scripts/setup_venv.py.")

    init_model = args.init_model
    resume = args.resume
    if init_model and not (ROOT / init_model).is_file():
        raise SystemExit(f"Initial model not found: {init_model}")
    if resume and not (ROOT / resume).is_file():
        raise SystemExit(f"Resume checkpoint not found: {resume}")

    if init_model is None and resume is None:
        if (ROOT / PRETRAIN).is_file():
            init_model = str(PRETRAIN)
        else:
            if not (ROOT / LABELS).is_file():
                if args.skip_labels:
                    raise SystemExit(f"--skip-labels was set but labels are missing: {LABELS}")
                run([sys.executable, "scripts/build_teacher_labels.py",
                     "--output", str(LABELS), "--num-positions", str(args.num_positions),
                     "--depth", str(args.depth), "--threads", str(args.teacher_threads),
                     "--hash-mb", str(args.teacher_hash_mb)], args.dry_run)
            if args.skip_pretrain:
                raise SystemExit(
                    f"--skip-pretrain was set but the initial model is missing: {PRETRAIN}")
            run([sys.executable, "scripts/pretrain_supervised.py",
                 "--data", str(LABELS), "--config", config,
                 "--output", str(PRETRAIN), "--epochs", str(args.pretrain_epochs)], args.dry_run)
            init_model = str(PRETRAIN)

    command = [sys.executable, "scripts/train.py", "--config", config,
               "--max-seconds", str(int(args.hours * 3600))]
    if init_model is not None:
        command += ["--init-model", str(init_model)]
    if resume is not None:
        command += ["--resume", str(resume)]
    run(command, args.dry_run)


if __name__ == "__main__":
    main()
