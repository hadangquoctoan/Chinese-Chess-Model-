"""Create a local venv with CUDA PyTorch and the project requirements."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_VENV = ROOT / ".venv"
DEFAULT_CUDA_INDEX = "https://download.pytorch.org/whl/cu126"
MIN_PYTHON = (3, 10)
TORCH_PROBE = """
import sys
import torch
version = tuple(int(part) for part in torch.__version__.split('+')[0].split('.')[:2])
ready = bool(torch.version.cuda) and torch.cuda.is_available() and version >= (2, 6)
sys.exit(0 if ready else 1)
"""


def venv_python(venv: Path) -> Path:
    """Return the interpreter path inside a venv for this platform."""
    return venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def torch_ready(python: Path) -> bool:
    """Return whether the venv interpreter has CUDA torch >= 2.6."""
    return subprocess.run(
        [str(python), "-c", TORCH_PROBE],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ).returncode == 0


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Create the project venv with CUDA PyTorch.")
    parser.add_argument("--venv", type=Path, default=DEFAULT_VENV)
    parser.add_argument("--cuda-index", default=DEFAULT_CUDA_INDEX,
                        help="PyTorch wheel index (cu126, cu124, cu118, ...)")
    parser.add_argument("--recreate", action="store_true",
                        help="Delete an existing venv before creating it")
    args = parser.parse_args(argv)

    if sys.version_info < MIN_PYTHON:
        raise SystemExit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required; found {sys.version.split()[0]}")
    venv = args.venv.resolve()
    if args.recreate and venv.exists():
        shutil.rmtree(venv)
    try:
        subprocess.run([sys.executable, "-m", "venv", str(venv)], check=True)
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            "Could not create the venv. On Ubuntu install python3-venv first: "
            "sudo apt install python3-venv") from error

    python = venv_python(venv)
    if not python.is_file():
        raise SystemExit(f"venv interpreter not found: {python}")
    subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"], check=True)
    if not torch_ready(python):
        print(f"Installing CUDA PyTorch from {args.cuda_index}", flush=True)
        subprocess.run(
            [str(python), "-m", "pip", "install", "--force-reinstall", "torch",
             "--index-url", args.cuda_index],
            check=True,
        )
    subprocess.run(
        [str(python), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")],
        check=True,
    )
    if not torch_ready(python):
        raise SystemExit(
            "CUDA torch >= 2.6 is still unavailable. Check the NVIDIA driver, or rerun "
            f"with --cuda-index (current: {args.cuda_index}).")

    linux_engine = ROOT / "pikafish" / "Pikafish-Linux-x86-64-universal"
    if os.name != "nt" and linux_engine.is_file():
        linux_engine.chmod(linux_engine.stat().st_mode | 0o111)

    runner = ".venv\\Scripts\\python.exe" if os.name == "nt" else ".venv/bin/python"
    print("\nVenv ready. Next command:")
    print(f"  {runner} scripts/run_server_train.py --hours 7.5 --init-model models/best_model.pt")
    print("If you do not have a model on this server, omit --init-model; "
          "the runner will create teacher labels and pretrain automatically.")


if __name__ == "__main__":
    main()
