"""Build untrained checkpoints for all presets (no training)."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import torch

from src.game.game_state import GameState
from src.model.mcts import MCTS
from src.model.model_config import ModelConfig
from src.model.neural_network import ChineseChessNet
from src.utils.model_io import build_model_artifact, load_model_artifact


def build_one(size: str, out_dir: Path) -> Path:
    out_path = out_dir / f"untrained_{size}.pt"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing model: {out_path}")
    cfg = ModelConfig.from_size(size)
    model = ChineseChessNet(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("xb") as model_file:
        torch.save(build_model_artifact(model), model_file)
    # round-trip check includes architecture and encoding metadata.
    model2 = load_model_artifact(out_path, device="cpu")
    sd = model2.state_dict()
    model2.eval()
    with torch.no_grad():
        x = torch.randn(1, 19, 10, 9)
        pol, val = model2(x)
        assert pol.shape == (1, 1800)
        assert val.shape == (1, 1)
        assert -1.0 <= float(val.item()) <= 1.0
    params = model.count_parameters()
    numel = sum(v.numel() for v in sd.values())
    print(f"{size:6s} params={params:,} state_numel={numel:,} file={out_path.stat().st_size/1024:.1f}KB")
    return out_path


def main() -> None:
    out_dir = REPO_ROOT / "models"
    for size in ("small", "medium", "large"):
        build_one(size, out_dir)
    # one-pipeline demo
    print("--- MCTS demo on small untrained (16 sims) ---")
    cfg = ModelConfig.from_size("small")
    m = ChineseChessNet(cfg)
    m.eval()
    gs = GameState()
    mcts = MCTS(m, num_simulations=16, temperature=0.0, dirichlet_epsilon=0.0)
    move, policy = mcts.search(gs)
    print(f"move={move} legal={move in gs.get_legal_moves()} policy_sum={float(policy.sum()):.4f}")
    print("--- models/*.pt ---")
    for p in sorted(out_dir.glob("*.pt")):
        print(f"  {p.name} {p.stat().st_size/1024:.1f}KB")


if __name__ == "__main__":
    main()
