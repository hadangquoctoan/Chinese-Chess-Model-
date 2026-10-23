"""Supervised pretrain on Pikafish teacher labels. Does not write best_model.pt."""
import argparse
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_teacher_labels import load_teacher_npz
from scripts.train import build_model_config, load_config
from src.model.neural_network import ChineseChessNet
from src.training.replay_buffer import ReplayBuffer
from src.training.trainer import Trainer
from src.utils.model_io import ARCHITECTURE_FIELDS, load_model_artifact

DEFAULT_OUTPUT = Path("models") / "pretrain_depth2.pt"
TURN_CHANNEL = 16


def check_init_architecture(loaded, model_config) -> None:
    """Reject a learner artifact whose tensor contract differs from this config."""
    mismatch = [name for name in ARCHITECTURE_FIELDS
                if getattr(loaded.config, name, None) != getattr(model_config, name, None)]
    if mismatch:
        raise ValueError(f'init-model architecture mismatch: {", ".join(mismatch)}')


def side_to_move_value(state, red_centric: float) -> float:
    """Map a red-centric teacher value onto the side-to-move tensor."""
    return float(red_centric) if float(state[TURN_CHANNEL, 0, 0]) >= 0.0 else -float(red_centric)


def buffer_from_npz(path: Path) -> ReplayBuffer:
    payload = load_teacher_npz(path)
    states, policies, values = payload["states"], payload["policies"], payload["values"]
    if states.shape[0] != policies.shape[0] or states.shape[0] != values.shape[0]:
        raise ValueError("Teacher arrays disagree on position count")
    replay = ReplayBuffer(max(1, len(values)))
    for state, policy, value in zip(states, policies, values):
        replay.add(state, policy, side_to_move_value(state, float(value)))
    return replay


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Supervised pretrain from teacher NPZ")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--config", default="configs/train_depth2_16h.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--init-model", type=Path, default=None,
                        help="Optional learner artifact to fine-tune")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Optional learning-rate override")
    parser.add_argument("--log-dir", default="logs/pretrain")
    parser.add_argument("--checkpoint-dir", default="models/pretrain_checkpoints")
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.output.resolve() == Path("models/best_model.pt").resolve():
        parser.error("Refusing to overwrite models/best_model.pt")
    config = load_config(args.config)
    model_config = build_model_config(config)
    if args.learning_rate is not None:
        if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
            parser.error("--learning-rate must be finite and positive")
        model_config.learning_rate = args.learning_rate
    model = ChineseChessNet(model_config)
    if args.init_model is not None:
        loaded = load_model_artifact(args.init_model)
        check_init_architecture(loaded, model_config)
        model.load_state_dict(loaded.state_dict())
    replay = buffer_from_npz(args.data)
    trainer = Trainer(
        model=model, config=model_config,
        checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir,
        num_workers=config.get("optimization", {}).get("num_workers", 0))
    try:
        for _ in range(args.epochs):
            policy_loss, value_loss, total = trainer.train_epoch(replay, len(replay))
            print(f"epoch {trainer.epoch} policy={policy_loss:.4f} "
                  f"value={value_loss:.4f} total={total:.4f}")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        trainer.save_model(str(args.output))
        print(f"Wrote {args.output}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
