"""Supervised pretrain on Pikafish teacher labels. Does not write best_model.pt."""
import argparse
import math
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_teacher_labels import load_teacher_npz
from scripts.train import build_model_config, load_config
from src.model.neural_network import ChineseChessNet
from src.training.replay_buffer import ReplayBuffer
from src.training.supervised import fit_supervised
from src.training.trainer import Trainer
from src.utils.model_io import ARCHITECTURE_FIELDS, load_model_artifact

DEFAULT_OUTPUT = Path("models") / "pretrain_depth2.pt"
TURN_CHANNEL = 16
VALUE_CONVENTION = "red_centric"
CONTRACT_METADATA = ("action_encoding_version", "ruleset_version")
STATE_LIMIT_METADATA = ("max_game_plies", "no_capture_limit")


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


def _load_contract_archive(path: Path, label: str) -> dict:
    """Load one archive and fail clearly when the file is missing."""
    if not Path(path).is_file():
        raise ValueError(f"{label} dataset does not exist: {path}")
    return load_teacher_npz(path)


def validate_validation_contract(train_path: Path, validation_path: Path,
                                 model_config) -> None:
    """Validate train/validation NPZ metadata before an epoch is trained.

    Validation enables early stopping, so both archives must share the same
    encoding and value convention, carry disjoint per-game IDs, and declare
    the state limits used by ``model_config``.
    """
    if Path(train_path).resolve() == Path(validation_path).resolve():
        raise ValueError("--data and --val-data must resolve to different files")
    train_payload = _load_contract_archive(train_path, "train")
    validation_payload = _load_contract_archive(validation_path, "validation")
    for field in CONTRACT_METADATA:
        train_value = train_payload.get(field)
        validation_value = validation_payload.get(field)
        if train_value != validation_value:
            raise ValueError(
                f"train and validation {field} must match: "
                f"{train_value!r} != {validation_value!r}")
    game_ids_by_split: dict[str, set[int]] = {}
    for label, payload in (("train", train_payload),
                           ("validation", validation_payload)):
        if payload.get("value_convention") != VALUE_CONVENTION:
            raise ValueError(
                f"{label} value_convention must be {VALUE_CONVENTION!r}, "
                f"got {payload.get('value_convention')!r}")
        if payload.get("game_ids") is None:
            raise ValueError(f"{label} data is missing game_ids")
        state_count = len(payload["states"])
        game_ids = np.asarray(payload["game_ids"])
        if game_ids.ndim != 1 or game_ids.size != state_count:
            raise ValueError(
                f"{label} game_ids must have one entry per state "
                f"({game_ids.size} != {state_count})")
        game_ids_by_split[label] = {int(game_id) for game_id in game_ids}
        for field in STATE_LIMIT_METADATA:
            if field not in payload:
                raise ValueError(f"{label} data is missing {field}")
            declared = int(payload[field])
            expected = int(getattr(model_config, field))
            if declared != expected:
                raise ValueError(
                    f"{label} {field}={declared} does not match model config "
                    f"{expected}")
    overlap = game_ids_by_split["train"] & game_ids_by_split["validation"]
    if overlap:
        raise ValueError(
            "train and validation game_ids must be disjoint; "
            f"overlap: {sorted(overlap)}")


def _format_history_record(record: dict) -> str:
    """Format one supervised epoch, including validation metrics when present."""
    policy_loss, value_loss, total_loss = record['train_losses']
    message = (f"epoch {record['epoch']} policy={policy_loss:.4f} "
               f"value={value_loss:.4f} total={total_loss:.4f}")
    metrics = record.get('validation')
    if metrics is not None:
        message += (f" val_policy={metrics['policy_loss']:.4f}"
                    f" val_value={metrics['value_loss']:.4f}"
                    f" val_total={metrics['total_loss']:.4f}"
                    f" val_top1={metrics['top1_accuracy']:.4f}")
    return message


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Parse and validate supervised pretraining options."""
    parser = argparse.ArgumentParser(description="Supervised pretrain from teacher NPZ")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--val-data", type=Path, default=None)
    parser.add_argument("--config", default="configs/supervised.yaml")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--init-model", type=Path, default=None,
                        help="Optional learner artifact to fine-tune")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Optional learning-rate override")
    parser.add_argument("--log-dir", default="logs/pretrain")
    parser.add_argument("--checkpoint-dir", default="models/pretrain_checkpoints")
    args = parser.parse_args(argv)
    if args.epochs <= 0:
        parser.error("--epochs must be positive")
    if args.patience <= 0:
        parser.error("--patience must be positive")
    if args.output.resolve() == Path("models/best_model.pt").resolve():
        parser.error("Refusing to overwrite models/best_model.pt")
    if args.learning_rate is not None and (
            not math.isfinite(args.learning_rate) or args.learning_rate <= 0):
        parser.error("--learning-rate must be finite and positive")
    return args


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config)
    model_config = build_model_config(config)
    if args.val_data is not None:
        validate_validation_contract(args.data, args.val_data, model_config)
    if args.learning_rate is not None:
        model_config.learning_rate = args.learning_rate
    model = ChineseChessNet(model_config)
    if args.init_model is not None:
        loaded = load_model_artifact(args.init_model)
        check_init_architecture(loaded, model_config)
        model.load_state_dict(loaded.state_dict())
    replay = buffer_from_npz(args.data)
    validation_replay = buffer_from_npz(args.val_data) if args.val_data is not None else None
    trainer = Trainer(
        model=model, config=model_config,
        checkpoint_dir=args.checkpoint_dir, log_dir=args.log_dir,
        num_workers=config.get("optimization", {}).get("num_workers", 0))
    try:
        history = fit_supervised(
            trainer, replay, validation_replay, args.output,
            epochs=args.epochs, patience=args.patience)
        for record in history:
            print(_format_history_record(record))
        if not args.output.is_file():
            raise RuntimeError(
                f"No supervised artifact was written to {args.output}; "
                "the run produced no saveable model (validation may be non-finite)")
        print(f"Wrote {args.output}")
    finally:
        trainer.close()


if __name__ == "__main__":
    main()
