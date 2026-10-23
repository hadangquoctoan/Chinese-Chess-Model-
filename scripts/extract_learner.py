"""Extract learner weights from a resumable training checkpoint."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.train import build_model_config, load_config
from src.model.neural_network import ChineseChessNet
from src.utils.model_io import atomic_torch_save, build_model_artifact, safe_torch_load


def extract_learner(checkpoint_path: str | Path, config_path: str | Path,
                    output_path: str | Path) -> Path:
    """Write the checkpoint's learner weights as a versioned model artifact."""
    checkpoint = safe_torch_load(checkpoint_path)
    state = checkpoint.get('model_state_dict') if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict):
        raise ValueError('Checkpoint lacks model_state_dict')
    model_config = build_model_config(load_config(str(config_path)))
    model = ChineseChessNet(model_config)
    model.load_state_dict(state)
    output = Path(output_path)
    atomic_torch_save(build_model_artifact(model), output)
    return output


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Extract learner weights from a checkpoint')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--config', default='configs/train_server_selfplay_v2.yaml')
    parser.add_argument('--output', default='models/learner.pt')
    args = parser.parse_args(argv)
    output = extract_learner(args.checkpoint, args.config, args.output)
    print(f'Wrote {output}')


if __name__ == '__main__':
    main()
