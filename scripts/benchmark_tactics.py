"""Measure how often MCTS chooses a capture of a >=4 point piece."""
import argparse
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.build_teacher_labels import load_probe_npz
from src.game.game_state import GameState
from src.model.mcts import MCTS
from src.training.adjudication import MATERIAL_VALUES
from src.utils.model_io import load_model_artifact


def replay_history(history) -> GameState:
    """Rebuild the probe position by replaying its project-coordinate moves."""
    game = GameState()
    for move in history:
        if not game.make_move(*move):
            raise ValueError(f'Invalid probe history move: {move}')
    return game


def capture_value(game: GameState, move) -> float:
    _, _, to_row, to_col = move
    target = game.board.get_piece(to_row, to_col)
    return MATERIAL_VALUES[target.piece_type]


def evaluate_probes(model, probes, sims: int) -> tuple[float, int, int]:
    mcts = MCTS(model, num_simulations=sims, temperature=0.0,
                dirichlet_epsilon=0.0)
    correct = 0
    total = len(probes['states'])
    for index in range(total):
        game = replay_history(probes['history_groups'][index])
        move, _ = mcts.get_action_probs(game, temperature=0.0)
        if move is not None and capture_value(game, move) >= 4.0:
            correct += 1
    return (correct / total if total else 0.0), correct, total


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Benchmark material capture choices')
    parser.add_argument('--model', required=True)
    parser.add_argument('--probes', required=True)
    parser.add_argument('--sims', type=int, default=96)
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    model = load_model_artifact(args.model, device=args.device)
    probes = load_probe_npz(Path(args.probes))
    accuracy, correct, total = evaluate_probes(model, probes, args.sims)
    print(f'capture accuracy {accuracy:.1%} ({correct}/{total}) at {args.sims} sims')
    if accuracy < args.threshold:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
