"""Play a model artifact against Pikafish with paired colors."""
import argparse
import math
import random
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.player import load_pikafish_player
from src.game.game_state import GameState
from src.training.sparring_game import SparringWorker
from src.utils.model_io import load_model_artifact


def generate_opening(plies: int, rng: random.Random) -> tuple:
    """Generate a reproducible random legal opening."""
    game = GameState()
    moves = []
    for _ in range(plies):
        if game.is_terminal():
            break
        legal = game.get_legal_moves()
        move = legal[rng.randrange(len(legal))]
        game.make_move(*move)
        moves.append(move)
    return tuple(moves)


def summarize_results(results) -> dict:
    games = len(results)
    wins = sum(1 for winner, _ in results if winner > 0)
    losses = sum(1 for winner, _ in results if winner < 0)
    draws = games - wins - losses
    plies = sum(plies for _, plies in results)
    return {'games': games, 'wins': wins, 'losses': losses, 'draws': draws,
            'score': (wins + 0.5 * draws) / games if games else 0.0,
            'average_plies': plies / games if games else 0.0}


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    if total == 0:
        return 0.0, 1.0
    p = successes / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    margin = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description='Benchmark a model against Pikafish')
    parser.add_argument('--model', required=True)
    parser.add_argument('--games', type=int, default=10)
    parser.add_argument('--sims', type=int, default=64)
    parser.add_argument('--depth', type=int, default=1)
    parser.add_argument('--opening-plies', type=int, default=4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 2:
        parser.error('--games must be a positive even integer')
    model = load_model_artifact(args.model, device=args.device)
    opponent = load_pikafish_player(depth=args.depth)
    results = []
    try:
        for index in range(args.games):
            opening = generate_opening(
                args.opening_plies, random.Random(args.seed + index // 2))
            model_is_red = index % 2 == 0
            worker = SparringWorker(
                model=model, opponent=opponent, opponent_is_red=not model_is_red,
                mcts_simulations=args.sims, temperature=0.0,
                temperature_threshold=1000, dirichlet_alpha=0.3,
                dirichlet_epsilon=0.0, max_game_plies=120,
                no_capture_limit=100, opening_moves=opening)
            states, _, red_winner = worker.play_game(verbose=False)
            winner = red_winner if model_is_red else -red_winner
            results.append((winner, len(states)))
            print(f'game {index + 1}: {"W" if winner > 0 else "L" if winner < 0 else "D"} '
                  f'({len(states)} plies)')
    finally:
        opponent.close()
    summary = summarize_results(results)
    low, high = wilson_interval(summary['wins'], summary['games'])
    print(summary)
    print(f'win rate 95% CI: {low:.1%}..{high:.1%}')


if __name__ == '__main__':
    main()
