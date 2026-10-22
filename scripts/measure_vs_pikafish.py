"""Measure one checkpoint against Pikafish at a fixed depth."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.player import load_pikafish_player
from src.game.game_state import GameState
from src.game.pieces import PieceType
from src.model.mcts import MCTS
from src.utils.model_io import load_model_artifact

DEFAULT_GAMES = 20
DEFAULT_DEPTH = 1
DEFAULT_MCTS = 128
MAX_GAME_PLIES = 200


def match_score(wins: int, draws: int, games: int) -> float:
    if games <= 0:
        raise ValueError("games must be positive")
    return (wins + 0.5 * draws) / games


def play_measured_game(model, opponent, model_is_red: bool, mcts_simulations: int,
                       max_game_plies: int = MAX_GAME_PLIES) -> tuple[float, bool, int]:
    """Return (red-centric winner, model_captured_first, plies)."""
    state = GameState(max_plies=max_game_plies, no_capture_limit=100)
    mcts = MCTS(model, num_simulations=mcts_simulations, temperature=0.0,
                dirichlet_epsilon=0.0)
    first_capture_model: bool | None = None
    plies = 0
    while not state.is_terminal():
        model_to_move = state.is_red_turn == model_is_red
        if model_to_move:
            move, _ = mcts.get_action_probs(state, temperature=0.0)
        else:
            move = opponent.choose_move(state)
        if move is None:
            raise RuntimeError("No move in a non-terminal measurement game")
        target = state.board.get_piece(move[2], move[3])
        is_capture = target.piece_type != PieceType.EMPTY
        if not state.make_move(*move):
            raise RuntimeError(f"Illegal measurement move: {move}")
        plies += 1
        if is_capture and first_capture_model is None:
            first_capture_model = model_to_move
    winner_bool = state.get_winner()
    if winner_bool is None:
        winner = 0.0
    else:
        winner = 1.0 if winner_bool else -1.0
    return winner, bool(first_capture_model), plies


def measure(model, opponent, games: int, mcts_simulations: int) -> dict:
    wins = losses = draws = first_captures = 0
    for index in range(games):
        model_is_red = index % 2 == 0
        winner, model_first, _ = play_measured_game(
            model, opponent, model_is_red, mcts_simulations)
        model_view = winner if model_is_red else -winner
        if model_view > 0:
            wins += 1
        elif model_view < 0:
            losses += 1
        else:
            draws += 1
        first_captures += int(model_first)
    return {
        "games": games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "score": match_score(wins, draws, games),
        "first_capture_rate": first_captures / games,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Match a .pt file vs Pikafish")
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--mcts-sims", type=int, default=DEFAULT_MCTS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=1024)
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even integer")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_artifact(args.model, device=device)
    model.eval()
    opponent = load_pikafish_player(depth=args.depth, threads=args.threads,
                                    hash_mb=args.hash_mb)
    try:
        result = measure(model, opponent, args.games, args.mcts_sims)
    finally:
        opponent.close()
    print(f"depth {args.depth} games {result['games']} "
          f"W{result['wins']}-L{result['losses']}-D{result['draws']} "
          f"score {result['score']:.1%} first-capture {result['first_capture_rate']:.1%}")
    return result


if __name__ == "__main__":
    main()
