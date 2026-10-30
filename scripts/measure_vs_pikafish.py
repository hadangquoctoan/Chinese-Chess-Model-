"""Measure one checkpoint against Pikafish at a fixed depth."""
import argparse
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.player import load_pikafish_player
from src.game.action_encoding import Move
from src.game.game_state import GameState, NO_CAPTURE_DRAW_PLY_LIMIT
from src.game.pieces import PieceType
from src.game.rules import Rules
from src.model.mcts import MCTS
from src.utils.model_io import load_model_artifact

DEFAULT_GAMES = 20
DEFAULT_DEPTH = 1
DEFAULT_MCTS = 128
DEFAULT_OPENING_PLIES = 0
DEFAULT_OPENING_SEED = 0
MAX_GAME_PLIES = 200
MEASUREMENT_MODES = ("raw", "mcts")


def match_score(wins: int, draws: int, games: int) -> float:
    if games <= 0:
        raise ValueError("games must be positive")
    return (wins + 0.5 * draws) / games


def _validate_mode(mode: str) -> None:
    if mode not in MEASUREMENT_MODES:
        raise ValueError(f"Unsupported measurement mode: {mode!r}")


def generate_opening(plies: int, seed: int) -> tuple[Move, ...]:
    """Return a reproducible random legal opening for one color pair."""
    if type(plies) is not int or plies < 0:
        raise ValueError("plies must be a nonnegative integer")
    rng = random.Random(seed)
    state = GameState(max_plies=MAX_GAME_PLIES,
                      no_capture_limit=NO_CAPTURE_DRAW_PLY_LIMIT)
    moves: list[Move] = []
    for _ in range(plies):
        if state.is_terminal():
            break
        legal = state.get_legal_moves()
        if not legal:
            break
        move = legal[rng.randrange(len(legal))]
        if not state.make_move(*move):
            raise RuntimeError(f"Generated opening move is illegal: {move}")
        moves.append(move)
    return tuple(moves)


def choose_raw_policy_move(model, state: GameState) -> Move | None:
    """Return the highest-probability legal move without tree search."""
    legal_moves = state.get_legal_moves()
    if not legal_moves:
        return None
    policy, _ = model.predict(state.to_tensor())
    return max(
        legal_moves,
        key=lambda move: float(
            policy[Rules.move_to_action_index(*move, state.board)]),
    )


def play_measured_game(model, opponent, model_is_red: bool, mcts_simulations: int,
                       max_game_plies: int = MAX_GAME_PLIES,
                       opening_moves: tuple = (),
                       mode: str = "mcts") -> tuple[float, bool, int]:
    """Return (red-centric winner, model_captured_first, plies).

    In ``raw`` mode the model plays the legal policy argmax and no MCTS is
    created; ``mcts`` mode searches with ``mcts_simulations`` as before.
    """
    _validate_mode(mode)
    state = GameState(max_plies=max_game_plies,
                      no_capture_limit=NO_CAPTURE_DRAW_PLY_LIMIT)
    for move in opening_moves:
        if state.is_terminal() or not state.make_move(*move):
            raise ValueError(f"Invalid measurement opening move: {move}")
    mcts = None
    if mode == "mcts":
        mcts = MCTS(model, num_simulations=mcts_simulations, temperature=0.0,
                    dirichlet_epsilon=0.0)
    first_capture_model: bool | None = None
    plies = 0
    while not state.is_terminal():
        model_to_move = state.is_red_turn == model_is_red
        if model_to_move and mode == "raw":
            move = choose_raw_policy_move(model, state)
        elif model_to_move:
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


def measure(model, opponent, games: int, mcts_simulations: int,
            max_game_plies: int = MAX_GAME_PLIES,
            mode: str = "mcts",
            opening_plies: int = DEFAULT_OPENING_PLIES,
            opening_seed: int = DEFAULT_OPENING_SEED) -> dict:
    """Play paired games and aggregate the model's score.

    ``max_game_plies`` sets both the game limit and the channel-16
    normalization, which must match the supervised training profile. Paired
    color games share one reproducible random opening; each later pair uses a
    different seed derived from ``opening_seed``.
    """
    _validate_mode(mode)
    if type(opening_plies) is not int or opening_plies < 0:
        raise ValueError("opening_plies must be a nonnegative integer")
    wins = losses = draws = first_captures = 0
    for index in range(games):
        model_is_red = index % 2 == 0
        opening_moves = (
            generate_opening(opening_plies, opening_seed + index // 2)
            if opening_plies else ())
        winner, model_first, _ = play_measured_game(
            model, opponent, model_is_red, mcts_simulations,
            max_game_plies=max_game_plies, opening_moves=opening_moves,
            mode=mode)
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
        "mode": mode,
    }


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description="Match a .pt file vs Pikafish")
    parser.add_argument("--model", required=True)
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--mcts-sims", type=int, default=DEFAULT_MCTS)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--hash-mb", type=int, default=1024)
    parser.add_argument("--max-game-plies", type=int, default=MAX_GAME_PLIES)
    parser.add_argument("--opening-plies", type=int, default=DEFAULT_OPENING_PLIES,
                        help="random legal opening plies shared by each color pair")
    parser.add_argument("--opening-seed", type=int, default=DEFAULT_OPENING_SEED,
                        help="base random seed for measurement openings")
    parser.add_argument("--mode", choices=MEASUREMENT_MODES, default="mcts")
    args = parser.parse_args(argv)
    if args.games <= 0 or args.games % 2:
        parser.error("--games must be a positive even integer")
    if args.max_game_plies <= 0:
        parser.error("--max-game-plies must be a positive integer")
    if args.opening_plies < 0:
        parser.error("--opening-plies must be a nonnegative integer")
    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model_artifact(args.model, device=device)
    model.eval()
    opponent = load_pikafish_player(depth=args.depth, threads=args.threads,
                                    hash_mb=args.hash_mb)
    try:
        result = measure(model, opponent, args.games, args.mcts_sims,
                         max_game_plies=args.max_game_plies, mode=args.mode,
                         opening_plies=args.opening_plies,
                         opening_seed=args.opening_seed)
    finally:
        opponent.close()
    print(f"mode {result['mode']} depth {args.depth} games {result['games']} "
          f"W{result['wins']}-L{result['losses']}-D{result['draws']} "
          f"score {result['score']:.1%} "
          f"first-capture {result['first_capture_rate']:.1%}")
    return result


if __name__ == "__main__":
    main()
