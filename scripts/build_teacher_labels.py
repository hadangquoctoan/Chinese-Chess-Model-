"""Generate supervised (policy, value) labels with one Pikafish process."""
import argparse
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.fen import board_to_fen
from src.engine.uci import PikafishEngine, find_engine_binary
from src.game.game_state import GameState
from src.game.rules import ACTION_ENCODING_VERSION, RULESET_VERSION
from src.training.teacher_label import SCORE_SCALE, build_teacher_label

DEFAULT_OUTPUT = Path("data") / "teacher_labels" / "teacher_depth2.npz"
DEFAULT_POSITIONS = 1000
DEFAULT_DEPTH = 2
DEFAULT_THREADS = 8
DEFAULT_HASH_MB = 1024
MAX_RANDOM_PLIES = 80


def _collect_position(engine: PikafishEngine, state: GameState, depth: int,
                      timeout: float) -> tuple[np.ndarray, np.ndarray, float] | None:
    fen = board_to_fen(state.board, state.is_red_turn)
    token, score_cp = engine.search_with_eval(fen, [], depth=depth, timeout=timeout)
    try:
        policy, value = build_teacher_label(
            state.board, state.is_red_turn, token, score_cp)
    except ValueError:
        return None
    return state.to_tensor(), policy, value


def generate_positions(engine: PikafishEngine, num_positions: int, depth: int,
                       rng: random.Random, timeout: float) -> tuple[list, list, list]:
    """Play random legal games and label each non-terminal ply with Pikafish."""
    states, policies, values = [], [], []
    games = 0
    while len(states) < num_positions:
        games += 1
        if games > num_positions * 5:
            raise RuntimeError(f"Could not collect {num_positions} teacher labels")
        game = GameState(max_plies=MAX_RANDOM_PLIES, no_capture_limit=100)
        while (not game.is_terminal()) and len(states) < num_positions:
            labeled = _collect_position(engine, game, depth, timeout)
            if labeled is not None:
                state, policy, value = labeled
                states.append(state)
                policies.append(policy)
                values.append(value)
            legal = game.get_legal_moves()
            if not legal:
                break
            game.make_move(*legal[rng.randrange(len(legal))])
    return states, policies, values


def save_teacher_npz(path: Path, states, policies, values, *, depth: int,
                     engine_path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": np.int64(1),
        "action_encoding_version": np.asarray(ACTION_ENCODING_VERSION),
        "ruleset_version": np.asarray(RULESET_VERSION),
        "score_scale": np.float32(SCORE_SCALE),
        "engine_depth": np.int64(depth),
        "engine_name": np.asarray("pikafish"),
        "engine_path": np.asarray(str(engine_path)),
        "value_convention": np.asarray("red_centric"),
        "states": np.stack(states).astype(np.float32),
        "policies": np.stack(policies).astype(np.float32),
        "values": np.asarray(values, dtype=np.float32),
    }
    np.savez_compressed(path, **payload)


def load_teacher_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    for name in ("format_version", "engine_depth", "score_scale",
                 "action_encoding_version", "ruleset_version", "engine_name",
                 "value_convention"):
        if name in payload:
            payload[name] = payload[name].item()
    return payload


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Build Pikafish teacher labels")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-positions", type=int, default=DEFAULT_POSITIONS)
    parser.add_argument("--depth", type=int, default=DEFAULT_DEPTH)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS)
    parser.add_argument("--hash-mb", type=int, default=DEFAULT_HASH_MB)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args(argv)
    if args.num_positions <= 0:
        parser.error("--num-positions must be positive")
    binary = find_engine_binary()
    rng = random.Random(args.seed)
    engine = PikafishEngine(path=binary, threads=args.threads, hash_mb=args.hash_mb)
    try:
        states, policies, values = generate_positions(
            engine, args.num_positions, args.depth, rng, args.timeout)
    finally:
        engine.close()
    save_teacher_npz(args.output, states, policies, values, depth=args.depth,
                     engine_path=binary)
    loaded = load_teacher_npz(args.output)
    if loaded["states"].shape[0] != args.num_positions:
        raise RuntimeError("Saved teacher label count disagrees with request")
    print(f"Wrote {args.num_positions} labels to {args.output}")


if __name__ == "__main__":
    main()
