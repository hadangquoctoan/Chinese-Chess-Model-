"""Generate supervised (policy, value) labels with one Pikafish process."""
import argparse
from pathlib import Path
import random
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.engine.coordinates import parse_uci_move
from src.engine.fen import board_to_fen
from src.engine.uci import PikafishEngine, find_engine_binary
from src.game.game_state import GameState
from src.game.pieces import PieceType
from src.game.rules import ACTION_ENCODING_VERSION, RULESET_VERSION
from src.training.adjudication import MATERIAL_VALUES
from src.training.teacher_label import SCORE_SCALE, build_teacher_label

DEFAULT_OUTPUT = Path("data") / "teacher_labels" / "teacher_depth2.npz"
DEFAULT_POSITIONS = 1000
DEFAULT_DEPTH = 2
DEFAULT_THREADS = 8
DEFAULT_HASH_MB = 1024
MAX_RANDOM_PLIES = 80
MAX_TACTICAL_PROBES = 100


def _append_probe(probes, probe, limit: int = MAX_TACTICAL_PROBES) -> None:
    """Keep the tactical probe file small enough for a practical benchmark."""
    if len(probes) < limit:
        probes.append(probe)


def _capture_value(board, move) -> float:
    """Return the material value of the target piece for one move."""
    _, _, to_row, to_col = move
    target = board.get_piece(to_row, to_col)
    if target.piece_type == PieceType.EMPTY:
        return 0.0
    return MATERIAL_VALUES[target.piece_type]


def _collect_position(engine: PikafishEngine, state: GameState, depth: int,
                      timeout: float) -> tuple[np.ndarray, np.ndarray, float, tuple, float] | None:
    fen = board_to_fen(state.board, state.is_red_turn)
    token, score_cp = engine.search_with_eval(fen, [], depth=depth, timeout=timeout)
    try:
        move = parse_uci_move(token)
        policy, value = build_teacher_label(
            state.board, state.is_red_turn, token, score_cp)
    except ValueError:
        return None
    return state.to_tensor(), policy, value, move, _capture_value(state.board, move)


def generate_positions(engine: PikafishEngine, num_positions: int, depth: int,
                       rng: random.Random, timeout: float) -> tuple[list, list, list, list]:
    """Play random legal games and label each non-terminal ply with Pikafish."""
    states, policies, values, probes = [], [], [], []
    games = 0
    while len(states) < num_positions:
        games += 1
        if games > num_positions * 5:
            raise RuntimeError(f"Could not collect {num_positions} teacher labels")
        game = GameState(max_plies=MAX_RANDOM_PLIES, no_capture_limit=100)
        while (not game.is_terminal()) and len(states) < num_positions:
            labeled = _collect_position(engine, game, depth, timeout)
            if labeled is not None:
                state, policy, value, move, capture = labeled
                states.append(state)
                policies.append(policy)
                values.append(value)
                if capture >= 4.0:
                    history = tuple(tuple(record[:4]) for record in game.move_history)
                    _append_probe(probes, (state, history, move, capture))
            legal = game.get_legal_moves()
            if not legal:
                break
            game.make_move(*legal[rng.randrange(len(legal))])
    return states, policies, values, probes


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


def save_probes_npz(path: Path, probes) -> None:
    """Save capture probes with the move history needed to replay each position."""
    if not probes:
        raise ValueError("No probes to save")
    path.parent.mkdir(parents=True, exist_ok=True)
    histories = [move for _, history, _, _ in probes for move in history]
    offsets = [0]
    for _, history, _, _ in probes:
        offsets.append(offsets[-1] + len(history))
    np.savez_compressed(
        path,
        states=np.stack([state for state, _, _, _ in probes]).astype(np.float32),
        histories=np.asarray(histories, dtype=np.int16).reshape(-1, 4),
        history_offsets=np.asarray(offsets, dtype=np.int32),
        moves=np.asarray([move for _, _, move, _ in probes], dtype=np.int16).reshape(-1, 4),
        captured=np.asarray([value for _, _, _, value in probes], dtype=np.float32),
    )


def load_probe_npz(path: Path) -> dict:
    """Load capture probes and group their flat histories per position."""
    with np.load(path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    offsets = payload['history_offsets']
    histories = payload['histories']
    payload['history_groups'] = [
        [tuple(int(v) for v in histories[index])
         for index in range(offsets[i], offsets[i + 1])]
        for i in range(len(offsets) - 1)]
    return payload


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
    parser.add_argument("--tactical-probes-out", type=Path, default=None)
    args = parser.parse_args(argv)
    if args.num_positions <= 0:
        parser.error("--num-positions must be positive")
    binary = find_engine_binary()
    rng = random.Random(args.seed)
    engine = PikafishEngine(path=binary, threads=args.threads, hash_mb=args.hash_mb)
    try:
        states, policies, values, probes = generate_positions(
            engine, args.num_positions, args.depth, rng, args.timeout)
    finally:
        engine.close()
    save_teacher_npz(args.output, states, policies, values, depth=args.depth,
                     engine_path=binary)
    loaded = load_teacher_npz(args.output)
    if loaded["states"].shape[0] != args.num_positions:
        raise RuntimeError("Saved teacher label count disagrees with request")
    if args.tactical_probes_out is not None:
        if not probes:
            raise RuntimeError("No tactical capture probes were collected")
        save_probes_npz(args.tactical_probes_out, probes)
        print(f"Wrote {len(probes)} tactical probes to {args.tactical_probes_out}")
    print(f"Wrote {args.num_positions} labels to {args.output}")


if __name__ == "__main__":
    main()
