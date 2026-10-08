# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AlphaZero-style Xiangqi (Chinese chess) engine: game rules, a ResNet policy/value network, MCTS search, and a self-play training loop.

## Commands

### Install dependencies
```powershell
pip install -r requirements.txt
```
PyTorch is required for model/MCTS/training work but optional for game-logic-only changes.

### Run tests
```powershell
# Game logic tests (always run first for engine changes):
python tests/test_game.py

# Model tests (requires PyTorch):
python tests/test_model.py

# Full suite via pytest:
pytest tests/

# Single test:
pytest tests/test_game.py::test_initial_position -v
```

### Run the app
```powershell
# Terminal play (human vs AI, requires PyTorch + model checkpoint):
python scripts/play_cli.py --model models/best_model.pt

# Web UI (human vs human, no PyTorch needed):
python scripts/play_web.py --port 8000

# Quick demo (create model → self-play → train 1 epoch):
python scripts/quick_start.py

# Full training loop:
python scripts/train.py --config configs/training_config.yaml
```

## Architecture

### Package layout

- `src/game/` — Board state, piece movement, legal-move validation, game results. No dependency on PyTorch.
- `src/model/` — PyTorch neural network (`ChineseChessNet`) and MCTS. Depends on game layer only through `GameState.to_tensor()`, `get_legal_moves()`, `make_move()`.
- `src/training/` — Self-play data generation, replay buffer, trainer, evaluator. Lazy-imports PyTorch modules.
- `src/utils/` — Logging and checkpoint management.
- `scripts/` — CLI entry points. Each does `sys.path.insert(0, repo_root)` to resolve `src/` imports.
- `tests/` — `test_game.py` and `test_model.py` support both `python file.py` and `pytest`. `test_play_web.py` is pytest-only.

### Key data flow

```
GameState → .to_tensor() → (19, 10, 9) float32 → ChineseChessNet → (policy[1800], value[-1,1])
                                                                          ↓
MCTS search ← neural network evaluation ← GameState.get_legal_moves()
    ↓
SelfPlayWorker collects (state, policy_target, value) → ReplayBuffer → Trainer
```

### Board and coordinates

- 10 rows × 9 columns. A move is `(from_row, from_col, to_row, to_col)`.
- Red at rows 0–4 (advances toward higher rows). Black at rows 5–9 (advances toward lower rows).
- Red palace: rows 0–2, cols 3–5. Black palace: rows 7–9, cols 3–5.
- Empty squares use `Piece(PieceType.EMPTY, ...)`, never `None`.

### Move legality (two-layer design)

`Piece.get_possible_moves()` generates **pseudo-legal** moves (respects piece movement rules and board edges). `Rules.is_move_legal()` then rejects moves that leave the moving side in check. Always go through `Rules` / `GameState.get_legal_moves()` for actual legality.

### Action space encoding (1800)

Defined in `src/game/rules.py`. Each of 90 board squares gets 20 destination buckets: `action = (from_row * 9 + from_col) * 20 + ((to_row * 9 + to_col) % 20)`. Collisions are intentional — `get_policy_target()` accumulates probabilities with `+=`, and `action_index_to_move()` decodes by brute-force matching against legal moves. This encoding is shared by the policy head, MCTS, and all move encoders. Changing it requires retraining.

### Tensor representation (19 channels)

Perspective-relative: channels 0–6 = own pieces, 7–13 = opponent pieces, 14 = palace zones, 15 = river, 16 = turn flag, 17 = no-capture draw progress (`moves_since_capture / 100`), 18 = repetition count (`reps / 3`). When encoding for black, rows are mirrored and own/opponent channels swap.

### Neural network

`ChineseChessNet`: Conv → N×ResidualBlock → dual heads (policy logits[1800] + value tanh[-1,1]). Three sizes via `ModelConfig.from_size('small'|'medium'|'large')`. Default training size is medium (10 blocks, 256 channels, ~12M params).

### MCTS value convention

Values are from the **current node's side-to-move** perspective. During backup, the value **flips sign** at each parent level. Dirichlet noise is added at the root for exploration.

## Contracts to Preserve

- The 1800 action space, (19, 10, 9) tensor shape, and MCTS sign-flip convention are central contracts. Changes require updating all producers, consumers, tests, and a retraining/compatibility decision.
- Game-rule changes go in `src/game/`; keep NN/search code independent of CLI/web presentation.
- Model defaults live in `ModelConfig`; training parameters live in YAML under `configs/`. Don't hardcode training budgets or paths.
- Rule correctness > search/training performance. Add position-based tests for any change to movement, check, checkmate, stalemate, repetition, or undo.
- Model artifacts (`.pt`, `.pth`, `.onnx`), checkpoints, self-play data, and logs are gitignored generated outputs. Don't commit them.

## Code Style

- Max ~300 lines/file, ~40 lines/function. Use early returns to keep nesting ≤ 3 levels.
- Type hints and docstrings on public APIs. English identifiers; comments/docstrings may be Vietnamese or English.
- Imports: stdlib → third-party → local, separated by blank lines. No wildcards. Package-relative imports inside `src/`.
- Catch only expected exceptions — no bare `except`.
- Named constants for non-obvious numbers.
