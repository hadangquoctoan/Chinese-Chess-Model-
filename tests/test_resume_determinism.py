"""Real execution must continue identically after a saved game boundary."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.train import create_session


def test_real_resume_matches_uninterrupted_next_iteration(tmp_path):
    torch.set_num_threads(1)
    config = {
        "model": {"size": "small", "num_residual_blocks": 0, "num_channels": 4,
                  "policy_channels": 1, "value_channels": 1},
        "training": {"learning_rate": 0.001, "weight_decay": 0.0001,
                     "batch_size": 4, "use_amp": False},
        "self_play": {"num_games_per_iteration": 2, "mcts_simulations": 2,
                      "temperature": 1.0, "temperature_threshold": 15,
                      "dirichlet_alpha": 0.3, "dirichlet_epsilon": 0.25,
                      "max_game_plies": 4, "no_capture_limit": 100},
        "replay_buffer": {"max_size": 20},
        "training_loop": {"num_iterations": 1, "samples_per_iteration": 8,
                          "evaluation_games": 2, "score_threshold": 0.55,
                          "evaluation_mcts_simulations": 2,
                          "evaluation_opening_plies": 1},
        "paths": {"best_model_path": str(tmp_path / "best.pt"),
                  "checkpoint_dir": str(tmp_path / "checkpoints"),
                  "log_dir": str(tmp_path / "logs")},
    }
    torch.manual_seed(17)
    np.random.seed(19)
    run = create_session(config)
    restored = None
    try:
        run.start()
        run.advance()
        checkpoint = str(tmp_path / "game-boundary.pt")
        run.save_progress(checkpoint)
        expected = run.run_iteration()
        weights = {name: tensor.clone() for name, tensor in run.trainer.model.state_dict().items()}
        replay = list(run.replay_buffer.buffer)
        restored = create_session(config, resume=checkpoint)
        restored.start(resume=checkpoint)
        actual = restored.run_iteration()
        assert actual == expected
        for name, tensor in restored.trainer.model.state_dict().items():
            torch.testing.assert_close(tensor, weights[name], rtol=0, atol=0)
        for first, second in zip(replay, restored.replay_buffer.buffer):
            np.testing.assert_array_equal(first[0], second[0])
            np.testing.assert_array_equal(first[1], second[1])
            assert first[2] == second[2]
        assert restored.trainer.epoch == run.trainer.epoch
        assert restored.trainer.global_step == run.trainer.global_step
    finally:
        run.trainer.close()
        if restored is not None:
            restored.trainer.close()
