import pytest

from src.game.game_state import GameState
from src.game.pieces import PieceType, create_piece
from src.training.adjudication import adjudicated_winner, material_balance


def test_material_balance_counts_both_colors():
    game = GameState()
    assert material_balance(game) == 0.0
    game.board.set_piece(9, 0, create_piece(PieceType.EMPTY, is_red=True))
    assert material_balance(game) == 9.0


def test_non_terminal_game_is_not_adjudicated():
    assert adjudicated_winner(GameState(), margin=3.0) is None


def test_terminal_draw_is_adjudicated_by_material():
    game = GameState(max_plies=1)
    game.board.set_piece(9, 0, create_piece(PieceType.EMPTY, is_red=True))
    assert game.make_move(3, 0, 4, 0)
    assert game.is_terminal() and game.get_winner() is None
    assert adjudicated_winner(game, margin=3.0) is True
    assert adjudicated_winner(game, margin=10.0) is None
    assert adjudicated_winner(game, margin=0.0) is None


class _Decisive:
    def __init__(self, winner):
        self.winner = winner

    def is_terminal(self):
        return True

    def get_winner(self):
        return self.winner


def test_existing_decisive_result_is_never_overridden():
    assert adjudicated_winner(_Decisive(True), margin=3.0) is True
    assert adjudicated_winner(_Decisive(False), margin=3.0) is False


def test_negative_margin_is_rejected():
    with pytest.raises(ValueError, match="margin"):
        adjudicated_winner(GameState(), margin=-1.0)


def test_self_play_worker_stores_and_validates_margin():
    import torch

    from src.training.self_play import SelfPlayWorker

    worker = SelfPlayWorker(torch.nn.Linear(1, 1), adjudication_margin=3.0)
    assert worker.adjudication_margin == 3.0
    with pytest.raises(ValueError, match="adjudication_margin"):
        SelfPlayWorker(torch.nn.Linear(1, 1), adjudication_margin=-1.0)


def test_pipeline_rejects_negative_adjudication_margin(tmp_path):
    from src.training.pipeline import validate_session_config

    config = {
        'model': {'size': 'small'},
        'training': {'learning_rate': .001, 'weight_decay': 0., 'batch_size': 2,
                     'use_amp': False},
        'self_play': {'num_games_per_iteration': 1, 'mcts_simulations': 1,
                      'adjudication_margin': -1.0},
        'replay_buffer': {'max_size': 2},
        'training_loop': {'num_iterations': 1, 'samples_per_iteration': 1,
                          'evaluation_games': 2, 'score_threshold': .5},
        'paths': {'best_model_path': str(tmp_path / 'best.pt'),
                  'checkpoint_dir': str(tmp_path / 'ckpt'),
                  'log_dir': str(tmp_path / 'logs')},
    }
    with pytest.raises(ValueError, match='adjudication_margin'):
        validate_session_config(config)
