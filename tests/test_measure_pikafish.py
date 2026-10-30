"""Unit tests for Pikafish match scoring without a live engine."""
import numpy as np
import pytest

from scripts.measure_vs_pikafish import (
    MAX_GAME_PLIES,
    generate_opening,
    match_score,
    measure,
    play_measured_game,
)
from src.game.game_state import GameState
from src.game.rules import Rules


def test_match_score_half_draws():
    assert match_score(15, 5, 20) == 0.875
    assert match_score(10, 0, 20) == 0.5
    assert match_score(0, 20, 20) == 0.5


def test_play_measured_game_with_fakes(monkeypatch):
    class FakeOpponent:
        def choose_move(self, state):
            return state.get_legal_moves()[0]

    class FakeMCTS:
        def __init__(self, model, **kwargs):
            pass

        def get_action_probs(self, game_state, temperature=0.0):
            return game_state.get_legal_moves()[0], np.zeros(1800, dtype=np.float32)

    monkeypatch.setattr("scripts.measure_vs_pikafish.MCTS", FakeMCTS)
    winner, _, plies = play_measured_game(
        object(), FakeOpponent(), model_is_red=True, mcts_simulations=1,
        max_game_plies=4)
    assert winner in (-1.0, 0.0, 1.0)
    assert plies == 4


def test_raw_policy_masks_illegal_logits():
    from scripts.measure_vs_pikafish import choose_raw_policy_move

    state = GameState()
    legal = state.get_legal_moves()
    legal_indices = {Rules.move_to_action_index(*move, state.board) for move in legal}
    illegal = next(index for index in range(1800) if index not in legal_indices)
    policy = np.zeros(1800, dtype=np.float32)
    policy[illegal] = 100.0
    policy[Rules.move_to_action_index(*legal[0], state.board)] = 1.0

    class FakeModel:
        def predict(self, tensor):
            return policy, 0.0

    assert choose_raw_policy_move(FakeModel(), state) == legal[0]


def test_raw_policy_chooses_highest_probability_legal_move():
    from scripts.measure_vs_pikafish import choose_raw_policy_move

    state = GameState()
    legal = state.get_legal_moves()
    policy = np.zeros(1800, dtype=np.float32)
    for move in legal:
        policy[Rules.move_to_action_index(*move, state.board)] = 0.5
    preferred = legal[-1]
    policy[Rules.move_to_action_index(*preferred, state.board)] = 0.9

    class FakeModel:
        def predict(self, tensor):
            return policy, 0.0

    assert choose_raw_policy_move(FakeModel(), state) == preferred


def test_raw_policy_returns_none_without_legal_moves():
    from scripts.measure_vs_pikafish import choose_raw_policy_move

    state = GameState(max_plies=1)
    state.make_move(*state.get_legal_moves()[0])
    assert state.get_legal_moves() == []

    class FakeModel:
        def predict(self, tensor):
            raise AssertionError("no policy query without legal moves")

    assert choose_raw_policy_move(FakeModel(), state) is None


def test_raw_mode_does_not_construct_mcts(monkeypatch):
    class FakeModel:
        def predict(self, tensor):
            return np.full(1800, 1 / 1800, dtype=np.float32), 0.0

    class FakeOpponent:
        def choose_move(self, state):
            return state.get_legal_moves()[0]

    def forbidden(*args, **kwargs):
        raise AssertionError('MCTS must not be constructed in raw mode')

    monkeypatch.setattr('scripts.measure_vs_pikafish.MCTS', forbidden)
    _, _, plies = play_measured_game(
        FakeModel(), FakeOpponent(), True, 1, max_game_plies=2, mode='raw')
    assert plies == 2


def test_play_measured_game_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Unsupported measurement mode"):
        play_measured_game(object(), object(), True, 1, max_game_plies=2,
                           mode='greedy')


def test_measure_forwards_ply_limit_and_mode(monkeypatch):
    calls = []

    def fake_play_measured_game(model, opponent, model_is_red, mcts_simulations,
                                **kwargs):
        calls.append((model_is_red, mcts_simulations, kwargs))
        return 0.0, False, 10

    monkeypatch.setattr('scripts.measure_vs_pikafish.play_measured_game',
                        fake_play_measured_game)
    result = measure('model', 'opponent', games=2, mcts_simulations=7,
                     max_game_plies=24, mode='raw')
    assert result['mode'] == 'raw'
    assert calls == [
        (True, 7, {'max_game_plies': 24, 'opening_moves': (), 'mode': 'raw'}),
        (False, 7, {'max_game_plies': 24, 'opening_moves': (), 'mode': 'raw'}),
    ]


def test_measure_defaults_to_mcts_and_default_ply_limit(monkeypatch):
    calls = []

    def fake_play_measured_game(model, opponent, model_is_red, mcts_simulations,
                                **kwargs):
        calls.append((model_is_red, mcts_simulations, kwargs))
        return 1.0, False, 10

    monkeypatch.setattr('scripts.measure_vs_pikafish.play_measured_game',
                        fake_play_measured_game)
    result = measure('model', 'opponent', games=1, mcts_simulations=3)
    assert result['mode'] == 'mcts'
    assert calls == [(True, 3, {'max_game_plies': MAX_GAME_PLIES,
                                'opening_moves': (), 'mode': 'mcts'})]


def test_opening_is_reproducible_and_varies_by_seed():
    first = generate_opening(4, 5)
    assert len(first) == 4
    assert first == generate_opening(4, 5)
    assert first != generate_opening(4, 6)
    state = GameState()
    for move in first:
        assert state.make_move(*move)


def test_measure_pairs_share_one_opening(monkeypatch):
    calls = []

    def fake_play_measured_game(model, opponent, model_is_red, mcts_simulations,
                                **kwargs):
        calls.append((model_is_red, kwargs['opening_moves']))
        return 0.0, False, 8

    monkeypatch.setattr('scripts.measure_vs_pikafish.play_measured_game',
                        fake_play_measured_game)
    measure('model', 'opponent', games=4, mcts_simulations=1,
            max_game_plies=24, mode='raw', opening_plies=4, opening_seed=9)
    assert [model_is_red for model_is_red, _ in calls] == [True, False, True, False]
    assert calls[0][1] == calls[1][1]
    assert calls[2][1] == calls[3][1]
    assert calls[0][1] != calls[2][1]


def test_cli_rejects_nonpositive_max_game_plies(capsys):
    from scripts import measure_vs_pikafish as module

    with pytest.raises(SystemExit):
        module.main(['--model', 'dummy.pt', '--games', '2',
                     '--max-game-plies', '0'])
    assert '--max-game-plies must be a positive integer' in capsys.readouterr().err


def test_cli_forwards_raw_mode_and_ply_limit(monkeypatch):
    from scripts import measure_vs_pikafish as module

    recorded = {}

    class FakeModel:
        def eval(self):
            return self

    class FakeOpponent:
        def close(self):
            recorded['closed'] = True

    monkeypatch.setattr(module, 'load_model_artifact',
                        lambda *args, **kwargs: FakeModel())
    monkeypatch.setattr(module, 'load_pikafish_player',
                        lambda **kwargs: FakeOpponent())

    def fake_measure(model, opponent, games, mcts_simulations, **kwargs):
        recorded['games'] = games
        recorded['mcts_simulations'] = mcts_simulations
        recorded['kwargs'] = kwargs
        return {'games': games, 'wins': 1, 'losses': 0, 'draws': 1,
                'score': 0.75, 'first_capture_rate': 0.5, 'mode': kwargs['mode']}

    monkeypatch.setattr(module, 'measure', fake_measure)
    result = module.main(['--model', 'dummy.pt', '--games', '2', '--mode', 'raw',
                          '--max-game-plies', '24', '--mcts-sims', '7',
                          '--opening-plies', '4', '--opening-seed', '3'])
    assert result['mode'] == 'raw'
    assert recorded == {
        'games': 2,
        'mcts_simulations': 7,
        'kwargs': {'max_game_plies': 24, 'mode': 'raw',
                   'opening_plies': 4, 'opening_seed': 3},
        'closed': True,
    }


def test_cli_rejects_negative_opening_plies(capsys):
    from scripts import measure_vs_pikafish as module

    with pytest.raises(SystemExit):
        module.main(['--model', 'dummy.pt', '--games', '2',
                     '--opening-plies', '-1'])
    assert '--opening-plies must be a nonnegative integer' in capsys.readouterr().err
