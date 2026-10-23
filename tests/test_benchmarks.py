import random

from scripts.benchmark_pikafish import generate_opening, summarize_results, wilson_interval
from scripts.benchmark_tactics import capture_value, replay_history
from src.game.game_state import GameState


def test_summarize_results_uses_half_draw_scoring():
    summary = summarize_results([(1.0, 10), (-1.0, 20), (0.0, 30)])
    assert summary == {'games': 3, 'wins': 1, 'losses': 1, 'draws': 1,
                       'score': 0.5, 'average_plies': 20.0}


def test_opening_is_reproducible():
    first = generate_opening(2, random.Random(7))
    second = generate_opening(2, random.Random(7))
    assert first == second
    assert len(first) == 2


def test_wilson_interval_is_bounded():
    low, high = wilson_interval(5, 10)
    assert 0.0 < low < 0.5 < high < 1.0


def test_tactical_helpers_replay_and_detect_captures():
    game = GameState()
    assert capture_value(game, (0, 0, 9, 0)) == 9.0
    replayed = replay_history(((3, 0, 4, 0),))
    assert len(replayed.move_history) == 1
