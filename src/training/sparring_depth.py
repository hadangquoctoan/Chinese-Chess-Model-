"""Pick Pikafish sparring depth from measured model win rates."""
from typing import Any

MIN_SAMPLE = 5
LOW_WIN_RATE = 0.20
MIX_DEPTH2_RATE = 0.60
HIGH_DEPTH1_RATE = 0.75
HIGH_DEPTH2_RATE = 0.40


def choose_sparring_depth(rng: Any, wins1: int, games1: int,
                          wins2: int, games2: int) -> int:
    """Return 1 or 2 from win rates, not from iteration count.

    Starts at depth 1. Mixes depth 2 into one third of games after beating
    depth 1 at 60%. Half depth 2 after beating depth 1 at 75% and depth 2
    at 40%. Drops back to depth 1 if the depth-1 win rate falls below 20%.
    """
    if type(wins1) is not int or type(games1) is not int:
        raise ValueError("depth-1 counters must be ints")
    if type(wins2) is not int or type(games2) is not int:
        raise ValueError("depth-2 counters must be ints")
    if min(wins1, games1, wins2, games2) < 0 or wins1 > games1 or wins2 > games2:
        raise ValueError("win counters disagree with game counts")
    if games1 < MIN_SAMPLE:
        return 1
    rate1 = wins1 / games1
    if rate1 < LOW_WIN_RATE:
        return 1
    mix_two = 1.0 / 3.0
    if (rate1 > HIGH_DEPTH1_RATE and games2 >= MIN_SAMPLE
            and wins2 / games2 > HIGH_DEPTH2_RATE):
        mix_two = 0.5
    elif rate1 <= MIX_DEPTH2_RATE:
        return 1
    random_method = getattr(rng, "random", None)
    if not callable(random_method):
        raise TypeError("rng must have a callable .random() method")
    return 2 if random_method() < mix_two else 1
