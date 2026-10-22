"""Choose between self-play and sparring for the next game."""
import math
from typing import Any


def choose_worker_for_game(rng: Any, sparring_ratio: float, sparring_enabled: bool) -> str:
    """Return "self" or "sparring" for the next game.

    Args:
        rng: Random source with a ``.random()`` method like ``random.Random``.
        sparring_ratio: Probability of picking sparring, in [0, 1].
        sparring_enabled: When False, always return "self" without validating ratio.

    Raises:
        TypeError: If ``rng`` has no callable ``.random()``.
        ValueError: If enabled and ``sparring_ratio`` is not a finite number in [0, 1].
    """
    if not sparring_enabled:
        return "self"
    if type(sparring_ratio) not in (int, float) or not math.isfinite(sparring_ratio):
        raise ValueError("sparring_ratio must be a finite number in [0, 1]")
    if not 0 <= sparring_ratio <= 1:
        raise ValueError("sparring_ratio must be a finite number in [0, 1]")
    random_method = getattr(rng, "random", None)
    if not callable(random_method):
        raise TypeError("rng must have a callable .random() method")
    return "sparring" if random_method() < sparring_ratio else "self"
