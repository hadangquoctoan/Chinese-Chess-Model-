"""Parse UCI info-line scores for Pikafish labels and measurements.

Plan ref: docs/plan_pikafish_9h.md section 1.2. Single responsibility:
extract a centipawn score from UCI stdout lines. Does not touch
``PikafishEngine.bestmove()``.
"""

MATE_SCORE = 100000


def parse_uci_score(lines: list[str]) -> int:
    """Return the last UCI score found in ``lines`` as centipawns.

    ``score cp <int>`` returns the int verbatim. ``score mate <N>``
    returns ``MATE_SCORE * sign(N)``. ``bestmove`` lines and bare
    ``score`` tokens without ``cp``/``mate`` are ignored.

    Raises:
        ValueError: if no score is found, or ``score mate 0`` occurs.
    """
    found: int | None = None
    for line in lines:
        tokens = line.split()
        for i, tok in enumerate(tokens):
            if tok != "score" or i + 2 >= len(tokens):
                continue
            kind, raw = tokens[i + 1], tokens[i + 2]
            if kind == "cp":
                try:
                    found = int(raw)
                except ValueError:
                    continue
            elif kind == "mate":
                try:
                    mate_in = int(raw)
                except ValueError:
                    continue
                if mate_in == 0:
                    raise ValueError("score mate 0 is invalid")
                found = MATE_SCORE if mate_in > 0 else -MATE_SCORE
    if found is None:
        raise ValueError("no UCI score found")
    return found
