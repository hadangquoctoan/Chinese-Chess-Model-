"""UCI move-string conversion between Pikafish and the project coordinate system.

Project coordinates are (from_row, from_col, to_row, to_col) with rows 0-9 and
cols 0-8. Red sits at rows 0-4. Pikafish UCI tokens look like ``h2e2``: file
``a``-``i`` maps to col 0-8 and the digit maps to the row.

``FLIP_RANK`` is the only orientation switch. Leave it False when engine
opening moves are already in ``Rules.get_all_legal_moves()``; set True only
if live tokens consistently decode as ``9 - row``. Do not change Board/Rules.
"""

import re

from ..game.rules import Move

_MOVE_PATTERN = re.compile(r"^([a-i])([0-9])([a-i])([0-9])$")
_FILES = "abcdefghi"
FLIP_RANK = False


def _rank_to_row(rank: int) -> int:
    return 9 - rank if FLIP_RANK else rank


def _row_to_rank(row: int) -> int:
    return 9 - row if FLIP_RANK else row


def move_to_uci(move: Move) -> str:
    """Convert a project move to a Pikafish UCI move string."""
    from_row, from_col, to_row, to_col = move
    return (
        f"{_FILES[from_col]}{_row_to_rank(from_row)}"
        f"{_FILES[to_col]}{_row_to_rank(to_row)}"
    )


def parse_uci_move(text: str) -> Move:
    """Parse a Pikafish ``bestmove`` token into project coordinates.

    Raises:
        ValueError: if the token is not a 4-character ``[a-i][0-9]`` pair.
    """
    match = _MOVE_PATTERN.match((text or "").strip())
    if match is None:
        raise ValueError(f"Invalid Pikafish move: {text!r}")
    from_file, from_rank, to_file, to_rank = match.groups()
    return (_rank_to_row(int(from_rank)), _FILES.index(from_file),
            _rank_to_row(int(to_rank)), _FILES.index(to_file))
