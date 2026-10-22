"""Validate the ``sparring`` section of the training YAML config."""
import math
from pathlib import Path
from typing import Any, Dict

_REPO_ROOT = Path(__file__).resolve().parents[2]
KNOWN_KEYS = frozenset({"enabled", "ratio", "depth", "threads", "hash_mb", "engine_path"})

DEFAULTS = {
    "enabled": False,
    "ratio": 0.3,
    "depth": 1,
    "threads": 8,
    "hash_mb": 1024,
    "engine_path": None,
}


def _check_ratio(value: Any) -> None:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("sparring ratio must be a finite number in [0, 1]")


def _check_positive_int(value: Any, name: str) -> None:
    if type(value) is not int or value <= 0:
        raise ValueError(f"sparring {name} must be a positive int")


def _resolve_engine_path(engine_path: Any) -> str:
    if engine_path is None:
        raise ValueError("sparring engine_path must point to an existing file when enabled")
    candidates = [Path(engine_path)]
    if not candidates[0].is_absolute():
        candidates.append(_REPO_ROOT / engine_path)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise ValueError("sparring engine_path must point to an existing file when enabled")


def validate_sparring_config(section: Any) -> Dict[str, Any]:
    """Validate the ``sparring`` YAML section and return a normalized dict.

    Args:
        section: The ``sparring`` mapping from YAML, or None if absent.

    Returns:
        Normalized dict with keys ``enabled, ratio, depth, threads,
        hash_mb, engine_path``.

    Raises:
        ValueError: On unknown keys, bad types/ranges, or a missing
            engine binary when ``enabled`` is true.
        TypeError: If ``section`` is neither None nor a dict.
    """
    if section is None:
        return dict(DEFAULTS)
    if not isinstance(section, dict):
        raise TypeError("sparring section must be a mapping or None")
    unknown = set(section) - KNOWN_KEYS
    if unknown:
        raise ValueError(f"unknown sparring keys: {sorted(unknown)}")

    enabled = section.get("enabled", False)
    if type(enabled) is not bool:
        raise ValueError("sparring enabled must be a bool")

    ratio = section.get("ratio", DEFAULTS["ratio"])
    depth = section.get("depth", DEFAULTS["depth"])
    threads = section.get("threads", DEFAULTS["threads"])
    hash_mb = section.get("hash_mb", DEFAULTS["hash_mb"])
    engine_path = section.get("engine_path", DEFAULTS["engine_path"])

    _check_ratio(ratio)
    if type(depth) is not int or not 1 <= depth <= 30:
        raise ValueError("sparring depth must be an int in [1, 30]")
    _check_positive_int(threads, "threads")
    _check_positive_int(hash_mb, "hash_mb")
    if engine_path is not None and not isinstance(engine_path, (str, Path)):
        raise ValueError("sparring engine_path must be a string path or None")

    if enabled:
        engine_path = _resolve_engine_path(engine_path)

    return {
        "enabled": enabled,
        "ratio": ratio,
        "depth": depth,
        "threads": threads,
        "hash_mb": hash_mb,
        "engine_path": engine_path,
    }
