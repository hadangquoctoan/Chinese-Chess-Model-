"""Tests for src/engine/search_score.py (plan section 1.2)."""

import pytest

from src.engine.search_score import parse_uci_score


def test_cp_positive():
    lines = ["info depth 10 score cp 87 nodes 100 nps 1000"]
    assert parse_uci_score(lines) == 87


def test_cp_negative():
    lines = ["info depth 12 score cp -120 nodes 200"]
    assert parse_uci_score(lines) == -120


def test_mate_positive():
    lines = ["info depth 8 score mate 3 pv e0e1"]
    assert parse_uci_score(lines) == 100000


def test_mate_negative():
    lines = ["info depth 8 score mate -2 pv e9e8"]
    assert parse_uci_score(lines) == -100000


def test_last_score_wins():
    lines = [
        "info depth 5 score cp 10 nodes 50",
        "info depth 10 score cp 87 nodes 100",
        "bestmove e0e1",
    ]
    assert parse_uci_score(lines) == 87


def test_missing_score_raises():
    with pytest.raises(ValueError):
        parse_uci_score(["info depth 3 nodes 10", "bestmove e0e1"])


def test_mate_zero_raises():
    with pytest.raises(ValueError):
        parse_uci_score(["info depth 8 score mate 0"])


def test_empty_or_incomplete_score_raises():
    with pytest.raises(ValueError):
        parse_uci_score([])
    with pytest.raises(ValueError):
        parse_uci_score(["info depth 3 score", "bestmove e0e1"])
