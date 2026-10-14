"""Tests for the ChartQA prompt and relaxed-accuracy metric."""
import pytest

from common.chartqa import INSTRUCTION, chart_messages, relaxed_match


@pytest.mark.parametrize("pred, gold", [
    ("23.4", "23.4"),
    ("24", "23"),          # within 5 %
    ("1,000", "1000"),
    ("45%", "45"),
    ("$12.5", "12.5"),
    ("Yes.", "yes"),
    (" Canada ", "canada"),
    ("0", "0"),
])
def test_relaxed_match_accepts(pred, gold):
    assert relaxed_match(pred, gold)


@pytest.mark.parametrize("pred, gold", [
    ("25", "23"),          # 8.7 % off
    ("0.1", "0"),          # gold 0 requires equality
    ("Canada", "USA"),
    ("12 million", "12"),  # text vs number -> exact string match fails
])
def test_relaxed_match_rejects(pred, gold):
    assert not relaxed_match(pred, gold)


def test_relaxed_match_tolerance_boundary():
    assert relaxed_match("105", "100")
    assert not relaxed_match("105.1", "100")


def test_chart_messages():
    ex = {"query": "What is the max?", "label": [42]}
    msgs = chart_messages(ex, False)
    assert len(msgs) == 1 and msgs[0]["content"][0] == {"type": "image"}
    assert msgs[0]["content"][1]["text"] == f"What is the max?\n{INSTRUCTION}"
    assert chart_messages(ex, True)[1]["content"][0]["text"] == "42"
