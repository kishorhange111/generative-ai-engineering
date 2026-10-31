"""Tests for the guardrail's binary-classification metrics."""

import pytest

pytest.importorskip("sklearn")

from genai_common.classification import binary_scores  # noqa: E402


def test_binary_scores_hand_computed():
    y = [1, 1, 1, 0, 0, 0, 0]
    p = [0.9, 0.6, 0.2, 0.7, 0.1, 0.3, 0.4]  # TP=2, FN=1, FP=1, TN=3
    s = binary_scores(y, p)
    assert s["precision"] == pytest.approx(2 / 3)
    assert s["recall"] == pytest.approx(2 / 3)
    assert s["f1"] == pytest.approx(2 / 3)
    assert s["false_positive_rate"] == pytest.approx(1 / 4)
    assert s["n"] == 7
    assert 0.0 <= s["roc_auc"] <= 1.0


def test_threshold_is_inclusive():
    assert binary_scores([1, 0], [0.5, 0.49])["recall"] == 1.0


def test_single_class_sets():
    only_attacks = binary_scores([1, 1], [0.9, 0.1])
    assert only_attacks["roc_auc"] is None and only_attacks["false_positive_rate"] is None
    only_benign = binary_scores([0, 0], [0.9, 0.1])
    assert only_benign["false_positive_rate"] == 0.5 and only_benign["recall"] == 0.0
