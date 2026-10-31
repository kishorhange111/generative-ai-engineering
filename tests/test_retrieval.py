"""Tests for the from-scratch IR metrics."""

import math

import pytest

from genai_common.retrieval import bm25_tokenize, ir_metrics, mrr_at_k, ndcg_at_k, recall_at_k


def test_ndcg_perfect_and_zero():
    assert ndcg_at_k(["a", "b", "c"], {"a", "b"}) == pytest.approx(1.0)
    assert ndcg_at_k(["x", "y", "z"], {"a"}) == 0.0


def test_ndcg_hand_computed():
    # single relevant doc at rank 2: DCG = 1/log2(3), IDCG = 1/log2(2) = 1
    assert ndcg_at_k(["x", "a"], {"a"}) == pytest.approx(1 / math.log2(3))
    # two relevant docs at ranks 1 and 3
    expected = (1 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert ndcg_at_k(["a", "x", "b"], {"a", "b"}) == pytest.approx(expected)


def test_ndcg_ignores_hits_beyond_k():
    docs = [f"x{i}" for i in range(10)] + ["a"]
    assert ndcg_at_k(docs, {"a"}, k=10) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "x", "b"], {"a", "b", "c", "d"}, 2) == 0.25
    assert recall_at_k(["a", "x", "b"], {"a", "b"}, 3) == 1.0


def test_mrr_at_k():
    assert mrr_at_k(["a"], {"a"}) == 1.0
    assert mrr_at_k(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert mrr_at_k(["x"] * 10 + ["a"], {"a"}) == 0.0


def test_ir_metrics_macro_average():
    ranked = {"q1": ["a", "x"], "q2": ["y", "b"], "q3": ["z"]}
    relevant = {"q1": {"a"}, "q2": {"b"}}  # q3 has no labels and is not scored
    m = ir_metrics(ranked, relevant)
    assert set(m) == {"nDCG@10", "Recall@10", "Recall@100", "MRR@10"}
    assert m["MRR@10"] == pytest.approx((1 + 0.5) / 2)
    assert m["Recall@10"] == 1.0
    assert m["nDCG@10"] == pytest.approx((1 + 1 / math.log2(3)) / 2)
    assert all(isinstance(v, float) for v in m.values())


def test_bm25_tokenize():
    assert bm25_tokenize("What's the P/E ratio of AAPL, 2024?") == [
        "what",
        "s",
        "the",
        "p",
        "e",
        "ratio",
        "of",
        "aapl",
        "2024",
    ]
