"""
Information-retrieval metrics, implemented from scratch, for
`rag/retrieval-evaluation-and-finetuning`.

Notation: for one question we have a ranked list of document ids (best first)
and the *set* of relevant ids from the ground-truth labels ("qrels").
All relevance labels are binary here (relevant / not relevant).

    nDCG@10    Discounted cumulative gain of the top 10, divided by the best
               possible DCG. A hit at rank r is worth 1 / log2(r + 1), so hits
               near the top count more. 1.0 = perfect ordering.
    Recall@k   Fraction of the relevant documents that appear in the top k.
               Recall@k bounds what an LLM could ever see in a k-passage prompt.
    MRR@10     1 / rank of the FIRST relevant document (0 if none in the top 10).

Each metric is averaged over questions (macro average), as in BEIR / MTEB.
"""

import re

import numpy as np


def ndcg_at_k(docs, rel, k=10):
    """nDCG@k for one ranked list `docs` and a set of relevant ids `rel`."""
    gains = [1.0 if d in rel else 0.0 for d in docs[:k]]
    # i is 0-based, so rank = i + 1 and the discount is log2(rank + 1) = log2(i + 2).
    dcg = sum(g / np.log2(i + 2) for i, g in enumerate(gains))
    # Ideal DCG: all relevant documents ranked first (at most k of them count).
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg


def recall_at_k(docs, rel, k):
    """Fraction of the relevant ids found in the top k of `docs`."""
    return len(rel & set(docs[:k])) / len(rel)


def mrr_at_k(docs, rel, k=10):
    """Reciprocal rank of the first relevant id within the top k (0.0 if none)."""
    first = next((i for i, d in enumerate(docs[:k]) if d in rel), None)
    return 0.0 if first is None else 1 / (first + 1)


def ir_metrics(ranked, relevant):
    """Average nDCG@10, Recall@10, Recall@100 and MRR@10 over all questions.

    ranked   : {question_id: [doc ids, best first]}
    relevant : {question_id: set(relevant doc ids)} - only questions in here are scored.
    """
    ndcg, r10, r100, mrr = [], [], [], []
    for q, rel in relevant.items():
        docs = ranked[q]
        ndcg.append(ndcg_at_k(docs, rel, 10))
        r10.append(recall_at_k(docs, rel, 10))
        r100.append(recall_at_k(docs, rel, 100))
        mrr.append(mrr_at_k(docs, rel, 10))
    return {
        "nDCG@10": float(np.mean(ndcg)),
        "Recall@10": float(np.mean(r10)),
        "Recall@100": float(np.mean(r100)),
        "MRR@10": float(np.mean(mrr)),
    }


def bm25_tokenize(text):
    """Lower-case word tokens (letters, digits, underscore) - the BM25 baseline's tokenizer."""
    return re.findall(r"\w+", text.lower())
