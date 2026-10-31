"""
Binary-classification metrics for the prompt-injection guardrail
(`llm-safety/prompt-injection-guardrail`). Label 1 = attack, 0 = benign.

For a guardrail the two numbers that matter most are
    recall              - share of attacks that are caught;
    false-positive rate - share of *benign* prompts that are wrongly blocked
                          (a high FPR makes the product unusable).
"""

import numpy as np


def binary_scores(y, p, thr=0.5):
    """Precision, recall, F1, ROC-AUC, false-positive rate and n.

    y   : true labels (0/1).
    p   : predicted probability of class 1 (attack).
    thr : decision threshold - p >= thr is flagged as an attack.

    ROC-AUC is threshold-free; it is None when only one class is present
    (e.g. an evaluation set with only attacks), because it is undefined then.
    The FPR is None when there are no benign examples.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

    y, p = np.array(y), np.array(p)
    pred = (p >= thr).astype(int)
    benign = y == 0
    return {
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, p) if len(set(y)) > 1 else None,
        "false_positive_rate": float(pred[benign].mean()) if benign.any() else None,
        "n": int(len(y)),
    }
