"""Metrics that match how a fraud team uses a model.

PR-AUC is the headline ranking metric. Recall at a fixed precision and
precision in the top-k% of scores answer the operational question: "if
analysts can review N alerts, how much fraud do we catch?"
"""

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
)


def recall_at_precision(y_true, y_score, min_precision: float) -> float:
    precision, recall, _ = precision_recall_curve(y_true, y_score)
    ok = precision >= min_precision
    return float(recall[ok].max()) if ok.any() else 0.0


def precision_at_k(y_true, y_score, k_frac: float) -> float:
    y_true = np.asarray(y_true)
    k = max(1, int(round(len(y_true) * k_frac)))
    top = np.argsort(-np.asarray(y_score))[:k]
    return float(y_true[top].mean())


def recall_at_k(y_true, y_score, k_frac: float) -> float:
    y_true = np.asarray(y_true)
    if y_true.sum() == 0:
        return 0.0
    k = max(1, int(round(len(y_true) * k_frac)))
    top = np.argsort(-np.asarray(y_score))[:k]
    return float(y_true[top].sum() / y_true.sum())


def choose_threshold(y_true, y_score) -> float:
    """F1-optimal threshold. Call this on the validation split, never on test."""
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    best = int(np.argmax(f1[:-1]))  # last P/R point has no threshold
    return float(thresholds[best])


def evaluate(y_true, y_score, threshold: float) -> dict:
    y_true = np.asarray(y_true)
    y_pred = (np.asarray(y_score) >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    return {
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "base_rate": float(y_true.mean()),
        "precision_at_threshold": tp / (tp + fp) if tp + fp else 0.0,
        "recall_at_threshold": tp / (tp + fn) if tp + fn else 0.0,
        "recall_at_10pct_precision": recall_at_precision(y_true, y_score, 0.10),
        "precision_at_top_2pct": precision_at_k(y_true, y_score, 0.02),
        "recall_at_top_5pct": recall_at_k(y_true, y_score, 0.05),
        "n": int(len(y_true)),
        "n_fraud": int(y_true.sum()),
    }


def bootstrap_pr_auc(y_true, y_score, n_boot: int = 1000, seed: int = 42) -> tuple:
    """95% bootstrap CI. With a few dozen fraud cases PR-AUC is noisy."""
    rng = np.random.default_rng(seed)
    y_true, y_score = np.asarray(y_true), np.asarray(y_score)
    scores = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y_true), len(y_true))
        if y_true[idx].sum() == 0:
            continue
        scores.append(average_precision_score(y_true[idx], y_score[idx]))
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return float(lo), float(hi)
