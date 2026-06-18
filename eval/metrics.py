"""Ranking-oriented evaluation metrics.

The use case is "pick promising sites", so we lead with ranking metrics:
  - AUPRC (average precision): primary, robust to class imbalance.
  - precision@k: precision among the top-k predicted sites (k = 20, 50).
AUROC and accuracy are secondary; class imbalance makes raw accuracy meaningless.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss


def precision_at_k(y_true, y_score, k):
    y_true = np.asarray(y_true)
    order = np.argsort(-np.asarray(y_score))
    topk = order[:k]
    return float(y_true[topk].sum() / k)


def all_metrics(y_true, y_score, ks=(20, 50)):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    out = {
        "n": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "auprc": float(average_precision_score(y_true, y_score)),
        "auroc": float(roc_auc_score(y_true, y_score)),
        "base_rate": float(y_true.mean()),
    }
    for k in ks:
        if k <= len(y_true):
            out[f"precision_at_{k}"] = precision_at_k(y_true, y_score, k)
    # Brier needs probabilities in [0,1]; only meaningful if y_score is a prob
    if y_score.min() >= 0 and y_score.max() <= 1:
        out["brier"] = float(brier_score_loss(y_true, y_score))
    return out
