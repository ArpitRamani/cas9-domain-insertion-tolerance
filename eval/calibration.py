"""Calibration: reliability diagram + Brier score.

A decision-support probability must be calibrated: among sites predicted ~30% tolerant,
about 30% should actually be tolerant. Discrimination (AUPRC) alone isn't enough.
"""
from __future__ import annotations
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss


def reliability_table(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, bins) - 1, 0, n_bins - 1)
    rows = []
    for b in range(n_bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append({
            "bin_lo": bins[b], "bin_hi": bins[b + 1],
            "n": int(m.sum()),
            "mean_pred": float(y_prob[m].mean()),
            "frac_pos": float(y_true[m].mean()),
        })
    return rows


def plot_reliability(curves: dict, out_path: str, n_bins=10):
    """curves: {model_name: (y_true, y_prob)} -> reliability diagram with Brier in legend."""
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect")
    for name, (yt, yp) in curves.items():
        tbl = reliability_table(yt, yp, n_bins)
        xs = [r["mean_pred"] for r in tbl]
        ys = [r["frac_pos"] for r in tbl]
        brier = brier_score_loss(yt, yp)
        ax.plot(xs, ys, "o-", label=f"{name} (Brier={brier:.3f})")
    ax.set_xlabel("mean predicted probability")
    ax.set_ylabel("observed fraction tolerant")
    ax.set_title("Reliability diagram (held-out grouped folds)")
    ax.legend(loc="upper left", fontsize=8)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path
