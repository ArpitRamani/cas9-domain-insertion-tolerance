"""Does tuning BART buy anything here? A small k x ntree grid under nested CV.

For each selection objective (AUPRC and AUROC) we report:
  - inner_best   : mean over outer folds of the best inner-CV score the grid found
  - honest_auprc / honest_auroc : pooled out-of-fold metrics (what we actually get)
  - optimism_gap : inner_best - honest(selected metric). Larger = more overfit to the
                   noisy CV objective.

BART self-regularizes through its priors, so the expectation is a small grid effect and a
small optimism gap. Run:

    python tuning_bakeoff.py
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from features.common import PROCESSED, OUTPUTS
from eval.split import make_folds
from models.bart import run_bart

METRIC_FN = {"auprc": average_precision_score, "auroc": roc_auc_score}

BART_GRID = [(k, nt) for k in (1.0, 2.0, 3.0) for nt in (50, 200)]
TUNE_NDPOST, TUNE_NSKIP = 200, 200   # light draws for the inner selection loop
FINAL_NDPOST, FINAL_NSKIP = 1000, 1000


def load_data():
    feats = pd.read_csv(os.path.join(PROCESSED, "features.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED, "labels.csv"))
    with open(os.path.join(ROOT, "features", "feature_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    fnames = [n for n, s in cfg["features"].items()
              if s.get("enabled") and n not in ("sec_struct", "ss3")]
    df = feats.merge(labels[["site", "label", "measured"]], on="site", how="left")
    m = df[df["measured"] == 1].reset_index(drop=True)
    return (m[fnames].values.astype(float), m["label"].values.astype(int),
            m["site"].values, m["domain"].values, fnames, cfg)


def impute(tr, *others):
    med = np.nanmedian(tr, axis=0)
    def f(A):
        A = A.copy(); idx = np.where(np.isnan(A)); A[idx] = np.take(med, idx[1]); return A
    return (f(tr), *[f(o) for o in others])


def bart_inner_score(k, ntree, X, y, feat, inner, metric):
    fn = METRIC_FN[metric]; sc = []
    for tr, te, *_ in inner:
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        Xtr, Xte = impute(X[tr], X[te])
        pred, _ = run_bart(Xtr, y[tr], Xte, feat, ndpost=TUNE_NDPOST, nskip=TUNE_NSKIP,
                           ntree=ntree, k=k)
        sc.append(fn(y[te], pred["prob_mean"].values))
    return float(np.mean(sc)) if sc else -1.0


def bakeoff_bart(X, y, sites, domains, feat, cfg):
    print("small k x ntree grid (default is k=2, ntree=200).")
    outer = make_folds(sites, domains, cfg["group_by"], cfg["n_blocks"])
    rows = []
    for metric in ("auprc", "auroc"):
        oof = np.full(len(y), np.nan); inner_bests = []
        for tr, te, fold in outer:
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            inner = make_folds(sites[tr], domains[tr], cfg["group_by"], cfg["n_blocks"])
            best, bs = None, -np.inf
            for k, nt in BART_GRID:
                s = bart_inner_score(k, nt, X[tr], y[tr], feat, inner, metric)
                if s > bs:
                    bs, best = s, (k, nt)
            inner_bests.append(bs)
            Xtr, Xte = impute(X[tr], X[te])
            pred, _ = run_bart(Xtr, y[tr], Xte, feat, ndpost=FINAL_NDPOST, nskip=FINAL_NSKIP,
                               ntree=best[1], k=best[0])
            oof[te] = pred["prob_mean"].values
        mask = ~np.isnan(oof)
        ha = average_precision_score(y[mask], oof[mask])
        hr = roc_auc_score(y[mask], oof[mask])
        ib = float(np.mean(inner_bests))
        honest_sel = ha if metric == "auprc" else hr
        rows.append({"model": "BART", "tuner": "BART-grid", "select_on": metric,
                     "inner_best": round(ib, 3), "honest_auprc": round(ha, 3),
                     "honest_auroc": round(hr, 3), "optimism_gap": round(ib - honest_sel, 3)})
        print(f"bart grid select={metric}: inner_best={ib:.3f} "
              f"honest_auprc={ha:.3f} honest_auroc={hr:.3f} gap={ib-honest_sel:+.3f}")
    return rows


def main():
    os.makedirs(OUTPUTS, exist_ok=True)
    X, y, sites, domains, feat, cfg = load_data()
    print(f"measured={len(y)} positives={int(y.sum())} features={len(feat)}")

    rows = bakeoff_bart(X, y, sites, domains, feat, cfg)

    tbl = pd.DataFrame(rows)
    out = os.path.join(OUTPUTS, "tuning_comparison.csv")
    tbl.to_csv(out, index=False)
    print("\nTuning bake-off:")
    print(tbl.to_string(index=False))
    print(f"\nwrote {out}")
    print("\na small optimism_gap means the grid's inner-CV estimate matches the honest\n"
          "out-of-fold score, i.e. the model generalizes faithfully and tuning is not the lever.")


if __name__ == "__main__":
    main()
