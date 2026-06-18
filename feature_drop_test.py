"""Test whether dropping a suspected-harmful feature improves the production BART model,
under both CV schemes:

  - domain : leave-a-domain-out (harsh; generalize to a wholly unseen domain)
  - block  : contiguous-sequence blocks (closer to the real use case: scoring unmeasured
             residues interspersed among measured ones across all domains)

The ablation flagged dist_to_domain_boundary (drop1 = -0.042) and backbone_sasa (-0.014)
as hurting under domain-holdout. This re-checks with the real dbarts model and whether
the verdict survives a more deployment-realistic split.

    python feature_drop_test.py    # writes outputs/feature_drop_test.csv
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
from features.common import PROCESSED, OUTPUTS, domain_of
from eval.split import make_folds
from models.bart import run_bart

NDPOST, NSKIP = 500, 500   # lighter draws for the sweep (production uses 1000/1000)


def load():
    feats = pd.read_csv(os.path.join(PROCESSED, "features.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED, "labels.csv"))
    with open(os.path.join(ROOT, "features", "feature_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    prod = [n for n, s in cfg["features"].items()
            if s.get("enabled") and n not in ("sec_struct", "ss3")]
    df = feats.merge(labels[["site", "label", "measured"]], on="site", how="left")
    m = df[df["measured"] == 1].reset_index(drop=True)
    m["domain"] = m["site"].map(domain_of)
    return m, prod, cfg


def bart_oof(m, feat_list, group_by, cfg):
    X = m[feat_list].values.astype(float)
    y = m["label"].values.astype(int)
    folds = make_folds(m["site"].values, m["domain"].values, group_by, cfg["n_blocks"])
    oof = np.full(len(y), np.nan)
    for tr, te, *_ in folds:
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        med = np.nanmedian(X[tr], axis=0)
        def fill(A):
            A = A.copy(); idx = np.where(np.isnan(A)); A[idx] = np.take(med, idx[1]); return A
        pred, _ = run_bart(fill(X[tr]), y[tr], fill(X[te]), feat_list,
                           ndpost=NDPOST, nskip=NSKIP)
        oof[te] = pred["prob_mean"].values
    mk = ~np.isnan(oof)
    return (average_precision_score(y[mk], oof[mk]), roc_auc_score(y[mk], oof[mk]),
            int(mk.sum()))


def main():
    os.makedirs(OUTPUTS, exist_ok=True)
    m, prod, cfg = load()
    ddb, bsa = "dist_to_domain_boundary", "backbone_sasa"
    sets = {
        "production12": prod,
        "minus_dist_to_domain_boundary": [f for f in prod if f != ddb],
        "minus_backbone_sasa": [f for f in prod if f != bsa],
        "minus_both": [f for f in prod if f not in (ddb, bsa)],
    }
    rows = []
    for gb in ("domain", "block"):
        for name, fl in sets.items():
            ap, auc, n = bart_oof(m, fl, gb, cfg)
            rows.append({"cv": gb, "feature_set": name, "n_feats": len(fl),
                         "auprc": round(ap, 3), "auroc": round(auc, 3), "n_oof": n})
            print(f"{gb}: {name:32s} nfeat={len(fl):2d} AUPRC={ap:.3f} AUROC={auc:.3f}")
    tbl = pd.DataFrame(rows)
    out = os.path.join(OUTPUTS, "feature_drop_test.csv")
    tbl.to_csv(out, index=False)
    print("\nDrop test (BART):")
    print(tbl.to_string(index=False))
    print(f"\nwrote {out}")
    print("compare each 'minus_*' row to production12 within the same cv block. "
          "a consistent gain across both cv schemes = a real keep-it-out decision.")


if __name__ == "__main__":
    main()
