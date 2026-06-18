"""BART confirmation for the surviving candidates, under both CV schemes.

From the HGB ablation only two things are worth confirming with the real model:
  - apo_holo_disp  : add1 +0.018 (the one additive winner), but domain-correlated, so the
                     leave-a-domain-out test is the leakage check (it killed
                     dist_to_domain_boundary).
  - channel swap   : replace broad {min_dist_rna, min_dist_dna} with the specific
                     {min_dist_na_ends, min_dist_heteroduplex, min_dist_nontarget}. They
                     don't help as additions (redundant), but min_dist_dna is mildly
                     harmful, so the decomposition might work better as a replacement.

    python feature_add_test.py     # writes outputs/feature_add_test.csv
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

NDPOST, NSKIP = 500, 500
CHANNELS = ["min_dist_na_ends", "min_dist_heteroduplex", "min_dist_nontarget"]
BROAD = ["min_dist_rna", "min_dist_dna"]


def load():
    feats = pd.read_csv(os.path.join(PROCESSED, "features.csv"))
    nv = pd.read_csv(os.path.join(PROCESSED, "feat_novel.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED, "labels.csv"))
    with open(os.path.join(ROOT, "features", "feature_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    prod = [n for n, s in cfg["features"].items()
            if s.get("enabled") and n not in ("sec_struct", "ss3")]
    nv = nv[["site"] + [c for c in nv.columns if c != "site" and c not in feats.columns]]
    df = feats.merge(nv, on="site", how="left").merge(
        labels[["site", "label", "measured"]], on="site", how="left")
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
        pred, _ = run_bart(fill(X[tr]), y[tr], fill(X[te]), feat_list, ndpost=NDPOST, nskip=NSKIP)
        oof[te] = pred["prob_mean"].values
    mk = ~np.isnan(oof)
    return average_precision_score(y[mk], oof[mk]), roc_auc_score(y[mk], oof[mk])


def main():
    os.makedirs(OUTPUTS, exist_ok=True)
    m, prod, cfg = load()
    swap = [f for f in prod if f not in BROAD] + CHANNELS
    sets = {
        "production": prod,
        "production+apo_holo_disp": prod + ["apo_holo_disp"],
        "production_swap_channels": swap,
        "production+apo_holo_disp+swap": swap + ["apo_holo_disp"],
    }
    rows = []
    for gb in ("domain", "block"):
        for name, fl in sets.items():
            ap, auc = bart_oof(m, fl, gb, cfg)
            rows.append({"cv": gb, "feature_set": name, "n_feats": len(fl),
                         "auprc": round(ap, 3), "auroc": round(auc, 3)})
            print(f"{gb}: {name:32s} nfeat={len(fl):2d} AUPRC={ap:.3f} AUROC={auc:.3f}")
    tbl = pd.DataFrame(rows)
    out = os.path.join(OUTPUTS, "feature_add_test.csv")
    tbl.to_csv(out, index=False)
    print("\nAdd/swap test (BART):")
    print(tbl.to_string(index=False))
    print(f"\nwrote {out}")
    print("keep apo_holo_disp only if it helps under both cv (domain = leakage test). "
          "keep swap if it >= production on both.")


if __name__ == "__main__":
    main()
