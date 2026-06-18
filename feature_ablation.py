"""Feature ablation. For every feature (the 12 production features + the novel candidates
in features/novel.py), measure under identical leave-a-domain-out grouped CV:

  - univariate   : OOF AUPRC/AUROC using only that feature
  - drop1_dAUPRC : AUPRC(full) - AUPRC(full minus this feature). >0 the feature helps,
                   <0 it's detrimental/redundant.
  - add1_dAUPRC  : (novel only) AUPRC(production12 + feature) - AUPRC(production12), the
                   marginal value on top of the production set.

HistGradientBoostingClassifier is a fast, NaN-native, tree-based proxy for BART for
ranking features. --confirm-bart re-checks production12 vs production12 + winners with
the real dbarts model.

    python feature_ablation.py
    python feature_ablation.py --confirm-bart
"""
from __future__ import annotations
import os
import sys
import argparse
import warnings
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from features.common import PROCESSED, OUTPUTS, domain_of
from features import novel
from eval.split import make_folds

SEED = 0


def hgb():
    return HistGradientBoostingClassifier(
        max_iter=200, max_leaf_nodes=15, min_samples_leaf=20,
        learning_rate=0.08, l2_regularization=1.0, random_state=SEED)


def load():
    feats = pd.read_csv(os.path.join(PROCESSED, "features.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED, "labels.csv"))
    nv_path = os.path.join(PROCESSED, "feat_novel.csv")
    nv = pd.read_csv(nv_path) if os.path.exists(nv_path) else novel.compute()
    with open(os.path.join(ROOT, "features", "feature_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    prod = [n for n, s in cfg["features"].items()
            if s.get("enabled") and n not in ("sec_struct", "ss3")]
    novel_feats = [c for c in nv.columns if c != "site"]
    nv = nv[["site"] + [c for c in nv.columns if c != "site" and c not in feats.columns]]
    df = feats.merge(nv, on="site", how="left").merge(
        labels[["site", "label", "measured"]], on="site", how="left")
    m = df[df["measured"] == 1].reset_index(drop=True)
    return m, prod, novel_feats, cfg


def oof_scores(m, feat_list, cfg):
    """Out-of-fold AUPRC/AUROC for a HGB trained on feat_list under grouped CV."""
    X = m[feat_list].values.astype(float)
    y = m["label"].values.astype(int)
    sites = m["site"].values
    domains = m["domain"].values if "domain" in m else m["site"].map(domain_of).values
    folds = make_folds(sites, domains, cfg["group_by"], cfg["n_blocks"])
    oof = np.full(len(y), np.nan)
    for tr, te, *_ in folds:
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        oof[te] = hgb().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    mask = ~np.isnan(oof)
    return (average_precision_score(y[mask], oof[mask]),
            roc_auc_score(y[mask], oof[mask]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confirm-bart", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTPUTS, exist_ok=True)

    m, prod, novel_feats, cfg = load()
    full = prod + novel_feats
    if "domain" not in m:
        m["domain"] = m["site"].map(domain_of)
    print(f"measured={len(m)} pos={int(m['label'].sum())} "
          f"production={len(prod)} novel={novel_feats}")

    base_ap, base_auc = oof_scores(m, prod, cfg)
    full_ap, full_auc = oof_scores(m, full, cfg)
    print(f"production12   AUPRC={base_ap:.3f} AUROC={base_auc:.3f}")
    print(f"production+novel AUPRC={full_ap:.3f} AUROC={full_auc:.3f}")

    rows = []
    for f in full:
        uap, uauc = oof_scores(m, [f], cfg)
        dap, _ = oof_scores(m, [x for x in full if x != f], cfg)
        row = {"feature": f, "group": "novel" if f in novel_feats else "production",
               "univariate_auprc": round(uap, 3), "univariate_auroc": round(uauc, 3),
               "drop1_dAUPRC": round(full_ap - dap, 3)}
        if f in novel_feats:
            aap, _ = oof_scores(m, prod + [f], cfg)
            row["add1_dAUPRC"] = round(aap - base_ap, 3)
        rows.append(row)

    tbl = pd.DataFrame(rows).sort_values(["group", "drop1_dAUPRC"], ascending=[True, False])
    out = os.path.join(OUTPUTS, "feature_ablation.csv")
    tbl.to_csv(out, index=False)
    print("\nFeature ablation (HistGradientBoosting, leave-a-domain-out OOF):")
    print(f"baseline production12 AUPRC={base_ap:.3f} | +all novel AUPRC={full_ap:.3f}\n")
    print(tbl.to_string(index=False))
    print(f"\nwrote {out}")
    print("drop1_dAUPRC>0 => feature helps; <0 => detrimental/redundant. "
          "add1_dAUPRC = novel feature's marginal value over the production set.")

    if args.confirm_bart:
        from models.bart import run_bart
        winners = [r["feature"] for r in rows
                   if r["group"] == "novel" and r.get("add1_dAUPRC", 0) > 0]
        print(f"\nnovel winners (add1>0): {winners}")
        for name, fl in (("production12", prod), ("production12+winners", prod + winners)):
            ap, auc = bart_oof(m, fl, cfg, run_bart)
            print(f"{name}: AUPRC={ap:.3f} AUROC={auc:.3f}")


def bart_oof(m, feat_list, cfg, run_bart):
    X = m[feat_list].values.astype(float)
    y = m["label"].values.astype(int)
    sites = m["site"].values
    domains = m["domain"].values
    folds = make_folds(sites, domains, cfg["group_by"], cfg["n_blocks"])
    oof = np.full(len(y), np.nan)
    for tr, te, *_ in folds:
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        med = np.nanmedian(X[tr], axis=0)
        def fill(A):
            A = A.copy(); idx = np.where(np.isnan(A)); A[idx] = np.take(med, idx[1]); return A
        pred, _ = run_bart(fill(X[tr]), y[tr], fill(X[te]), feat_list, ndpost=500, nskip=500)
        oof[te] = pred["prob_mean"].values
    mask = ~np.isnan(oof)
    return (average_precision_score(y[mask], oof[mask]),
            roc_auc_score(y[mask], oof[mask]))


if __name__ == "__main__":
    main()
