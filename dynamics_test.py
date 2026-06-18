"""Honest dual-CV test of the ANM dynamics feature (anm_msf), under the real BART model.

Same harness that vetted apo_holo_disp. Four feature sets, scored under both CV schemes:
  - production (apo_holo)    : the current 10-feature set (includes apo_holo_disp)
  - production + anm_msf      : does ANM add on top of apo_holo_disp?
  - swap apo_holo -> anm_msf  : is ANM a better single dynamics feature?
  - no dynamics               : dynamics-free reference

A feature earns its place only if it helps under BOTH CV schemes; the leave-a-domain-out
column is the leakage check (a benefit there, neutral on block, is genuine transferable signal).

    python dynamics_test.py    # writes outputs/dynamics_test.csv
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


def load():
    feats = pd.read_csv(os.path.join(PROCESSED, "features.csv"))
    dyn = pd.read_csv(os.path.join(PROCESSED, "feat_dynamics.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED, "labels.csv"))
    with open(os.path.join(ROOT, "features", "feature_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    prod = [n for n, s in cfg["features"].items()
            if s.get("enabled") and n not in ("sec_struct", "ss3")]
    dyn = dyn[["site"] + [c for c in dyn.columns if c != "site" and c not in feats.columns]]
    df = feats.merge(dyn, on="site", how="left").merge(
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
    no_dyn = [f for f in prod if f != "apo_holo_disp"]
    sets = {
        "production (apo_holo)":   prod,
        "production + anm_msf":     prod + ["anm_msf"],
        "swap apo_holo->anm_msf":   no_dyn + ["anm_msf"],
        "no dynamics":              no_dyn,
    }
    # quick correlation between the two dynamics features (resolved residues only)
    both = m[["apo_holo_disp", "anm_msf"]].dropna()
    if len(both) > 2:
        r = np.corrcoef(both["apo_holo_disp"], both["anm_msf"])[0, 1]
        print(f"corr(apo_holo_disp, anm_msf) = {r:+.3f} over {len(both)} measured residues\n")

    rows = []
    for gb in ("domain", "block"):
        for name, fl in sets.items():
            ap, auc = bart_oof(m, fl, gb, cfg)
            rows.append({"cv": gb, "feature_set": name, "n_feats": len(fl),
                         "auprc": round(ap, 3), "auroc": round(auc, 3)})
            print(f"{gb}: {name:26s} nfeat={len(fl):2d} AUPRC={ap:.3f} AUROC={auc:.3f}")
    tbl = pd.DataFrame(rows)
    out = os.path.join(OUTPUTS, "dynamics_test.csv")
    tbl.to_csv(out, index=False)
    print("\nANM dynamics test (BART, dual CV):")
    print(tbl.to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
