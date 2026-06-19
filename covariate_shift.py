"""Covariate-shift probe: are the 733 unmeasured residues (the product output) drawn from
the same feature distribution as the 635 measured sites the model is evaluated on?

The model is scored on the measured set but deployed on the unmeasured set. If the two
differ in feature space, the headline metrics overstate deployment quality and BART's
credible intervals (posterior-given-support, not out-of-support) understate the real risk.

Three measurements, all on the production 10-feature set, computed on features already on disk:

  1. SHIFT MAGNITUDE  : 5-fold CV AUC of an HGB classifier trained to tell measured (1) from
                        unmeasured (0). ~0.5 = exchangeable; high = strong shift.
  2. PER-FEATURE      : standardized mean difference (unmeasured - measured) / pooled SD, and
                        a KS statistic, per feature; shows where the shift lives.
  3. OUT-OF-SUPPORT   : fraction of the 733 with >=1 feature outside the measured set's
                        1st-99th percentile envelope, i.e. literal extrapolation.

The 733 unmeasured residues are two sub-populations: never-screened, and screened-but-not-
significant. We can only separate them if a coverage flag exists; otherwise this treats the
prediction set as a whole (the conservative reading).

    python covariate_shift.py     # writes outputs/covariate_shift.csv, prints a summary
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ks_2samp
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import roc_auc_score

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from features.common import PROCESSED, OUTPUTS

SEED = 0


def load():
    feats = pd.read_csv(os.path.join(PROCESSED, "features.csv"))
    labels = pd.read_csv(os.path.join(PROCESSED, "labels.csv"))
    with open(os.path.join(ROOT, "features", "feature_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    fnames = [n for n, s in cfg["features"].items()
              if s.get("enabled") and n not in ("sec_struct", "ss3")]
    df = feats.merge(labels[["site", "measured"]], on="site", how="left")
    df["measured"] = df["measured"].fillna(0).astype(int)
    return df, fnames


def main():
    os.makedirs(OUTPUTS, exist_ok=True)
    df, fnames = load()
    meas = df[df["measured"] == 1]
    pred = df[df["measured"] == 0]
    Xm = meas[fnames].values.astype(float)
    Xp = pred[fnames].values.astype(float)
    print(f"measured (eval) n={len(meas)} | unmeasured (deployment) n={len(pred)} | "
          f"features={len(fnames)}")

    # ---- 1. shift magnitude: classify measured vs unmeasured from features ----
    X = df[fnames].values.astype(float)
    y = df["measured"].values.astype(int)  # 1 = measured, 0 = unmeasured
    clf = HistGradientBoostingClassifier(random_state=SEED)
    proba = cross_val_predict(clf, X, y, cv=5, method="predict_proba")[:, 1]
    shift_auc = roc_auc_score(y, proba)
    separability = max(shift_auc, 1.0 - shift_auc)   # AUC is direction-symmetric; this is the
    print(f"\n[1] shift-classifier AUC for P(measured) = {shift_auc:.3f}; "          # magnitude
          f"separability = {separability:.3f}  (0.5 = no shift)")

    # ---- 2. per-feature standardized mean difference + KS ----
    rows = []
    for j, f in enumerate(fnames):
        a = Xm[:, j][~np.isnan(Xm[:, j])]
        b = Xp[:, j][~np.isnan(Xp[:, j])]
        pooled = np.sqrt((np.nanvar(a) + np.nanvar(b)) / 2.0) or np.nan
        smd = (np.nanmean(b) - np.nanmean(a)) / pooled if pooled else np.nan
        ks = ks_2samp(a, b).statistic if len(a) and len(b) else np.nan
        rows.append({"feature": f, "measured_median": round(float(np.nanmedian(a)), 3),
                     "unmeasured_median": round(float(np.nanmedian(b)), 3),
                     "std_mean_diff": round(float(smd), 3), "ks_stat": round(float(ks), 3)})
    perfeat = pd.DataFrame(rows).sort_values("ks_stat", ascending=False)
    print("\n[2] per-feature shift (sorted by KS; std_mean_diff = (unmeas - meas)/pooled SD):")
    print(perfeat.to_string(index=False))

    # ---- 3. out-of-support fraction (1st-99th pct envelope of measured set) ----
    lo = np.nanpercentile(Xm, 1, axis=0)
    hi = np.nanpercentile(Xm, 99, axis=0)
    oos_mask = np.zeros(len(Xp), dtype=bool)
    per_feat_oos = {}
    for j, f in enumerate(fnames):
        col = Xp[:, j]
        out = (col < lo[j]) | (col > hi[j])
        out = np.where(np.isnan(col), False, out)
        per_feat_oos[f] = float(np.mean(out))
        oos_mask |= out
    oos_frac = float(np.mean(oos_mask))
    print(f"\n[3] out-of-support: {oos_mask.sum()}/{len(Xp)} = {oos_frac:.1%} of the unmeasured "
          f"set has >=1 feature outside the measured 1st-99th pct envelope")
    top_oos = sorted(per_feat_oos.items(), key=lambda kv: -kv[1])[:3]
    print("    top contributors: " + ", ".join(f"{k} {v:.1%}" for k, v in top_oos))

    perfeat.to_csv(os.path.join(OUTPUTS, "covariate_shift.csv"), index=False)
    with open(os.path.join(OUTPUTS, "covariate_shift_summary.txt"), "w") as fh:
        fh.write(f"shift_auc_p_measured={shift_auc:.3f}\n")
        fh.write(f"separability={separability:.3f}\n")
        fh.write(f"out_of_support_frac={oos_frac:.3f} ({oos_mask.sum()}/{len(Xp)})\n")
        fh.write(f"top_oos={top_oos}\n")
    print("\nwrote outputs/covariate_shift.csv and covariate_shift_summary.txt")
    print("reading: AUC near 0.5 and OOS near 0 => eval and deployment sets are exchangeable; "
          "high AUC / large OOS => the product scores are partly extrapolation.")


if __name__ == "__main__":
    main()
