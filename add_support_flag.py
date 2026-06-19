"""Add a support / abstention flag to the deliverable (outputs/predictions.csv), in place.

The model trains on the 635 measured sites but scores all 1368; the 733 unmeasured residues are
the product output. For each residue we mark whether it lies inside the measured feature support:

  in_support      = 0 if any production feature falls outside the measured set's 1st-99th
                    percentile envelope (the same out-of-envelope test as covariate_shift.py),
                    else 1. NaN counts as in-support (those features are median-imputed).
  n_features_out  = how many features are outside the envelope.

Out-of-support rows are extrapolation: BART's credible interval reflects posterior uncertainty
given the training support, not out-of-support uncertainty, so a user should abstain on or
down-weight them. This applies the flag without re-running BART, so the probabilities are
unchanged; pipeline.py also emits these columns on a full run.

    python add_support_flag.py
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)
from features.common import OUTPUTS


def main():
    pred_path = os.path.join(OUTPUTS, "predictions.csv")
    pred = pd.read_csv(pred_path)
    with open(os.path.join(ROOT, "features", "feature_config.yaml")) as f:
        cfg = yaml.safe_load(f)
    feat_names = [n for n, s in cfg["features"].items()
                  if s.get("enabled") and n not in ("sec_struct", "ss3")]
    missing = [f for f in feat_names if f not in pred.columns]
    if missing:
        raise SystemExit(f"predictions.csv is missing feature columns: {missing}")

    Xmeas = pred.loc[pred["is_measured"] == 1, feat_names].values.astype(float)
    lo = np.nanpercentile(Xmeas, 1, axis=0)
    hi = np.nanpercentile(Xmeas, 99, axis=0)
    Xraw = pred[feat_names].values.astype(float)
    oos = np.where(np.isnan(Xraw), False, (Xraw < lo) | (Xraw > hi))

    pred = pred.drop(columns=[c for c in ("in_support", "n_features_out") if c in pred.columns])
    pred["n_features_out"] = oos.sum(axis=1).astype(int)
    pred["in_support"] = (pred["n_features_out"] == 0).astype(int)

    # reorder: put the two flags right after the BART interval columns
    cols = list(pred.columns)
    for c in ("in_support", "n_features_out"):
        cols.remove(c)
    anchor = cols.index("oof_bart_prob") + 1 if "oof_bart_prob" in cols else len(cols)
    cols[anchor:anchor] = ["in_support", "n_features_out"]
    pred = pred[cols]
    pred.to_csv(pred_path, index=False)

    dep = pred[pred["in_prediction_set"] == 1]
    n_oos = int((dep["in_support"] == 0).sum())
    print(f"added in_support / n_features_out to {pred_path}")
    print(f"deployment set (n={len(dep)}): out-of-support = {n_oos} ({n_oos/len(dep):.1%}); "
          f"abstain on these or treat their intervals as lower bounds.")


if __name__ == "__main__":
    main()
