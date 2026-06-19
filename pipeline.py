"""Assemble features, nested-CV evaluate, calibrate, fit, predict for all 1368 residues.

    python pipeline.py --build-features   # compute feature CSVs (slow: ESM-2 + MAFFT)
    python pipeline.py                    # reuse cached features, train + eval + predict

Outputs land in outputs/: predictions.csv, metrics.json, reliability.png, axis_importance.csv.
"""
from __future__ import annotations
import os
import sys
import json
import argparse
import warnings
import numpy as np
import pandas as pd
import yaml

warnings.simplefilter("ignore")
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from features.common import PROCESSED, OUTPUTS, SEQ_LEN, domain_of
from features import sasa, distances, structure, geometry, conservation, novel
from eval.split import make_folds
from eval.metrics import all_metrics
from eval.calibration import plot_reliability
from models.bart import run_bart

CONFIG_PATH = os.path.join(ROOT, "features", "feature_config.yaml")


def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_features(force=False) -> pd.DataFrame:
    os.makedirs(PROCESSED, exist_ok=True)
    paths = {
        "sasa": (os.path.join(PROCESSED, "feat_sasa.csv"), sasa.compute),
        "distances": (os.path.join(PROCESSED, "feat_distances.csv"), distances.compute),
        "structure": (os.path.join(PROCESSED, "feat_structure.csv"), structure.compute),
    }
    dfs = []
    for name, (p, fn) in paths.items():
        if force or not os.path.exists(p):
            df = fn(); df.to_csv(p, index=False)
        else:
            df = pd.read_csv(p)
        dfs.append(df)

    # geometry needs the structure ss3 column
    gp = os.path.join(PROCESSED, "feat_geometry.csv")
    sdf = pd.read_csv(os.path.join(PROCESSED, "feat_structure.csv"))
    if force or not os.path.exists(gp):
        gdf = geometry.compute(structure_df=sdf); gdf.to_csv(gp, index=False)
    else:
        gdf = pd.read_csv(gp)
    dfs.append(gdf)

    cp = os.path.join(PROCESSED, "feat_conservation.csv")
    if force or not os.path.exists(cp):
        cdf = conservation.compute(); cdf.to_csv(cp, index=False)
    else:
        cdf = pd.read_csv(cp)
    dfs.append(cdf)

    # apo->holo conformational displacement (Axis G). Cheap superposition; the other
    # features in features/novel.py stay experimental (tested via feature_ablation).
    ap = os.path.join(PROCESSED, "feat_apo_holo.csv")
    if force or not os.path.exists(ap):
        adf = novel.apo_holo_displacement(); adf.to_csv(ap, index=False)
    else:
        adf = pd.read_csv(ap)
    dfs.append(adf)

    # master table over all residues 1..1368
    master = pd.DataFrame({"site": np.arange(1, SEQ_LEN + 1)})
    for df in dfs:
        master = master.merge(df, on="site", how="left")
    master["domain"] = master["site"].apply(domain_of)
    out = os.path.join(PROCESSED, "features.csv")
    master.to_csv(out, index=False)
    print(f"assembled {master.shape} -> {out}")
    return master


def enabled_features(config):
    feats = []
    for name, spec in config["features"].items():
        if spec.get("enabled") and name not in ("sec_struct", "ss3"):
            feats.append(name)
    return feats


def axis_of(config, feat):
    return config["features"][feat]["axis"]


# Impute with train medians, applied to train and test (no leakage).
def impute(train_X, *others):
    med = np.nanmedian(train_X, axis=0)
    def fill(A):
        A = A.copy()
        idx = np.where(np.isnan(A))
        A[idx] = np.take(med, idx[1])
        return A
    return (fill(train_X), *[fill(o) for o in others])


def nested_cv(Xm, ym, sites, domains, feat_names, config):
    group_by = config["group_by"]; n_blocks = config["n_blocks"]
    outer = make_folds(sites, domains, group_by, n_blocks)
    oof = np.full(len(ym), np.nan)

    for tr, te, fold_name in outer:
        if ym[tr].sum() == 0 or ym[te].sum() == 0:
            print(f"skip fold {fold_name}: no positives in train or test")
            continue
        # BART, no tuning (self-regularizing via priors)
        Xtr_i, Xte_i = impute(Xm[tr], Xm[te])
        pred, _ = run_bart(Xtr_i, ym[tr], Xte_i, feat_names)
        oof[te] = pred["prob_mean"].values
        print(f"fold {fold_name}: n_test={len(te)} pos_test={int(ym[te].sum())}")

    return oof, outer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-features", action="store_true")
    ap.add_argument("--force-features", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUTPUTS, exist_ok=True)
    config = load_config()
    np.random.seed(config["random_seed"])

    # features
    feat_path = os.path.join(PROCESSED, "features.csv")
    if args.build_features or args.force_features or not os.path.exists(feat_path):
        features = build_features(force=args.force_features)
    else:
        features = pd.read_csv(feat_path)

    # labels
    labels = pd.read_csv(os.path.join(PROCESSED, "labels.csv"))
    feat_names = enabled_features(config)
    print(f"features ({len(feat_names)}): {feat_names}")

    df = features.merge(labels[["site", "label", "fold_change", "measured"]],
                        on="site", how="left")
    df["is_measured"] = df["measured"].fillna(0).astype(int)

    measured = df[df["is_measured"] == 1].copy().reset_index(drop=True)
    Xm = measured[feat_names].values.astype(float)
    ym = measured["label"].values.astype(int)
    sites = measured["site"].values
    domains = measured["domain"].values
    print(f"measured={len(measured)} positives={int(ym.sum())}")

    # nested CV for out-of-fold predictions
    oof, outer = nested_cv(Xm, ym, sites, domains, feat_names, config)

    metrics = {"config": {k: config[k] for k in
                          ["target", "group_by", "n_blocks", "random_seed"]},
               "models": {}}
    mask = ~np.isnan(oof)
    metrics["models"]["bart"] = all_metrics(ym[mask], oof[mask],
                                            ks=tuple(config["precision_at_k"]))
    print(f"bart: {metrics['models']['bart']}")
    plot_reliability({"BART": (ym[mask], oof[mask])},
                     os.path.join(OUTPUTS, "reliability.png"))

    # final fit on all measured, predict all 1368 residues
    Xall = df[feat_names].values.astype(float)
    Xm_i, Xall_i = impute(Xm, Xall)
    pred_all, varcount = run_bart(Xm_i, ym, Xall_i, feat_names)
    df["bart_prob"] = pred_all["prob_mean"].values
    df["bart_lo"] = pred_all["prob_lo"].values
    df["bart_hi"] = pred_all["prob_hi"].values
    df["bart_sd"] = pred_all["prob_sd"].values

    axis_imp = {}
    if varcount is not None:
        for _, row in varcount.iterrows():
            a = axis_of(config, row["feature"])
            axis_imp.setdefault(a, 0.0)
            axis_imp[a] += row["inclusion"]

    # OOF preds back onto measured rows for the table (measured sites only)
    df["oof_bart_prob"] = np.nan
    midx = df.index[df["is_measured"] == 1]
    df.loc[midx, "oof_bart_prob"] = oof

    df["in_prediction_set"] = (df["is_measured"] == 0).astype(int)

    # Support / abstention flag: a residue is out-of-support if any feature falls outside the
    # measured set's 1st-99th percentile envelope. The model trains on measured sites but scores
    # all 1368, so out-of-support rows are extrapolation and BART's interval understates their
    # uncertainty -- a user should abstain on or down-weight them. (NaN counts as in-support;
    # those features are median-imputed.)
    Xmeas = df.loc[df["is_measured"] == 1, feat_names].values.astype(float)
    lo = np.nanpercentile(Xmeas, 1, axis=0)
    hi = np.nanpercentile(Xmeas, 99, axis=0)
    Xraw = df[feat_names].values.astype(float)
    oos = np.where(np.isnan(Xraw), False, (Xraw < lo) | (Xraw > hi))
    df["n_features_out"] = oos.sum(axis=1).astype(int)
    df["in_support"] = (df["n_features_out"] == 0).astype(int)
    n_dep_oos = int(((df["in_prediction_set"] == 1) & (df["in_support"] == 0)).sum())
    print(f"support flag: {n_dep_oos} of {int(df['in_prediction_set'].sum())} prediction-set "
          f"residues out-of-support")

    # write outputs
    cols = (["site", "domain", "is_measured", "in_prediction_set", "label", "fold_change",
             "bart_prob", "bart_lo", "bart_hi", "bart_sd", "oof_bart_prob",
             "in_support", "n_features_out"]
            + feat_names)
    pred_table = df[cols].sort_values("site")
    pred_path = os.path.join(OUTPUTS, "predictions.csv")
    pred_table.to_csv(pred_path, index=False)
    print(f"wrote {pred_path}  ({len(pred_table)} residues, "
          f"{int(df['in_prediction_set'].sum())} in prediction set)")

    # axis importance (BART variable-inclusion), normalized
    ai = pd.DataFrame([{"axis": a, "bart_importance": v}
                       for a, v in sorted(axis_imp.items())])
    if ai["bart_importance"].sum() > 0:
        ai["bart_importance"] = ai["bart_importance"] / ai["bart_importance"].sum()
    ai.to_csv(os.path.join(OUTPUTS, "axis_importance.csv"), index=False)
    metrics["axis_importance"] = ai.to_dict(orient="records")

    with open(os.path.join(OUTPUTS, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("wrote metrics.json, reliability.png, axis_importance.csv")
    print("\nAxis importance:")
    print(ai.to_string(index=False))


if __name__ == "__main__":
    main()
