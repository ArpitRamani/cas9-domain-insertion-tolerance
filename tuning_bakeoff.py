"""Tuning bake-off: small grid vs Bayesian optimization (Optuna TPE), under identical
nested CV, to test whether BO overfits in this regime.

Per tuner, per selection objective (AUPRC and AUROC):
  - inner_best   : mean over outer folds of the best inner-CV score the tuner saw
  - honest_auprc / honest_auroc : pooled out-of-fold metrics (what we actually get)
  - optimism_gap : inner_best - honest(selected metric). Larger = more overfit to the
                   noisy CV objective.

LR gets the full grid-vs-BO comparison (cheap). BART is opt-in (--bart) with a small grid
only; full BO on BART is hundreds of dbarts fits.

    python tuning_bakeoff.py            # LR grid vs BO, both objectives
    python tuning_bakeoff.py --bart     # also a small BART k x ntree grid (slow)
"""
from __future__ import annotations
import os
import sys
import argparse
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
from models import logreg
from models.bart import run_bart

METRIC_FN = {"auprc": average_precision_score, "auroc": roc_auc_score}
SEED = 0


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


# LR scorers
def lr_inner_score(C, l1, X, y, inner, metric):
    fn = METRIC_FN[metric]; sc = []
    for tr, te, *_ in inner:
        if y[tr].sum() == 0 or y[te].sum() == 0:
            continue
        est = logreg.make_pipeline(C, "elasticnet", l1).fit(X[tr], y[tr])
        sc.append(fn(y[te], est.predict_proba(X[te])[:, 1]))
    return float(np.mean(sc)) if sc else -1.0


LR_GRID = [(C, l1) for C in np.logspace(-3, 2, 6) for l1 in (0.0, 0.5, 1.0)]


def lr_tune_grid(X, y, inner, metric):
    best, bs = None, -np.inf
    for C, l1 in LR_GRID:
        s = lr_inner_score(C, l1, X, y, inner, metric)
        if s > bs:
            bs, best = s, (C, l1)
    return best, bs


def lr_tune_bo(X, y, inner, metric, n_trials=40):
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def obj(t):
        C = 10 ** t.suggest_float("logC", -3, 2)
        l1 = t.suggest_float("l1", 0.0, 1.0)
        return lr_inner_score(C, l1, X, y, inner, metric)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj, n_trials=n_trials, show_progress_bar=False)
    p = study.best_params
    return (10 ** p["logC"], p["l1"]), study.best_value


def lr_fit_predict(cfg, Xtr, ytr, Xte):
    C, l1 = cfg
    est = logreg.make_pipeline(C, "elasticnet", l1).fit(Xtr, ytr)
    return est.predict_proba(Xte)[:, 1]


# BART small grid (opt-in; BART self-regularizes)
BART_GRID = [(k, nt) for k in (1.0, 2.0, 3.0) for nt in (50, 200)]
TUNE_NDPOST, TUNE_NSKIP = 200, 200   # light draws for the inner selection loop
FINAL_NDPOST, FINAL_NSKIP = 1000, 1000


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
    print("small k x ntree grid (slow; opt-in). default is k=2, ntree=200.")
    outer = make_folds(sites, domains, cfg["group_by"], cfg["n_blocks"])
    rows = []
    # grid, selecting on each metric
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


def bakeoff_lr(X, y, sites, domains, cfg):
    outer = make_folds(sites, domains, cfg["group_by"], cfg["n_blocks"])
    rows = []
    for metric in ("auprc", "auroc"):
        for tuner_name, tuner in (("LR-grid", lr_tune_grid), ("LR-BO", lr_tune_bo)):
            oof = np.full(len(y), np.nan); inner_bests = []
            for tr, te, fold in outer:
                if y[tr].sum() == 0 or y[te].sum() == 0:
                    continue
                inner = make_folds(sites[tr], domains[tr], cfg["group_by"], cfg["n_blocks"])
                best_cfg, best_inner = tuner(X[tr], y[tr], inner, metric)
                inner_bests.append(best_inner)
                oof[te] = lr_fit_predict(best_cfg, X[tr], y[tr], X[te])
            mask = ~np.isnan(oof)
            ha = average_precision_score(y[mask], oof[mask])
            hr = roc_auc_score(y[mask], oof[mask])
            ib = float(np.mean(inner_bests))
            honest_sel = ha if metric == "auprc" else hr
            rows.append({"model": "LR", "tuner": tuner_name, "select_on": metric,
                         "inner_best": round(ib, 3), "honest_auprc": round(ha, 3),
                         "honest_auroc": round(hr, 3),
                         "optimism_gap": round(ib - honest_sel, 3)})
            print(f"lr {tuner_name} select={metric}: inner_best={ib:.3f} "
                  f"honest_auprc={ha:.3f} honest_auroc={hr:.3f} gap={ib-honest_sel:+.3f}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bart", action="store_true", help="also run the small BART k x ntree grid (slow)")
    args = ap.parse_args()
    os.makedirs(OUTPUTS, exist_ok=True)
    X, y, sites, domains, feat, cfg = load_data()
    print(f"measured={len(y)} positives={int(y.sum())} features={len(feat)}")

    rows = bakeoff_lr(X, y, sites, domains, cfg)

    if args.bart:
        rows += bakeoff_bart(X, y, sites, domains, feat, cfg)

    tbl = pd.DataFrame(rows)
    out = os.path.join(OUTPUTS, "tuning_comparison.csv")
    tbl.to_csv(out, index=False)
    print("\nTuning bake-off:")
    print(tbl.to_string(index=False))
    print(f"\nwrote {out}")
    print("\nif LR-BO's honest score is <= LR-grid's and its optimism_gap is larger,\n"
          "BO overfit the noisy CV objective without buying generalization.")


if __name__ == "__main__":
    main()
