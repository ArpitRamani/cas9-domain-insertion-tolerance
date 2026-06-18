"""Python wrapper around models/bart.R (dbarts). Writes temp CSVs, shells out to Rscript,
reads back posterior mean probability + 95% credible interval per test row.

dbarts handles missing values natively, but we median-impute upstream so both models see
identical feature matrices.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
BART_R = os.path.join(HERE, "bart.R")


def run_bart(X_train, y_train, X_test, feature_names, ndpost=1000, nskip=1000,
             ntree=200, k=2.0, rscript="Rscript"):
    """Returns (pred_df, varcount_df).
    pred_df columns: prob_mean, prob_lo, prob_hi, prob_sd (one row per X_test row).
    ntree/k expose the BART priors for the opt-in tuning bake-off; defaults are the usual
    self-regularizing settings.
    """
    with tempfile.TemporaryDirectory() as d:
        tr = pd.DataFrame(X_train, columns=feature_names)
        tr["label"] = np.asarray(y_train).astype(int)
        te = pd.DataFrame(X_test, columns=feature_names)
        tr_p = os.path.join(d, "train.csv")
        te_p = os.path.join(d, "test.csv")
        out_p = os.path.join(d, "out.csv")
        tr.to_csv(tr_p, index=False)
        te.to_csv(te_p, index=False)
        subprocess.run([rscript, BART_R, tr_p, te_p, out_p, str(ndpost), str(nskip),
                        str(ntree), str(k)], check=True)
        pred = pd.read_csv(out_p)
        vc_path = out_p + ".varcount.csv"
        varcount = pd.read_csv(vc_path) if os.path.exists(vc_path) else None
    return pred, varcount
