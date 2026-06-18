"""Penalized logistic regression: interpretable, naturally calibrated baseline. L2 by
default (elastic-net optional). The exposure cluster is collinear, so we lean on the
penalty rather than dropping columns.

Only the regularization strength (lambda == 1/C) is tuned, via a small grid in the inner
CV loop (no Bayesian optimization).
"""
from __future__ import annotations
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import average_precision_score

DEFAULT_CS = np.logspace(-3, 2, 12)  # C = 1/lambda


def make_pipeline(C=1.0, penalty="l2", l1_ratio=0.5):
    solver = "saga" if penalty == "elasticnet" else "lbfgs"
    kw = {"l1_ratio": l1_ratio} if penalty == "elasticnet" else {}
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=C, penalty=penalty, solver=solver,
                                  max_iter=5000, **kw)),
    ])


def tune_C(X, y, inner_folds, Cs=DEFAULT_CS, penalty="l2", l1_ratio=0.5):
    """inner_folds: list of (train_idx, test_idx) into X/y (grouped). Pick C by mean AUPRC."""
    best_C, best_score = 1.0, -np.inf
    for C in Cs:
        scores = []
        for tr, te, *_ in inner_folds:
            if y[tr].sum() == 0 or y[te].sum() == 0:
                continue
            est = make_pipeline(C, penalty, l1_ratio).fit(X[tr], y[tr])
            p = est.predict_proba(X[te])[:, 1]
            scores.append(average_precision_score(y[te], p))
        if scores and np.mean(scores) > best_score:
            best_score, best_C = np.mean(scores), C
    return best_C, best_score


def fit(X, y, C=1.0, penalty="l2", l1_ratio=0.5):
    return make_pipeline(C, penalty, l1_ratio).fit(X, y)


def coefficients(est, feature_names):
    """Standardized LR coefficients (features are z-scored inside the pipeline)."""
    lr = est.named_steps["lr"]
    return dict(zip(feature_names, lr.coef_.ravel()))
