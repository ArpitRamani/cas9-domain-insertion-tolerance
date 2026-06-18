"""Grouped / blocked cross-validation splits.

Residues are spatially autocorrelated, so random k-fold leaks (adjacent residues land in
both train and test). We split by:
  - "domain" : leave-a-domain-out. Each Cas9 domain held out in turn.
  - "block"  : contiguous sequence blocks. Sort by residue number, cut into n_blocks
               chunks; each chunk is a held-out fold.

Both keep neighbours together. Used for the outer CV and, via nested calls, the inner CV
(LR lambda tuning).
"""
from __future__ import annotations
import numpy as np


def domain_folds(domains):
    """Yield (train_idx, test_idx) leaving out one domain at a time."""
    domains = np.asarray(domains)
    folds = []
    for d in sorted(set(domains)):
        test = np.where(domains == d)[0]
        train = np.where(domains != d)[0]
        if len(test) and len(train):
            folds.append((train, test, str(d)))
    return folds


def block_folds(sites, n_blocks=8):
    """Contiguous-sequence-block folds. sites = residue numbers (any order)."""
    sites = np.asarray(sites)
    order = np.argsort(sites)
    chunks = np.array_split(order, n_blocks)
    folds = []
    for i, test in enumerate(chunks):
        train = np.concatenate([c for j, c in enumerate(chunks) if j != i])
        folds.append((np.sort(train), np.sort(test), f"block{i}"))
    return folds


def make_folds(sites, domains, group_by="domain", n_blocks=8):
    if group_by == "domain":
        return domain_folds(domains)
    elif group_by == "block":
        return block_folds(sites, n_blocks)
    raise ValueError(f"unknown group_by={group_by}")
