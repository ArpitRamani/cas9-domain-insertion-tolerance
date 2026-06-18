"""Axis A (room) + Axis C (local structure) + Axis D (architecture) geometry features.

- open_volume_18A        : protein heavy atoms within 18 A of each CA. Low count = open
                           neighbourhood = room for an ~86-aa domain. Larger radius than
                           residue-scale packing on purpose.
- dist_to_sse_end        : within a contiguous DSSP helix/strand, residue distance to the
                           nearest element end (0 = edge or loop). Edges tolerate insertion
                           better than mid-element.
- dist_to_domain_boundary: sequence distance to the nearest Cas9 domain boundary (UniProt
                           Q99ZW2 starts/ends, incl. the REC-NUC linker, an Oakes hotspot).

dist_to_sse_end needs the ss3 column from features/structure.py; run standalone it reads
data/processed/feat_structure.csv.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch, Selection

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.common import (PROTEIN_CHAIN, DOMAIN_BOUNDARIES, protein_residues,
                             load_structure, PROCESSED)

OPEN_VOLUME_RADIUS = 18.0


def _dist_to_sse_end(sites, ss3):
    """For each residue, distance to nearest end of its contiguous H/E run (loops -> 0)."""
    out = {}
    n = len(sites)
    i = 0
    while i < n:
        s = ss3[i]
        if s == "C":
            out[sites[i]] = 0
            i += 1
            continue
        # extend run of same class over contiguous residue numbers
        j = i
        while (j + 1 < n and ss3[j + 1] == s and sites[j + 1] == sites[j] + 1):
            j += 1
        run = sites[i:j + 1]
        L = len(run)
        for k, site in enumerate(run):
            out[site] = min(k, L - 1 - k)
        i = j + 1
    return out


def compute(structure_df: pd.DataFrame | None = None) -> pd.DataFrame:
    model = load_structure()
    res = protein_residues(model)

    # protein heavy atoms for the neighbour search
    chain = model[PROTEIN_CHAIN]
    heavy = [a for a in chain.get_atoms() if a.element != "H"]
    ns = NeighborSearch(heavy)

    rows = []
    for resnum, r in res.items():
        if "CA" in r:
            ca = r["CA"].coord
            n_near = len(ns.search(ca, OPEN_VOLUME_RADIUS, level="A"))
        else:
            n_near = np.nan
        dbound = min(abs(resnum - b) for b in DOMAIN_BOUNDARIES)
        rows.append({
            "site": resnum,
            "open_volume_18A": n_near,
            "dist_to_domain_boundary": dbound,
        })
    df = pd.DataFrame(rows).sort_values("site").reset_index(drop=True)

    # dist_to_sse_end from DSSP ss3
    if structure_df is None:
        sp = os.path.join(PROCESSED, "feat_structure.csv")
        if not os.path.exists(sp):
            raise FileNotFoundError("run features/structure.py first (need ss3 for dist_to_sse_end)")
        structure_df = pd.read_csv(sp)
    sdf = structure_df.sort_values("site").reset_index(drop=True)
    d = _dist_to_sse_end(list(sdf["site"]), list(sdf["ss3"]))
    df["dist_to_sse_end"] = df["site"].map(d)

    print(f"{len(df)} residues; open_volume_18A median={df['open_volume_18A'].median():.0f}")
    return df


if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)
    compute().to_csv(os.path.join(PROCESSED, "feat_geometry.csv"), index=False)
    print("wrote feat_geometry.csv")
