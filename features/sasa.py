"""Axis A (exposure/room) SASA features, computed on 4UN3 with FreeSASA.

- rel_sasa      : observed ASA / Tien-2013 max-ASA. Kept continuous (no buried/exposed
                  threshold; the 20-25% cutoff is just a convention, Rost & Sander 1994).
- backbone_sasa : ASA over backbone atoms (N, CA, C, O) only. Exposed backbone means an
                  inserted domain projects into solvent rather than the core.

Computed on the full complex (protein + RNA + DNA) so burial by the bound nucleic acids
counts: a residue facing the RNA/DNA channel is not exposed for accommodating a domain.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
import freesasa

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.common import (PDB_PATH, PROTEIN_CHAIN, MAXASA_TIEN2013,
                             protein_residues, load_structure, PROCESSED)

BACKBONE = {"N", "CA", "C", "O"}


def compute() -> pd.DataFrame:
    structure = freesasa.Structure(PDB_PATH)
    result = freesasa.calc(structure)

    # per-residue total and backbone ASA over the protein chain
    n = structure.nAtoms()
    total = {}      # resnum -> total ASA
    backbone = {}   # resnum -> backbone ASA
    for i in range(n):
        if structure.chainLabel(i) != PROTEIN_CHAIN:
            continue
        try:
            resnum = int(structure.residueNumber(i))
        except ValueError:
            continue  # insertion codes etc.
        a = result.atomArea(i)
        total[resnum] = total.get(resnum, 0.0) + a
        if structure.atomName(i).strip() in BACKBONE:
            backbone[resnum] = backbone.get(resnum, 0.0) + a

    model = load_structure()
    res = protein_residues(model)
    rows = []
    for resnum, r in res.items():
        maxasa = MAXASA_TIEN2013.get(r.resname)
        asa = total.get(resnum, np.nan)
        rel = asa / maxasa if (maxasa and not np.isnan(asa)) else np.nan
        rows.append({
            "site": resnum,
            "rel_sasa": rel,
            "backbone_sasa": backbone.get(resnum, np.nan),
        })
    df = pd.DataFrame(rows).sort_values("site").reset_index(drop=True)
    print(f"{len(df)} residues; rel_sasa median={df['rel_sasa'].median():.3f}")
    return df


if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)
    compute().to_csv(os.path.join(PROCESSED, "feat_sasa.csv"), index=False)
    print("wrote feat_sasa.csv")
