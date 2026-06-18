"""Axis B (proximity to function) distance features, computed on 4UN3.

- min_dist_rna : min heavy-atom distance from each Cas9 residue to any sgRNA atom
                 (chain A). Insertions near the sgRNA channel are selected against
                 (Oakes Fig 1c).
- min_dist_dna : min heavy-atom distance to any DNA atom (chains C + D).

These operationalize the Oakes finding that binding regions are insertion-intolerant.
min_dist_dna is non-monotonic (some tolerated hotspots sit ~10 A from the DNA termini);
plain distance only captures the dominant trend.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.common import (RNA_CHAINS, DNA_CHAINS, protein_residues,
                             load_structure, PROCESSED)


def _atoms_of_chains(model, chain_ids):
    atoms = []
    for cid in chain_ids:
        if cid in model:
            for res in model[cid]:
                for atom in res:
                    if not atom.element == "H":
                        atoms.append(atom)
    return atoms


def _min_dist(res, ns, target_coords):
    # min heavy-atom distance from this residue's atoms to the target atom set
    best = np.inf
    for atom in res:
        if atom.element == "H":
            continue
        d = np.sqrt(((target_coords - atom.coord) ** 2).sum(axis=1)).min()
        if d < best:
            best = d
    return best


def compute() -> pd.DataFrame:
    model = load_structure()
    rna_atoms = _atoms_of_chains(model, RNA_CHAINS)
    dna_atoms = _atoms_of_chains(model, DNA_CHAINS)
    rna_coords = np.array([a.coord for a in rna_atoms]) if rna_atoms else np.empty((0, 3))
    dna_coords = np.array([a.coord for a in dna_atoms]) if dna_atoms else np.empty((0, 3))

    res = protein_residues(model)
    rows = []
    for resnum, r in res.items():
        rows.append({
            "site": resnum,
            "min_dist_rna": _min_dist(r, None, rna_coords) if len(rna_coords) else np.nan,
            "min_dist_dna": _min_dist(r, None, dna_coords) if len(dna_coords) else np.nan,
        })
    df = pd.DataFrame(rows).sort_values("site").reset_index(drop=True)
    print(f"{len(df)} residues; "
          f"min_dist_rna median={df['min_dist_rna'].median():.1f} A, "
          f"min_dist_dna median={df['min_dist_dna'].median():.1f} A")
    return df


if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)
    compute().to_csv(os.path.join(PROCESSED, "feat_distances.csv"), index=False)
    print("wrote feat_distances.csv")
