"""Elastic-network normal-mode dynamics feature (ProDy ANM) for SpCas9.

An Anisotropic Network Model is built on the holo Cas9 Ca trace (4UN3 chain B): each Ca is
a node, Ca pairs within a cutoff are connected by springs, and diagonalizing the Hessian
gives the normal modes. The slowest nonzero modes are the large collective motions (the
REC-lobe activation swing and similar). A residue with high fluctuation in those slow modes
sits on a moving part, which an inserted domain tends to jam -> high slow-mode fluctuation =
intolerant.

This is the single-structure analogue of apo_holo_disp: it needs only one structure, so it
covers every resolved residue rather than only the apo/holo overlap.

    pip install prody
    python features/dynamics.py    # writes data/processed/feat_dynamics.csv
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.common import PDB_PATH, PROTEIN_CHAIN, PROCESSED, SEQ_LEN

CUTOFF = 15.0     # ANM Ca contact cutoff (A); standard for coarse-grained ENM
N_MODES = 10      # number of slowest nonzero (collective) modes to keep


def compute(n_modes: int = N_MODES, cutoff: float = CUTOFF) -> pd.DataFrame:
    try:
        import prody
    except ImportError as e:
        raise SystemExit("ProDy not installed. Run: pip install prody") from e
    prody.confProDy(verbosity="none")

    ag = prody.parsePDB(PDB_PATH)
    ca = ag.select(f"protein and chain {PROTEIN_CHAIN} and name CA")
    if ca is None:
        raise SystemExit(f"no Ca atoms selected for chain {PROTEIN_CHAIN}")

    anm = prody.ANM("cas9 holo")
    anm.buildHessian(ca, cutoff=cutoff)
    anm.calcModes(n_modes=n_modes, zeros=False)   # slowest n nonzero modes

    msf = prody.calcSqFlucts(anm)                 # per-Ca fluctuation summed over the modes
    resnums = ca.getResnums()

    # collapse any duplicate resnums (altlocs) by mean, then index over 1..SEQ_LEN
    by_res: dict[int, list[float]] = {}
    for rn, v in zip(resnums, msf):
        by_res.setdefault(int(rn), []).append(float(v))
    rows = [{"site": i,
             "anm_msf": float(np.mean(by_res[i])) if i in by_res else np.nan}
            for i in range(1, SEQ_LEN + 1)]
    df = pd.DataFrame(rows)

    cov = int(df["anm_msf"].notna().sum())
    print(f"ANM: {ca.numAtoms()} Ca nodes, {n_modes} slow modes, cutoff {cutoff} A; "
          f"covered {cov}/{SEQ_LEN} residues; anm_msf median={df['anm_msf'].median():.3e}")
    return df


if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)
    compute().to_csv(os.path.join(PROCESSED, "feat_dynamics.csv"), index=False)
    print("wrote feat_dynamics.csv")
