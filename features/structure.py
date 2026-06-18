"""Axis C (local structure) + Axis A flexibility features, computed on 4UN3.

- sec_struct  : DSSP 8-state code (H,G,I,E,B,T,S,-) per residue.
- ss3         : collapsed 3-state (H = helix {H,G,I}, E = strand {E,B}, C = loop/coil).
- is_loop     : 1 if ss3 == 'C'. Inserting into a helix/sheet breaks its H-bond network;
                a loop absorbs it.
- flexibility : CA crystallographic B-factor (free, vs NMA-RMSF). Flexible loops absorb
                insertion strain.

DSSP is run via Biopython's wrapper around mkdssp (installed in the conda env).
"""
from __future__ import annotations
import os
import sys
import subprocess
import tempfile
import numpy as np
import pandas as pd
from Bio.PDB.DSSP import make_dssp_dict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.common import (CIF_PATH, PROTEIN_CHAIN, protein_residues,
                             load_structure, PROCESSED, libcifpp_data_dir, find_exe)

HELIX = set("HGI")
STRAND = set("EB")


def run_dssp(cif_path: str = CIF_PATH) -> dict:
    """Run mkdssp 4.x on the mmCIF, return {(chain, resid): (aa, ss, ...)}.

    mkdssp 4.x is mmCIF-native and needs the libcifpp CCD, so we set $LIBCIFPP_DATA_DIR
    and feed it the .cif (PDB input misdetects as mmCIF and fails).
    """
    mkdssp = find_exe("mkdssp")
    if mkdssp is None:
        raise FileNotFoundError("mkdssp not found (install via conda: dssp)")
    env = dict(os.environ)
    data_dir = libcifpp_data_dir()
    if data_dir and "LIBCIFPP_DATA_DIR" not in env:
        env["LIBCIFPP_DATA_DIR"] = data_dir
    with tempfile.TemporaryDirectory() as d:
        out = os.path.join(d, "out.dssp")
        subprocess.run([mkdssp, "--output-format", "dssp", cif_path, out],
                       check=True, env=env, capture_output=True, text=True)
        dssp_dict, _ = make_dssp_dict(out)
    return dssp_dict


def _ss3(code: str) -> str:
    if code in HELIX:
        return "H"
    if code in STRAND:
        return "E"
    return "C"


def compute() -> pd.DataFrame:
    model = load_structure()
    res = protein_residues(model)

    # DSSP
    dssp = run_dssp()
    ss_by_site = {}
    for (chain_id, resid), val in dssp.items():
        if chain_id != PROTEIN_CHAIN:
            continue
        resnum = resid[1]
        ss = val[1]  # value tuple is (aa, ss, acc, ...)
        if ss in ("-", "", " "):
            ss = "-"
        ss_by_site[resnum] = ss

    rows = []
    for resnum, r in res.items():
        ss8 = ss_by_site.get(resnum, "-")
        ss3 = _ss3(ss8)
        # CA B-factor; fall back to residue mean if no CA
        if "CA" in r:
            bfac = float(r["CA"].get_bfactor())
        else:
            bvals = [a.get_bfactor() for a in r]
            bfac = float(np.mean(bvals)) if bvals else np.nan
        rows.append({
            "site": resnum,
            "sec_struct": ss8,
            "ss3": ss3,
            "is_loop": int(ss3 == "C"),
            "flexibility": bfac,
        })
    df = pd.DataFrame(rows).sort_values("site").reset_index(drop=True)
    print(f"{len(df)} residues; "
          f"loop frac={df['is_loop'].mean():.2f}; "
          f"B-factor median={df['flexibility'].median():.1f}")
    return df


if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)
    compute().to_csv(os.path.join(PROCESSED, "feat_structure.csv"), index=False)
    print("wrote feat_structure.csv")
