"""Mechanism-specific candidate features for domain-insertion tolerance, from data
already on disk (4UN3 + the cached ortholog MSA). Less generic than SASA/B-factor/
conservation; each encodes a specific requirement of grafting in a folded domain:

- solvent_cone        : directional open space. Of the protein heavy atoms within 15 A of
                        the CA, how many lie in the outward cone (toward solvent, away from
                        the centroid). Low = the insert can project outward. Sharper than
                        scalar SASA/open_volume, which can't tell an outward-facing loop
                        from one tucked into a groove.
- long_range_contacts : residues contacting this one (CB-CB < 8 A) that are far in
                        sequence (|i-j| > 12). Cutting here rips the folding nucleus; a
                        residue with only local contacts is benign.
- contact_order       : mean sequence separation of this residue's contacts. High = wired
                        into long-range structure.
- true_insertion_freq : fraction of orthologs with an actual insertion right after this
                        residue in the MSA (SpCas9 gap, homologs have residues). A strong
                        prior for engineered-insertion tolerance, mapped to the same "after
                        residue i" convention as the labels. Sharper than indel_frequency,
                        which lumps deletions in.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd
from Bio.PDB import NeighborSearch, PDBParser, Superimposer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.common import (PROTEIN_CHAIN, RNA_CHAINS, DNA_CHAINS, THREE_TO_ONE, RAW,
                             protein_residues, load_structure, PROCESSED, SEQ_LEN)
from features.conservation import MSA_FASTA

CONE_RADIUS = 15.0
CONE_COS = 0.5        # ~60 degree half-angle cone
CONTACT_DIST = 8.0    # CB-CB contact threshold
SEQ_SEP_LONG = 12     # |i-j| beyond this counts as long-range
APO_PDB = os.path.join(RAW, "4CMP.pdb")          # apo SpCas9 (no RNA/DNA)
RUVC_CORE = [(1, 59), (718, 775), (909, 1098)]   # rigid catalytic scaffold for superposition


def _cb_coord(res):
    if "CB" in res:
        return res["CB"].coord
    if "CA" in res:
        return res["CA"].coord
    return None


def true_insertion_freq(msa_path: str = MSA_FASTA) -> pd.DataFrame:
    names, seqs, name, buf = [], [], None, []
    for line in open(msa_path):
        if line.startswith(">"):
            if name is not None:
                names.append(name); seqs.append("".join(buf))
            name, buf = line[1:].strip(), []
        else:
            buf.append(line.strip())
    if name is not None:
        names.append(name); seqs.append("".join(buf))
    aln = np.array([list(s) for s in seqs])
    ref_idx = next(i for i, n in enumerate(names) if n.startswith("SpCas9_ref"))
    ref = aln[ref_idx]
    other = np.delete(aln, ref_idx, axis=0)

    ins_after = {i: 0.0 for i in range(0, SEQ_LEN + 1)}
    resnum = 0
    for col in range(aln.shape[1]):
        if ref[col] == "-":
            # insertion relative to SpCas9, sits after residue resnum
            frac = float(np.mean(other[:, col] != "-")) if other.size else 0.0
            ins_after[resnum] = max(ins_after[resnum], frac)
        else:
            resnum += 1
    rows = [{"site": i, "true_insertion_freq": ins_after.get(i, 0.0)}
            for i in range(1, SEQ_LEN + 1)]
    return pd.DataFrame(rows)


def _na_atoms(model, chain_ids, resnums=None):
    """Heavy atoms of the given nucleic-acid chains, optionally restricted to resnums."""
    out = []
    for cid in chain_ids:
        if cid not in model:
            continue
        for r in model[cid]:
            if resnums is not None and r.id[1] not in resnums:
                continue
            for a in r:
                if a.element != "H":
                    out.append(a.coord)
    return np.array(out) if out else np.empty((0, 3))


def channel_distances(model) -> pd.DataFrame:
    """Distances to specific nucleic-acid landmarks, splitting the non-monotonic broad
    min_dist_dna into mechanistically opposite signals (Oakes Fig 1c):

    - min_dist_na_ends     : to the 5'/3' termini of sgRNA + DNA. Access to the duplex ends
                             (~10 A) favors insertion -> closer = more tolerant.
    - min_dist_heteroduplex: to the interior of the paired sgRNA:target-DNA channel, which
                             suppresses insertion -> closer = less tolerant.
    - min_dist_nontarget   : to the non-target DNA strand (the groove Oakes highlights).
    """
    res = protein_residues(model)
    end_res, het_res = {}, {}
    for cid in RNA_CHAINS + DNA_CHAINS:
        if cid not in model:
            continue
        rn = sorted(r.id[1] for r in model[cid] if r.id[0] == " ")
        if not rn:
            continue
        end_res[cid] = {rn[0], rn[1] if len(rn) > 1 else rn[0], rn[-1], rn[-2] if len(rn) > 1 else rn[-1]}
        het_res[cid] = set(rn) - end_res[cid]

    ends = np.vstack([_na_atoms(model, [c], end_res[c]) for c in end_res]) if end_res else np.empty((0, 3))
    # heteroduplex = interior of sgRNA (chain A) + interior of target DNA (chain C)
    het_chains = [c for c in (RNA_CHAINS + DNA_CHAINS[:1]) if c in het_res]
    het = np.vstack([_na_atoms(model, [c], het_res[c]) for c in het_chains]) if het_chains else np.empty((0, 3))
    nontarget = _na_atoms(model, DNA_CHAINS[1:])  # chain D

    def mind(res_obj, target):
        if len(target) == 0:
            return np.nan
        best = np.inf
        for a in res_obj:
            if a.element == "H":
                continue
            d = np.sqrt(((target - a.coord) ** 2).sum(axis=1)).min()
            best = min(best, d)
        return best

    rows = []
    for n, r in res.items():
        rows.append({"site": n,
                     "min_dist_na_ends": mind(r, ends),
                     "min_dist_heteroduplex": mind(r, het),
                     "min_dist_nontarget": mind(r, nontarget)})
    return pd.DataFrame(rows)


def apo_holo_displacement() -> pd.DataFrame:
    """Per-residue Ca displacement between apo (4CMP) and holo (4UN3) Cas9, after
    superposing on the rigid RuvC core. Large rigid-body motion (e.g. the REC2 activation
    swing) is easily jammed by an insertion -> high displacement = intolerant.
    """
    holo = protein_residues(load_structure())
    holo_ca = {n: r["CA"] for n, r in holo.items() if "CA" in r}

    apo_model = PDBParser(QUIET=True).get_structure("apo", APO_PDB)[0]
    best = {}
    for ch in apo_model:                       # pick the apo chain with most resolved CAs
        ca = {r.id[1]: r["CA"] for r in ch
              if r.id[0] == " " and r.resname in THREE_TO_ONE and "CA" in r}
        if len(ca) > len(best):
            best = ca
    apo_ca = best

    common = sorted(set(holo_ca) & set(apo_ca))
    core = [n for n in common if any(a <= n <= b for a, b in RUVC_CORE)]
    fitset = core if len(core) >= 50 else common
    sup = Superimposer()
    sup.set_atoms([holo_ca[n] for n in fitset], [apo_ca[n] for n in fitset])
    rot, tran = sup.rotran

    rows = []
    for n in range(1, SEQ_LEN + 1):
        if n in holo_ca and n in apo_ca:
            ac = np.dot(apo_ca[n].coord, rot) + tran
            rows.append({"site": n, "apo_holo_disp": float(np.linalg.norm(ac - holo_ca[n].coord))})
        else:
            rows.append({"site": n, "apo_holo_disp": np.nan})
    print(f"apo-holo: {len(common)} common CA, fit on {len(fitset)} RuvC-core, "
          f"RMSD={sup.rms:.2f} A")
    return pd.DataFrame(rows)


def compute() -> pd.DataFrame:
    model = load_structure()
    res = protein_residues(model)
    chain = model[PROTEIN_CHAIN]
    heavy = [a for a in chain.get_atoms() if a.element != "H"]
    ns = NeighborSearch(heavy)

    ca = np.array([r["CA"].coord for r in res.values() if "CA" in r])
    centroid = ca.mean(axis=0)

    nums = [n for n in res if _cb_coord(res[n]) is not None]
    cb = np.array([_cb_coord(res[n]) for n in nums])
    num_index = {n: k for k, n in enumerate(nums)}

    rows = []
    for n, r in res.items():
        if "CA" not in r:
            rows.append({"site": n, "solvent_cone": np.nan,
                         "long_range_contacts": np.nan, "contact_order": np.nan})
            continue
        cap = r["CA"].coord
        outward = cap - centroid
        norm = np.linalg.norm(outward)
        outward = outward / norm if norm > 1e-6 else np.array([0.0, 0.0, 1.0])
        cone = 0
        for a in ns.search(cap, CONE_RADIUS, level="A"):
            v = a.coord - cap
            d = np.linalg.norm(v)
            if d < 1e-6:
                continue
            if np.dot(v / d, outward) > CONE_COS:
                cone += 1
        if n in num_index:
            d = np.sqrt(((cb - cb[num_index[n]]) ** 2).sum(axis=1))
            mask = (d < CONTACT_DIST) & (d > 0.1)
            partners = [nums[k] for k in np.where(mask)[0]]
            lrc = sum(1 for p in partners if abs(p - n) > SEQ_SEP_LONG)
            co = float(np.mean([abs(p - n) for p in partners])) if partners else 0.0
        else:
            lrc, co = np.nan, np.nan
        rows.append({"site": n, "solvent_cone": cone,
                     "long_range_contacts": lrc, "contact_order": co})

    df = (pd.DataFrame(rows)
          .merge(channel_distances(model), on="site", how="outer")
          .merge(apo_holo_displacement(), on="site", how="outer")
          .merge(true_insertion_freq(), on="site", how="right"))
    df = df.sort_values("site").reset_index(drop=True)
    print(f"{len(df)} residues; solvent_cone med={df['solvent_cone'].median():.0f}; "
          f"min_dist_na_ends med={df['min_dist_na_ends'].median():.1f}; "
          f"min_dist_heteroduplex med={df['min_dist_heteroduplex'].median():.1f}; "
          f"apo_holo_disp med={df['apo_holo_disp'].median():.2f}; "
          f"true_insertion_freq med={df['true_insertion_freq'].median():.3f}")
    return df


if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)
    compute().to_csv(os.path.join(PROCESSED, "feat_novel.csv"), index=False)
    print("wrote feat_novel.csv")
