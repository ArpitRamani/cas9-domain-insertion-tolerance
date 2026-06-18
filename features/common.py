"""Shared constants and structure/sequence loaders.

Structural features use PDB 4UN3 (holo SpCas9 + sgRNA + target/non-target DNA).
Sequence features use the canonical SpCas9 sequence (UniProt Q99ZW2, 1368 aa).
"""
from __future__ import annotations
import os
import sys
import warnings
import numpy as np
from Bio.PDB import PDBParser

warnings.simplefilter("ignore")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROCESSED = os.path.join(ROOT, "data", "processed")
OUTPUTS = os.path.join(ROOT, "outputs")

PDB_PATH = os.path.join(RAW, "4UN3.pdb")
CIF_PATH = os.path.join(RAW, "4UN3.cif")  # mmCIF for mkdssp 4.x (PDB input is unreliable)
FASTA_PATH = os.path.join(RAW, "SpCas9_Q99ZW2.fasta")
SUPPLEMENT_XLSX = os.path.join(RAW, "41587_2016_BFnbt3528_MOESM3_ESM.xlsx")


def find_exe(name: str) -> str | None:
    """Find an executable on PATH or in the active interpreter's bin/ (so it works even
    when the conda env isn't activated, just invoked by full python path)."""
    import shutil
    exe = shutil.which(name)
    if exe:
        return exe
    cand = os.path.join(os.path.dirname(sys.executable), name)
    return cand if os.path.exists(cand) else None


def libcifpp_data_dir() -> str | None:
    """mkdssp 4.x needs the libcifpp CCD (components.cif). Find it under the active
    conda prefix and return the dir (caller sets $LIBCIFPP_DATA_DIR)."""
    cand = os.path.join(sys.prefix, "share", "libcifpp")
    return cand if os.path.exists(os.path.join(cand, "components.cif")) else None

# 4UN3 chain layout
PROTEIN_CHAIN = "B"      # SpCas9
RNA_CHAINS = ["A"]       # sgRNA
DNA_CHAINS = ["C", "D"]  # target (C) + non-target (D) DNA

SEQ_LEN = 1368  # SpCas9 residues 1..1368

# Domain boundaries from UniProt Q99ZW2, used for dist_to_domain_boundary and for
# leave-a-domain-out CV grouping. Each tuple is (name, start, end) inclusive.
DOMAINS = [
    ("RuvC-I", 1, 62),
    ("BridgeHelix", 63, 93),   # ARM/bridge helix region (UniProt ARM 56-73 + classic BH 60-93)
    ("REC", 94, 717),          # Recognition lobe (UniProt REC 56-718, minus RuvC-I/BH overlap)
    ("RuvC-II", 718, 765),
    ("HNH", 766, 924),         # HNH Cas9-type domain (UniProt 770-921, padded to linkers)
    ("RuvC-III", 925, 1098),
    ("PI", 1099, 1368),        # PAM-interacting domain
]

# Unique domain start/end positions; dist_to_domain_boundary is the sequence distance
# to the nearest of these.
DOMAIN_BOUNDARIES = sorted({1, 62, 63, 93, 94, 717, 718, 765, 766, 924, 925, 1098, 1099, 1368})

# Tien et al. 2013 (PLoS ONE) "theoretical" max ASA (A^2) per residue, used to normalize
# observed ASA into relative SASA.
MAXASA_TIEN2013 = {
    "ALA": 129.0, "ARG": 274.0, "ASN": 195.0, "ASP": 193.0, "CYS": 167.0,
    "GLN": 225.0, "GLU": 223.0, "GLY": 104.0, "HIS": 224.0, "ILE": 197.0,
    "LEU": 201.0, "LYS": 236.0, "MET": 224.0, "PHE": 240.0, "PRO": 159.0,
    "SER": 155.0, "THR": 172.0, "TRP": 285.0, "TYR": 263.0, "VAL": 174.0,
}

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
    "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V",
}


def load_structure(path: str = PDB_PATH):
    parser = PDBParser(QUIET=True)
    return parser.get_structure("4un3", path)[0]  # first model


def load_sequence(path: str = FASTA_PATH) -> str:
    seq = "".join(l.strip() for l in open(path) if not l.startswith(">"))
    assert len(seq) == SEQ_LEN, f"expected {SEQ_LEN} aa, got {len(seq)}"
    return seq


def protein_residues(model):
    """Standard amino-acid residues of the Cas9 chain, keyed by residue number."""
    chain = model[PROTEIN_CHAIN]
    out = {}
    for res in chain:
        if res.id[0] == " " and res.resname in THREE_TO_ONE:
            out[res.id[1]] = res
    return out


def domain_of(resnum: int) -> str:
    for name, s, e in DOMAINS:
        if s <= resnum <= e:
            return name
    return "NA"
