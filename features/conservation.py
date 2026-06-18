"""Axis E (importance) + Axis D (architecture) conservation features.

Two conservation routes, compared in the writeup:

- esm2_entropy   : ESM-2 (150M, esm2_t30_150M_UR50D) per-position Shannon entropy of the
                   predicted AA distribution. Windowed (1000-residue windows, 200 overlap)
                   since Cas9 (1368) > ESM-2's 1024 limit; overlaps averaged. Low entropy =
                   conserved = load-bearing = insertion-intolerant. No MSA gaps.
- msa_conservation: 1 - normalized Shannon entropy of the aligned ortholog column. From a
                   MAFFT MSA of SpCas9 + Cas9 orthologs.
- indel_frequency : gap/insertion frequency of the aligned column among orthologs. Where
                   nature inserts/deletes, engineered insertions tend to be tolerated.

Each stage caches its output. ESM-2 runs on MPS if available.
"""
from __future__ import annotations
import os
import sys
import math
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.common import (FASTA_PATH, RAW, PROCESSED, SEQ_LEN, load_sequence, find_exe)

ORTHOLOG_FASTA = os.path.join(RAW, "cas9_orthologs.fasta")
MSA_FASTA = os.path.join(PROCESSED, "cas9_msa.fasta")
ESM_CSV = os.path.join(PROCESSED, "feat_esm.csv")
AA20 = "ACDEFGHIKLMNPQRSTVWY"

# UniProt query for type II Cas9 orthologs (cached to ORTHOLOG_FASTA).
UNIPROT_QUERY = (
    "https://rest.uniprot.org/uniprotkb/stream?format=fasta&query="
    "(protein_name:%22CRISPR-associated%20endonuclease%20Cas9%22)%20AND%20(length:%5B900%20TO%201700%5D)"
)


def esm2_entropy(window=1000, overlap=200) -> pd.DataFrame:
    import torch
    import esm
    seq = load_sequence()
    # CPU by default: ESM-2 150M is only ~2 windows for Cas9 (seconds on CPU), and the
    # long-sequence MPS path segfaults intermittently on Apple Silicon. Set ESM_DEVICE=mps
    # to opt back in.
    want = os.environ.get("ESM_DEVICE", "cpu").lower()
    device = "mps" if (want == "mps" and torch.backends.mps.is_available()) else "cpu"
    torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))
    model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
    model = model.to(device).eval()
    bc = alphabet.get_batch_converter()
    aa_ids = [alphabet.tok_to_idx[a] for a in AA20]

    ent_sum = np.zeros(SEQ_LEN)
    ent_cnt = np.zeros(SEQ_LEN)
    step = window - overlap
    starts = list(range(0, SEQ_LEN, step))
    with torch.no_grad():
        for st in starts:
            en = min(st + window, SEQ_LEN)
            sub = seq[st:en]
            _, _, toks = bc([("w", sub)])
            toks = toks.to(device)
            logits = model(toks)["logits"][0]            # [L+2, vocab]
            logits = logits[1:1 + len(sub), aa_ids]       # strip BOS/EOS, keep 20 AA
            p = torch.softmax(logits, dim=-1)
            ent = -(p * torch.log(p + 1e-12)).sum(-1).cpu().numpy()  # nat
            for k, e in enumerate(ent):
                pos = st + k
                ent_sum[pos] += e
                ent_cnt[pos] += 1
            if en == SEQ_LEN:
                break
    ent = ent_sum / np.maximum(ent_cnt, 1)
    df = pd.DataFrame({"site": np.arange(1, SEQ_LEN + 1), "esm2_entropy": ent})
    print(f"entropy median={np.median(ent):.3f} nat over {SEQ_LEN} residues ({device})")
    return df


def fetch_orthologs():
    if os.path.exists(ORTHOLOG_FASTA) and os.path.getsize(ORTHOLOG_FASTA) > 0:
        return
    print("fetching Cas9 orthologs from UniProt ...")
    r = requests.get(UNIPROT_QUERY, timeout=120)
    r.raise_for_status()
    with open(ORTHOLOG_FASTA, "w") as f:
        f.write(r.text)
    n = sum(1 for l in open(ORTHOLOG_FASTA) if l.startswith(">"))
    print(f"fetched {n} ortholog sequences -> {ORTHOLOG_FASTA}")


def run_mafft(max_seqs=400):
    import subprocess, random
    if os.path.exists(MSA_FASTA) and os.path.getsize(MSA_FASTA) > 0:
        return
    fetch_orthologs()
    # SpCas9 canonical first (the mapping reference), then orthologs
    sp = load_sequence()
    seqs = [(">SpCas9_ref", sp)]
    name, buf = None, []
    for line in open(ORTHOLOG_FASTA):
        if line.startswith(">"):
            if name and buf:
                seqs.append((name, "".join(buf)))
            name, buf = line.strip(), []
        else:
            buf.append(line.strip())
    if name and buf:
        seqs.append((name, "".join(buf)))
    # subsample orthologs for a tractable MSA (keep SpCas9_ref at index 0)
    rest = seqs[1:]
    if len(rest) > max_seqs:
        random.Random(0).shuffle(rest)
        rest = rest[:max_seqs]
    seqs = [seqs[0]] + rest

    tmp_in = os.path.join(PROCESSED, "_mafft_in.fasta")
    with open(tmp_in, "w") as f:
        for nm, sq in seqs:
            f.write(f"{nm if nm.startswith('>') else '>'+nm}\n{sq}\n")
    mafft = find_exe("mafft") or "mafft"
    print(f"aligning {len(seqs)} sequences with MAFFT --auto ...")
    with open(MSA_FASTA, "w") as out:
        subprocess.run([mafft, "--auto", "--anysymbol", tmp_in], stdout=out, check=True)
    print(f"wrote {MSA_FASTA}")


def msa_features() -> pd.DataFrame:
    run_mafft()
    # read alignment
    names, seqs, name, buf = [], [], None, []
    for line in open(MSA_FASTA):
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

    rows = []
    resnum = 0
    n_seqs = aln.shape[0]
    for col in range(aln.shape[1]):
        if ref[col] == "-":
            continue  # column not in SpCas9 -> no residue to map
        resnum += 1
        column = aln[:, col]
        gap = np.mean(column == "-")
        # Shannon entropy over the 20 AA among non-gap rows
        aas = [c for c in column if c in AA20]
        if aas:
            counts = np.array([aas.count(a) for a in AA20], float)
            p = counts / counts.sum()
            p = p[p > 0]
            ent = -(p * np.log(p)).sum() / math.log(20)  # normalized to 0..1
        else:
            ent = np.nan
        rows.append({"site": resnum,
                     "msa_conservation": 1.0 - ent if not np.isnan(ent) else np.nan,
                     "indel_frequency": gap})
    df = pd.DataFrame(rows)
    assert resnum == SEQ_LEN, f"SpCas9 ref mapped {resnum} residues, expected {SEQ_LEN}"
    print(f"{n_seqs} seqs; conservation median={df['msa_conservation'].median():.3f}; "
          f"indel_freq median={df['indel_frequency'].median():.3f}")
    return df


def esm2_entropy_isolated() -> pd.DataFrame:
    """Compute ESM-2 entropy in a separate process and read the result.

    torch's pip wheel ships its own OpenMP runtime; loading it alongside conda's
    numpy/scipy/sklearn BLAS causes intermittent segfaults on macOS (two OpenMP runtimes).
    A child process keeps torch isolated. Cached to ESM_CSV.
    """
    import subprocess
    if not (os.path.exists(ESM_CSV) and os.path.getsize(ESM_CSV) > 0):
        env = dict(os.environ)
        env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
        env.setdefault("OMP_NUM_THREADS", "1")
        print("running ESM-2 in isolated subprocess ...")
        subprocess.run([sys.executable, os.path.abspath(__file__), "--esm-only"],
                       check=True, env=env)
    return pd.read_csv(ESM_CSV)


def compute() -> pd.DataFrame:
    esm = esm2_entropy_isolated()
    msa = msa_features()
    return esm.merge(msa, on="site", how="outer").sort_values("site").reset_index(drop=True)


if __name__ == "__main__":
    import argparse
    os.makedirs(PROCESSED, exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--esm-only", action="store_true",
                    help="(internal) compute ESM-2 entropy and write feat_esm.csv, then exit")
    a = ap.parse_args()
    if a.esm_only:
        esm2_entropy().to_csv(ESM_CSV, index=False)
        print(f"wrote {ESM_CSV}")
    else:
        compute().to_csv(os.path.join(PROCESSED, "feat_conservation.csv"), index=False)
        print("wrote feat_conservation.csv")
