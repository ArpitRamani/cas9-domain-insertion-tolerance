# Cas9 Domain-Insertion Tolerance — Prediction Pipeline

Predicts, per residue of SpCas9 (1,368 aa), how tolerant that position is to insertion of a
folded domain, and outputs a **calibrated probability with an uncertainty estimate** so
promising insertion sites can be prioritized for wet-lab testing.

Data: Oakes et al. 2016 (*Nat Biotechnol*) PDZ-insertion screen. Structure: PDB **4UN3**.

## Setup

```bash
conda env create -f environment.yml
conda activate cas9
pip install torch fair-esm                                   # ESM-2 conservation feature
Rscript -e 'install.packages("dbarts", repos="https://cloud.r-project.org")'  # BART
```

Required raw inputs (already in `data/raw/`, or auto-downloaded):
- `41587_2016_BFnbt3528_MOESM3_ESM.xlsx` — Oakes Supplementary Table 1 (labels). **Provided.**
- `4UN3.pdb` — `curl -o data/raw/4UN3.pdb https://files.rcsb.org/download/4UN3.pdb`
- `4UN3.cif` — `curl -o data/raw/4UN3.cif https://files.rcsb.org/download/4UN3.cif`
  (mmCIF; mkdssp 4.x is mmCIF-native. Needs the libcifpp CCD `components.cif`, shipped with
  the conda `dssp` package; the pipeline sets `$LIBCIFPP_DATA_DIR` automatically.)
- `SpCas9_Q99ZW2.fasta` — `curl -o data/raw/SpCas9_Q99ZW2.fasta https://rest.uniprot.org/uniprotkb/Q99ZW2.fasta`
- Cas9 orthologs for the MSA are auto-fetched from UniProt on first run (cached).

## Run

```bash
conda activate cas9
python data/parse_labels.py          # -> data/processed/labels.csv (635 measured, 176 tolerant)
python pipeline.py --build-features   # compute all features (slow: ESM-2 on MPS + MAFFT MSA)
python pipeline.py                    # reuse cached features: train + nested-CV eval + predict
```

Individual feature modules are runnable standalone for debugging, e.g.
`python features/sasa.py`, `python features/structure.py` (DSSP), `python features/conservation.py`.

## Outputs (`outputs/`)

- **`predictions.csv`** — the deliverable. One row per SpCas9 residue (1–1368): BART
  probability + 95% credible interval (`bart_prob/lo/hi/sd`), honest out-of-fold prediction
  for measured sites, `in_prediction_set` flag, an `in_support` / `n_features_out` abstention
  flag (0 = the residue is outside the measured feature envelope, so its score is
  extrapolation), and all features. The unmeasured residues (`in_prediction_set == 1`) are the
  product output — the sites the model scores that the screen never tested.
- **`metrics.json`** — honest out-of-fold AUPRC, precision@{20,50}, AUROC, Brier, and
  by-axis importance.
- **`reliability.png`** — reliability diagram (held-out grouped folds).
- **`axis_importance.csv`** — BART variable-inclusion, aggregated **by biological axis**
  (not raw column).

## Layout

```
data/         raw supplement + parsed labels + assembled features
  parse_labels.py
features/     sasa, distances, structure (DSSP), geometry, conservation (ESM-2 + MSA), chimera (stretch)
  feature_config.yaml   on/off + params + axis per feature
models/       bart.R + bart.py (dbarts)
eval/         split.py (grouped/blocked CV), metrics.py (AUPRC, p@k), calibration.py
pipeline.py   assemble -> nested-CV eval -> calibrate -> final-fit -> predict
writeup.md    target choice, features by axis, evaluation, limitations, next steps
```

## Key design choices (see writeup.md / spec rigor rules)

1. **Unmeasured ≠ intolerant.** We train/evaluate ONLY on the 635 significant measured sites;
   the ~733 other residues are the *prediction set*, never imputed as negative.
2. **Grouped/blocked CV only** (leave-a-domain-out or contiguous blocks) — residues are
   spatially autocorrelated; random k-fold leaks.
3. **RSA kept continuous**, normalized with the Tien 2013 max-ASA scale.
4. **Calibration reported**, not just discrimination (reliability + Brier).
5. **Importance interpreted by axis**, since within-axis features are correlated by design.
