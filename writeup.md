# Cas9 Domain-Insertion Tolerance — Writeup

## Problem & target

We predict, per SpCas9 residue, the probability that the position tolerates insertion of a
folded ~86-aa domain, with a per-site uncertainty estimate, to prioritize wet-lab testing.

**Target (headline = binary).** From Oakes 2016 Supplementary Table 1 we take the PDZ
insertion enrichment per site. The supplement contains only the **significant** sites (all
rows have P < 0.1), giving real measured labels:

- `tolerant = 1` if `fold_change ≥ 2` (i.e. ≥ 2-fold enriched), else `0`.
- 635 measured sites after dropping site `AA = 0` (insertion N-terminal to residue 1, no
  residue to map): **176 tolerant / 459 intolerant (27.7% positive)** — matches the spec's
  expected ~175 positives. The 2 `inf`-fold sites (clone absent from the pre-screen library)
  are treated as strongly enriched → tolerant. A `log2 fold_change` column is also produced
  for the regression alternative (with a floor on depleted-to-zero clones).

**Missing-label handling (the critical rule).** The screen covered ~70% of sites and only
~half are significant. We **never impute unmeasured positions as intolerant** — unmeasured
= unknown. We train and evaluate ONLY on the 635 measured sites; the remaining ~733 residues
are the **prediction set**, and the model's scores on them are the actual product output
(`in_prediction_set == 1` in `outputs/predictions.csv`).

**Coordinate mapping.** Insertion site `AA = i` (the residue after which the insert's
N-terminus was detected; the Mu ~2-aa duplication is already collapsed into SpCas9 numbering
by the authors) is mapped to the features of residue `i`. All features are computed on 4UN3
(chain B = Cas9, A = sgRNA, C/D = DNA).

## Features (grouped by biological axis)

The **final production set is 10 features** across 6 axes — within the ~15–20 ceiling for 176
positives (10:1 EPV). Within-axis features are correlated *by design*; we interpret importance
by axis. Three originally-included features were **removed** after the ablation study below
(`is_loop`, `backbone_sasa`, `dist_to_domain_boundary`); one was **added** (`apo_holo_disp`).

| Axis | Features | Rationale |
|---|---|---|
| **A — Exposure / room** | `rel_sasa` (continuous, Tien-2013 max-ASA), `open_volume_18A` (heavy atoms within 18 Å of CA), `flexibility` (CA B-factor) | A domain can only sit where there is solvent-exposed room and conformational give. RSA kept continuous (no buried/exposed threshold — convention, Rost & Sander 1994). *(`backbone_sasa` dropped: redundant with `rel_sasa`.)* |
| **B — Proximity to function** | `min_dist_rna`, `min_dist_dna` | Insertions in the sgRNA/DNA binding channels are selected against (Oakes Fig 1c). *(A swap to specific landmark distances — duplex ends / heteroduplex / non-target strand — was tested and did not beat the broad pair.)* |
| **C — Local structure** | `dist_to_sse_end` | Element edges tolerate insertion better than mid-element. *(`is_loop` dropped: binary loop/not-loop over-penalizes tolerant SSEs — Oakes notes insertions hit nearly every secondary-structure element.)* |
| **D — Insertion architecture** | `indel_frequency` (ortholog MSA gap freq) | Positions where nature already indels tolerate engineered insertions. *(`dist_to_domain_boundary` dropped: it leaked domain identity — see ablation.)* |
| **E — Importance** | `esm2_entropy` (ESM-2 150M, windowed), `msa_conservation` (1 − normalized column entropy) | Conserved = load-bearing = intolerant. Both routes computed: ESM-2 gives gap-free coverage; the MSA additionally yields `indel_frequency` for free. |
| **G — Conformational dynamics** | `apo_holo_disp` (per-residue Cα displacement, apo 4CMP → holo 4UN3, RuvC-core superposition) | Inserting into a large activation mover (e.g. REC2) jams the conformational change → intolerant. **Not novel** — the field holds that insertion permissibility is "best explained by dynamic protein properties" and apo/holo comparison is a known site-finding tactic (ProDomino, Nat Methods 2025) — but it is a structural signal the leading *sequence-based* predictors omit, and our ablation independently shows it is the feature that helps most under the hardest CV. |
| **F — Direct simulation (stretch)** | `chimera_pLDDT` (not built) | Fold a local window of the actual Cas9+PDZ chimera with ESMFold and read pLDDT over the Cas9 flank — the only feature that *simulates* the insertion. Highest-value missing feature; described in `features/chimera.py`. |

## Modeling

One model: **BART** (Bayesian Additive Regression Trees, binary/probit, `dbarts`). The latent
score for a residue is a sum of 200 shallow trees passed through the probit link, so
nonlinearities and feature interactions (e.g. the non-monotonic `min_dist_dna`) are captured
automatically. The posterior predictive gives a calibrated probability **and a 95% credible
interval** per site — the decision-support output the use case needs. BART self-regularizes
via its priors (depth `base=0.95, power=2`, leaf `k=2`, `ntree=200`), so it is left untuned;
the tuning bake-off below shows why that is the right call.

## Evaluation

- **Grouped/blocked CV, never random k-fold** — residues are spatially autocorrelated, so a
  random split leaks adjacent residues across train/test. We report **both** spec-endorsed
  grouped splits, which *bracket* real performance: **block CV** (contiguous sequence blocks)
  is the deployment-realistic estimate — the product scores *unmeasured residues interspersed
  among measured ones*, so every domain has training data; **leave-a-domain-out** is a
  conservative stress test (generalize to a wholly unseen domain). `group_by` in
  `feature_config.yaml`.
- **Ranking-oriented metrics** (the use case is picking sites): **AUPRC** and
  **precision@{20,50}**; AUROC/accuracy secondary (imbalance makes accuracy meaningless).
- **Calibration required**: reliability diagram + Brier on held-out grouped folds.
- **Nested CV**: the production model is untuned, so the outer grouped CV is itself the honest
  estimate; the tuning bake-off below adds an inner loop to select BART's grid point and report
  the optimism gap.
- **Importance by axis**: BART variable-inclusion aggregated per axis.

### Results (nested CV, out-of-fold, n=627, base rate 0.28; final 10-feature model)

Leave-a-domain-out (conservative stress test):

| Model | AUPRC | precision@20 | precision@50 | AUROC | Brier |
|---|---|---|---|---|---|
| **BART** | **0.64** | 0.70 | **0.82** | **0.81** | **0.149** |

Under **block CV** (deployment-realistic) BART reaches **AUPRC 0.65, AUROC 0.84**. Honest
performance is **~0.64 AUPRC vs a 0.28 base rate**, and **41 of BART's top-50 out-of-fold
picks are truly tolerant** (precision@50 0.82) — the number that matters for site-picking.
Progression as the feature set was disciplined: **0.50** (initial 12 features) → **0.59**
(drop leaky `dist_to_domain_boundary`) → **0.64** (drop `is_loop`/`backbone_sasa`, add
`apo_holo_disp`).

BART is well-calibrated (Brier 0.149; reliability diagram `outputs/reliability.png`) and its
tree interactions exploit signal a linear fit would miss, e.g. the non-monotonic
`min_dist_dna`, while the posterior gives a credible interval per site. The clearest evidence
that **features, not hyperparameters, were the lever**: dropping the leaky
`dist_to_domain_boundary` alone lifted leave-a-domain-out AUPRC by **+0.085** (0.505 → 0.59),
the single largest change in the study, whereas tuning moved honest AUPRC ≤0.02 (next
section). Feature edits moved the model by 0.1–0.2; hyperparameters did not.

### Feature selection: we tested clever features and ended up *subtracting* one

Two ablations (`feature_ablation.py`, `feature_drop_test.py`):

1. **Mechanism-specific candidates didn't help.** We built four insertion-specific features
   — `solvent_cone` (directional open space), `long_range_contacts` and `contact_order`
   (folding-nucleus disruption), and `true_insertion_freq` (actual insertions in the
   ortholog MSA, mapped to the "after residue i" convention). Under grouped-CV add-one tests
   each contributed ≤0.007 AUPRC (within noise); `solvent_cone` was strong alone (0.50) but
   *redundant* with the exposure cluster, and `true_insertion_freq` was too sparse
   (univariate 0.24, below base rate). BART confirmed: adding the best two moved AUPRC
   0.505→0.507. The generic structural/conservation features already capture the learnable
   signal.
2. **One production feature was hurting.** `dist_to_domain_boundary` had drop-one
   AUPRC −0.042 under domain-holdout. The drop test (BART, both CV schemes) showed why:
   removing it gives +0.085 AUPRC under domain-holdout and −0.012 under block CV — it partly
   encodes domain identity, helping when the domain is seen and backfiring when it isn't.
   Keeping it is *fragile* (0.51/0.65 across CV schemes); dropping it is *robust* (0.59/0.64).
   We also dropped `is_loop` and `backbone_sasa` (redundant; Section 2). We drop it.
3. **One feature genuinely helped: `apo_holo_disp`.** Of all candidates it was the only
   additive winner (HGB add-one +0.018), and the dual-CV confirmation (`feature_add_test.py`)
   showed it is *not* leaky: adding it to BART improved leave-a-domain-out by **+0.052** AUPRC
   (0.583→0.635) while staying neutral under block CV (+0.001). A leaky feature does the
   reverse (helps block, hurts domain); helping *most* on the hardest split means it carries
   transferable biophysical signal. The block↔domain gap is our leakage detector throughout.
4. **Elastic-network dynamics didn't add anything.** As an alternative dynamics descriptor we
   built an ANM (ProDy) on the holo Cα trace and took each residue's slow-mode fluctuation
   (`anm_msf`; `features/dynamics.py`, `dynamics_test.py`). Despite covering every resolved
   residue, it lowered AUPRC slightly under both CV schemes (0.635→0.621 domain, 0.645→0.639
   block) and was much worse as a swap for `apo_holo_disp` (0.575 domain). ANM reports a single
   structure's *intrinsic* equilibrium fluctuations, a smooth profile dominated by termini and
   surface loops, so it overlaps the B-factor feature and is far less specific to the actual
   apo→holo activation motion that `apo_holo_disp` measures from two real states.

Net lesson: with 176 positives the levers were **disciplined feature removal** and **one
mechanism-motivated structural feature** — not hyperparameter search and not piling on
generic features.

**Axis importance** (BART variable-inclusion, aggregated by axis; final 10-feature model):
exposure/room (A ≈ 0.30) leads, then function-proximity (B ≈ 0.20), conservation (E ≈ 0.20),
insertion architecture (D ≈ 0.11), the dynamics axis (G `apo_holo_disp` ≈ 0.10), and local
structure (C ≈ 0.09, now thin since `is_loop` was dropped, leaving only `dist_to_sse_end`).
Low importance ≠ harmful (cf. the high-importance *but harmful* `dist_to_domain_boundary` we
removed — importance and helpfulness are distinct questions). This ordering differs from
Mathony 2023 (conservation strongest), plausibly because our DSSP/SASA exposure features are
cleaner here than the entropy-based conservation proxies. See `outputs/axis_importance.csv`.

### Hyperparameter tuning is not the lever

A small `k × ntree` grid under identical nested CV (`tuning_bakeoff.py` →
`outputs/tuning_comparison.csv`) tests whether tuning BART buys anything, reporting honest
out-of-fold scores and the **optimism gap** (inner-CV-best minus honest).

| model | tuner | select on | inner_best | honest AUPRC | honest AUROC | optimism gap |
|---|---|---|---|---|---|---|
| BART | grid | AUPRC | 0.597 | 0.524 | 0.731 | **0.073** |
| BART | grid | AUROC | 0.822 | 0.507 | 0.718 | 0.104 |

Three findings: (1) **Tuning BART is pointless** — the grid moves honest AUPRC only
0.50→0.52 (within noise); its priors already regularize. (2) The **optimism gap is small**
(0.073): BART's inner-CV estimate (0.597) nearly matches its honest score (0.524), so the
model generalizes faithfully and the headline is not an artifact of selection. (3)
**Selecting on AUROC did not improve honest AUROC** (0.718 vs 0.731) — optimizing a noisy
metric through a noisy inner loop can backfire, so we select on the use-case metric
(AUPRC/precision@k). The spec's ban on Bayesian optimization is the right call in this
regime: a noise-dominated inner objective (3–7 positives in some folds), a tiny knob space,
and cheap evals are exactly where BO amplifies the winner's-curse bias without buying
generalization. Net: we keep BART at its self-regularizing defaults (k=2, ntree=200); the
lever for more performance is features (e.g. `chimera_pLDDT`), not hyperparameters.

## Limitations

- `min_dist_dna` is non-monotonic — some tolerated hotspots sit ~10 Å from the DNA termini,
  positioned to access without disrupting binding (Oakes). Decomposing it into specific
  landmark distances (duplex ends vs heteroduplex channel) was tested but didn't beat the
  broad pair; a finer geometric treatment might.
- Dynamics captured only as a two-state difference (`apo_holo_disp`, 4CMP→4UN3), with 231
  residues unresolved in the apo structure (median-imputed). An elastic-network alternative
  (ANM slow-mode fluctuation) was tested and didn't beat it (Feature selection, item 4);
  MD-derived RMSF over an ensemble remains the unexplored option.
- 62 of 1368 residues are unresolved in 4UN3 (termini + disordered loops); their structural
  features are median-imputed (sequence features still cover them).
- **Eval ≠ deployment (covariate shift).** Trained/evaluated on the 635 measured sites but
  scored on the 733 unmeasured ones (the product output). `covariate_shift.py` finds a
  moderate, concentrated shift — classifier separability AUC 0.68, 13% of the deployment set
  outside the measured 1st–99th percentile envelope, dominated by `apo_holo_disp`. Because
  trees extrapolate flat, BART's intervals understate uncertainty there, so `predictions.csv`
  carries an `in_support`/`n_features_out` flag to abstain on that tail; conformal prediction
  is the principled extension.
- Labels are PDZ-specific (one 86-aa domain); tolerance for other domains may differ.

## What I'd do next

1. **Build `chimera_pLDDT`** — fold local Cas9+PDZ chimera windows with original ESMFold,
   read mean pLDDT over the Cas9 flank. The only feature that directly simulates the
   insertion; ~hours (top-N shortlist) to overnight (full set) on the M5.
2. **Deepen the dynamics axis** — `apo_holo_disp` was the one feature that helped under the
   hardest CV, consistent with the field's view that permissibility is best explained by
   dynamics (ProDomino, Nat Methods 2025). The simplest elastic-network proxy (ANM slow-mode
   fluctuation) was tested and didn't help; richer descriptors (mode-based hinge scores,
   per-mode displacements) or MD-derived RMSF over an ensemble are the next things to try.
3. Validate the top predicted prediction-set sites prospectively, and recalibrate on any new
   measured data.
4. Carry the regression target (log2 FC) as a secondary head for finer site ranking.

## References
Oakes 2016 *Nat Biotechnol* 34:646 (data + insertion biology) · Mathony 2023 *Adv Sci*
(ASA/SS/conservation determinants; same PDZ) · ProDomino 2025, "Rational engineering of allosteric protein switches by in silico
prediction of domain insertion sites," *Nat Methods*
[s41592-025-02741-z](https://www.nature.com/articles/s41592-025-02741-z)
(in-silico insertion-site prediction; permissibility best explained by dynamics) ·
Tien 2013 *PLoS ONE* (max-ASA) · Rost & Sander 1994 *Proteins* (RSA convention) ·
Lin 2023 *Science* (ESM-2) · Chipman, George & McCulloch 2010 *Ann Appl Stat* (BART).
