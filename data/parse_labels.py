"""Parse Oakes 2016 Supplementary Table 1 into a per-site label table.

Source: 41587_2016_BFnbt3528_MOESM3_ESM.xlsx, sheet "Table 2".
Columns: AA (SpCas9 insertion-site residue), Fold Change (linear, not log2), P-value.

Binary label: tolerant = (fold change >= 2) AND (P < 0.1). The supplement holds only
significant sites, so every row is a measured label. The ~730 residues not in the table
are the prediction set (unmeasured, not negative) and are never imputed as intolerant.

Coordinate convention: AA = i is the residue after which the insert's N-terminus was
detected (Oakes methods; Mu duplication already collapsed into this numbering). We map
site i to residue i. AA = 0 has no residue and is dropped (one negative).
"""
from __future__ import annotations
import math
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from features.common import SUPPLEMENT_XLSX, PROCESSED, SEQ_LEN, domain_of

FC_FLOOR = 1e-3  # floor for log2 of depleted-to-zero clones (regression target only)


def parse() -> pd.DataFrame:
    raw = pd.read_excel(SUPPLEMENT_XLSX, sheet_name="Table 2", skiprows=1)
    raw = raw.rename(columns={"AA": "site", "Fold Change": "fold_change", "P-value": "pvalue"})
    raw = raw.dropna(subset=["site"]).copy()
    raw["site"] = raw["site"].astype(int)

    # 'inf' = clone not sequenced in pre-screened library => strongly enriched / undefined.
    # pandas may already coerce the literal "inf" to float inf, so catch both forms.
    def _is_inf(v):
        if isinstance(v, str):
            return v.strip().lower() == "inf"
        return isinstance(v, float) and math.isinf(v)
    is_inf = raw["fold_change"].apply(_is_inf)
    fc = pd.to_numeric(raw["fold_change"].where(~is_inf), errors="coerce")
    raw["is_inf"] = is_inf.values
    raw["fold_change"] = fc.where(~is_inf, np.inf).values
    raw["fc_zero"] = (raw["fold_change"] == 0)

    # Binary target: tolerant if FC >= 2 (inf counts as enriched) and P < 0.1.
    raw["label"] = (((raw["fold_change"] >= 2.0) | raw["is_inf"]) & (raw["pvalue"] < 0.1)).astype(int)

    # Regression target: log2 fold change with a floor on depleted-to-zero clones.
    fc_clip = raw["fold_change"].replace(np.inf, np.nan)
    fc_clip = fc_clip.clip(lower=FC_FLOOR)
    raw["log2fc"] = np.log2(fc_clip)
    # inf -> set to max observed finite log2fc (highly enriched) for the regression target.
    raw.loc[raw["is_inf"], "log2fc"] = raw["log2fc"].max()

    raw["measured"] = 1
    raw["domain"] = raw["site"].apply(domain_of)

    # Drop AA = 0 (no residue 0) and anything outside 1..1368.
    before = len(raw)
    raw = raw[(raw["site"] >= 1) & (raw["site"] <= SEQ_LEN)].copy()
    dropped = before - len(raw)

    raw = raw[["site", "domain", "fold_change", "is_inf", "fc_zero",
               "log2fc", "pvalue", "label", "measured"]].sort_values("site").reset_index(drop=True)

    print(f"{len(raw)} measured sites (dropped {dropped} out-of-range/N-term)")
    print(f"positives (tolerant): {raw['label'].sum()}  "
          f"negatives: {(raw['label'] == 0).sum()}  "
          f"({100*raw['label'].mean():.1f}% positive)")
    print(f"inf sites: {raw['is_inf'].sum()}  zero-FC (depleted) sites: {raw['fc_zero'].sum()}")
    return raw


if __name__ == "__main__":
    os.makedirs(PROCESSED, exist_ok=True)
    df = parse()
    out = os.path.join(PROCESSED, "labels.csv")
    df.to_csv(out, index=False)
    print(f"wrote {out}")
