"""Axis F (direct simulation): chimera_pLDDT. Stretch feature, not run in the time box.

The only feature that directly simulates the insertion rather than proxying for it.
Described here and in writeup.md; it's the "with more time" item.

Plan:
  1. For insertion site i, build the chimera sequence:
         Cas9[1..i] + linker + PDZ(86 aa, alpha1-syntrophin) + linker + Cas9[i+1..end]
     using the same PDZ domain as Oakes/Mathony.
  2. Fold only a local window (PDZ + ~25 flanking Cas9 residues each side), not the whole
     1368-aa chimera.
  3. Use the original ESMFold (ESMFold2's Triton kernels don't run on Apple Silicon).
     recycles = 0-1.
  4. Read mean pLDDT over the Cas9 flank: high pLDDT => local fold preserved => tolerant.

Sketch (not executed):

    import torch, esm
    model = esm.pretrained.esmfold_v1().eval().cuda()   # or .to('mps')
    def chimera_plddt(i, cas9_seq, pdz_seq, linker="GSGSGS", flank=25):
        left  = cas9_seq[max(0, i-flank):i]
        right = cas9_seq[i:i+flank]
        chunk = left + linker + pdz_seq + linker + right
        with torch.no_grad():
            out = model.infer(chunk, num_recycles=1)
        plddt = out["plddt"][0, :, 1]           # per-residue pLDDT
        # average over the Cas9 flank positions only (exclude PDZ + linkers)
        ...
        return float(flank_plddt.mean())

Cost: ~overnight on an M-series Mac for ~700 prediction-set sites (local windows are short),
or minutes on a cloud GPU. Left as future work.
"""

ENABLED = False
