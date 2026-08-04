"""
Compute a HEIST-native pseudotime (DPT on the HEIST embedding, root in reaEGC) to
REPLACE the collaborator's gene-PCA DPT for the HEIST figure, then re-bin for MIOFlow.

Mirrors bp542's method (multi-root DPT) but on the HEIST embedding instead of gene PCA.
Writes obs['time_bin'] (new, HEIST) + obs['dpt_pseudotime'] into a copy of the run h5ad;
keeps the old bp542 values as obs['time_bin_bp542'] / obs['dpt_pseudotime_bp542'].

Env: HEIST_SUFFIX (run to re-time, e.g. _l0p5_ps), HEIST_USE_REP (default X_heist),
     HEIST_NBINS (default 4), N_ROOTS (default 15).
Run: HEIST_SUFFIX=_l0p5_ps .venv/bin/python mioflow_heist/09_heist_pseudotime.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.metrics import silhouette_score

SUFFIX = os.environ.get("HEIST_SUFFIX", "_l0p5_ps")
USE_REP = os.environ.get("HEIST_USE_REP", "X_heist")          # DPT on the HEIST embedding
NBINS = int(os.environ.get("HEIST_NBINS", "4"))
N_ROOTS = int(os.environ.get("N_ROOTS", "15"))
ROOT_TYPE = os.environ.get("ROOT_TYPE", "reaEGC")            # progenitor / radial glia
OUT_SUFFIX = SUFFIX + "_htime"
SEED = 0
rng = np.random.default_rng(SEED)

adata = sc.read_h5ad(paths.h5ad(f"axolotl_heist{SUFFIX}.h5ad"))
print(f"[step9] {adata.n_obs} cells | DPT on '{USE_REP}' {adata.obsm[USE_REP].shape} | root={ROOT_TYPE}")

# --- diffusion map on the HEIST embedding ---
sc.pp.neighbors(adata, use_rep=USE_REP, n_neighbors=15, random_state=SEED)
sc.tl.diffmap(adata)

# --- multi-root DPT from reaEGC cells (as bp542), averaged for robustness ---
root_pool = np.where(adata.obs['Annotation'].astype(str).to_numpy() == ROOT_TYPE)[0]
roots = rng.choice(root_pool, size=min(N_ROOTS, len(root_pool)), replace=False)
acc = np.zeros(adata.n_obs)
for r in roots:
    adata.uns['iroot'] = int(r)
    sc.tl.dpt(adata)
    acc += np.asarray(adata.obs['dpt_pseudotime'])
dpt = acc / len(roots)
dpt = (dpt - dpt.min()) / (dpt.max() - dpt.min() + 1e-12)     # normalize 0..1

# orientation sanity: reaEGC (progenitor) should be early, neurons late
ct = adata.obs['Annotation_grouped'].astype(str).to_numpy()
if dpt[ct == 'reaEGC'].mean() > dpt[np.isin(ct, ['nptxEX'])].mean():
    dpt = 1.0 - dpt
    print("[step9] flipped orientation (reaEGC set to early)")

# --- bin into NBINS well-separated groups (equal-count quantiles) ---
codes = pd.qcut(dpt, NBINS, labels=False, duplicates='drop')
time_bin = (codes / (codes.max())).astype(np.float32)          # -> {0, .., 1}

# --- report: cell-type ordering + bin separation ---
print("[step9] mean HEIST pseudotime by cell type (expect reaEGC<rIPCs<IMN/nptxEX):")
for c in ['reaEGC', 'rIPCs', 'IMN', 'nptxEX']:
    m = ct == c
    if m.any():
        print(f"    {c:8} {dpt[m].mean():.3f}")
emb = np.asarray(adata.obsm['X_gaga_2stage_heist'])
he = np.asarray(adata.obsm[USE_REP])
print(f"[step9] bin counts: {np.bincount(codes)}")
print(f"[step9] bin silhouette in {USE_REP}: {silhouette_score(he, codes, sample_size=3000, random_state=0):+.3f} | "
      f"in MIOFlow latent: {silhouette_score(emb, codes, sample_size=3000, random_state=0):+.3f}")
from scipy.stats import spearmanr
if 'dpt_pseudotime' in adata.obs and 'time_bin_bp542' not in adata.obs:
    print(f"[step9] spearman(HEIST dpt, bp542 dpt): {spearmanr(dpt, adata.obs['dpt_pseudotime'])[0]:.3f}")

# --- store: keep bp542 values, set new HEIST time as the active pseudotime ---
adata.obs['time_bin_bp542'] = adata.obs['time_bin'].values
adata.obs['dpt_pseudotime_bp542'] = adata.obs['dpt_pseudotime'].values
adata.obs['time_bin'] = time_bin
adata.obs['dpt_pseudotime'] = dpt.astype(np.float32)

out = paths.h5ad(f"axolotl_heist{OUT_SUFFIX}.h5ad")
adata.write_h5ad(out)
print(f"[step9] wrote {out}  (suffix for downstream: {OUT_SUFFIX})")
