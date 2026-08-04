"""
Build the MIOFlow latent as gene-PHATE concatenated with the lambda-weighted HEIST
spatial half, then reduce to 2D via the GAGA 2-stage AE.

  base = PHATE(X_pca_harmony, knn=7, t=15)   (N,2)  clean gene lineage (cached)
  aug  = X_heist[:, :128]                     (N,128) HEIST high-level (spatial) half
  combined = [ blocknorm(base) , lambda * blocknorm(aug) ]   # unit total-var per block
  latent   = GAGA(combined -> 2D)             -> obsm['X_gaga_2stage_heist']

Writes a per-lambda copy axolotl_heist{SUFFIX}.h5ad. No Harmony (X_pca_harmony already
batch-corrected; HEIST spatial half is batch-neutral).

Env: HEIST_LAMBDA (float), HEIST_SUFFIX (str), HEIST_BASE_H5AD (path).
Run: HEIST_LAMBDA=0.5 HEIST_SUFFIX=_l0p5 .venv/bin/python mioflow_heist/08_phate_heist_latent.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import scanpy as sc
import phate
from scipy.spatial.distance import squareform, pdist
from scipy.stats import spearmanr

from src.gaga import Autoencoder, train_gaga_two_phase, dataloader_from_pc

LAM = float(os.environ.get("HEIST_LAMBDA", "0.5"))
SUFFIX = os.environ.get("HEIST_SUFFIX", "_l0p5")
BASE_H5AD = os.environ.get("HEIST_BASE_H5AD", paths.h5ad("axolotl_heist.h5ad"))
OUT_H5AD = paths.h5ad(f"axolotl_heist{SUFFIX}.h5ad")
WEIGHTS = paths.ckpt(f"gaga_heist{SUFFIX}.pth")
PHATE_CACHE = paths.traj("phate_gene.npy")
SEED = 0
# combined-PHATE hyperparameters (tunable for visualization). The base gene-PHATE stays
# knn=7,t=15 (cached). USE_GAGA=0 uses PHATE(combined) directly as the 2D latent (no GAGA).
PKNN = int(os.environ.get("PHATE_KNN", "7"))
_pt = os.environ.get("PHATE_T", "15"); PT = 'auto' if _pt == 'auto' else int(_pt)
PDECAY = int(os.environ.get("PHATE_DECAY", "40"))
USE_GAGA = os.environ.get("USE_GAGA", "1") == "1"

torch.manual_seed(SEED); np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[step8] lambda={LAM} suffix='{SUFFIX}' device={device}")

adata = sc.read_h5ad(BASE_H5AD)
Xg = np.asarray(adata.obsm['X_pca_harmony'], dtype=np.float32)   # gene PCA (batch-corrected)
Xh = np.asarray(adata.obsm['X_heist'], dtype=np.float32)
half = Xh.shape[1] // 2
aug = Xh[:, :half].astype(np.float32)                          # HEIST spatial half (128)

# The HEIST spatial half carries a sequencing-chip batch effect (unlike the gene base,
# which is already Harmony-corrected). HARMONIZE_AUG=1 removes it before concatenation.
if os.environ.get("HARMONIZE_AUG", "0") == "1":
    import scanpy.external as sce
    adata.obsm['_aug_raw'] = aug
    print("[step8] Harmony batch-correcting the HEIST spatial half on 'Batch'...")
    sce.pp.harmony_integrate(adata, 'Batch', basis='_aug_raw',
                             adjusted_basis='_aug_harmony', max_iter_harmony=20)
    aug = np.asarray(adata.obsm['_aug_harmony'], dtype=np.float32)

# --- base = PHATE(gene PCA), cached once across lambdas ---
if os.path.exists(PHATE_CACHE):
    base = np.load(PHATE_CACHE).astype(np.float32)
    print(f"[step8] loaded cached gene-PHATE {base.shape}")
else:
    print("[step8] computing PHATE(X_pca_harmony)...")
    base = phate.PHATE(n_components=2, knn=7, t=15, n_jobs=-1,
                       random_state=SEED, verbose=False).fit_transform(Xg).astype(np.float32)
    np.save(PHATE_CACHE, base)
    print(f"[step8] cached gene-PHATE -> {PHATE_CACHE}")
adata.obsm['X_phate_gene_base'] = base

def block_norm(B):
    B = B - B.mean(0)
    tv = float((B.var(axis=0)).sum())          # total variance across dims
    return (B / np.sqrt(tv + 1e-12)).astype(np.float32)

# HEIST_ONLY=1 uses only the HEIST embedding (no gene-PHATE base).
# HEIST_ONLY_PART: full (256, default) | spatial (128) | gene (128).
if os.environ.get("HEIST_ONLY", "0") == "1":
    part = os.environ.get("HEIST_ONLY_PART", "full")
    feat = {"full": Xh, "spatial": Xh[:, :half], "gene": Xh[:, half:]}[part]
    combined = block_norm(feat)
    print(f"[step8] HEIST-only ({part}) combined {combined.shape} — no gene-PHATE base")
else:
    combined = np.concatenate([block_norm(base), LAM * block_norm(aug)], axis=1).astype(np.float32)
    print(f"[step8] combined {combined.shape} (base 2 + spatial {aug.shape[1]}), "
          f"total-var ratio base:aug = 1 : {LAM**2:.3g}")

# --- PHATE(combined): the 2D visualization manifold (also GAGA's distance target) ---
print(f"[step8] PHATE(combined) knn={PKNN} t={PT} decay={PDECAY} | use_gaga={USE_GAGA}")
pe = phate.PHATE(n_components=2, knn=PKNN, t=PT, decay=PDECAY, n_jobs=-1,
                 random_state=SEED, verbose=False).fit_transform(combined).astype(np.float32)

if USE_GAGA:
    pe_n = (pe / pe.std()).astype(np.float32)
    D = squareform(pdist(pe_n)).astype(np.float32)
    loader = dataloader_from_pc(combined, D, batch_size=1024, shuffle=True)
    model = Autoencoder(input_dim=combined.shape[1], latent_dim=2, hidden_dims=[128, 64])
    train_gaga_two_phase(model=model, train_loader=loader, encoder_epochs=300, decoder_epochs=300,
                         learning_rate=1e-3, device=device, dist_weight_phase1=1.0, recon_weight_phase2=1.0)
    model.eval()
    with torch.no_grad():
        Z = model.encode(torch.tensor(combined, device=device)).cpu().numpy().astype(np.float32)
    torch.save(model.state_dict(), WEIGHTS)
else:
    Z = pe.copy()   # PHATE(combined) is the latent directly (no GAGA re-embedding)

adata.obsm['X_gaga_2stage_heist'] = Z
tb = adata.obs['time_bin'].astype(float).to_numpy()
sp = max(abs(spearmanr(Z[:, i], tb)[0]) for i in range(2))
print(f"[step8] latent {Z.shape} | max|spearman(latent, time_bin)| = {sp:.3f}")
adata.write_h5ad(OUT_H5AD)
print(f"[step8] wrote {OUT_H5AD}")
