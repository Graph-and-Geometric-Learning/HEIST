"""
Step 2: Reduce the HEIST embedding (256D) to a 2D latent via a GAGA 2-stage AE.

- PHATE(X_heist) -> pairwise-distance target for phase-1 (distance preservation).
- train_gaga_two_phase: phase1 encoder (distance), phase2 decoder (reconstruction).
- Store adata.obsm['X_gaga_2stage_heist'] (2D); save encoder/decoder weights.

Run: .venv/bin/python mioflow_heist/02_gaga_latent.py
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
import scanpy.external as sce
import phate
from scipy.spatial.distance import squareform, pdist

from src.gaga import Autoencoder, train_gaga_two_phase, dataloader_from_pc

SUFFIX = os.environ.get("HEIST_SUFFIX", "")
H5AD = os.environ.get("HEIST_H5AD", paths.h5ad("axolotl_heist.h5ad"))
OUT_H5AD = H5AD  # add obsm in place
WEIGHTS = paths.ckpt(f"gaga_heist{SUFFIX}.pth")
LATENT_DIM = 2
BATCH_KEY = 'Batch'
SEED = 0

torch.manual_seed(SEED); np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[step2] device={device}")

adata = sc.read_h5ad(H5AD)

# --- Harmony batch-correct the HEIST embedding (HEIST encodes spatial/section context,
#     so the raw embedding separates tissue sections; Harmony on Batch reveals the shared
#     lineage axis, mirroring bp542's use of harmony-corrected PCA) ---
print(f"[step2] Harmony batch-correcting X_heist on '{BATCH_KEY}' ...")
sce.pp.harmony_integrate(adata, BATCH_KEY, basis='X_heist',
                         adjusted_basis='X_heist_harmony', max_iter_harmony=20)
X = np.asarray(adata.obsm['X_heist_harmony'], dtype=np.float32)
print(f"[step2] X_heist_harmony {X.shape}")

# --- PHATE embedding of the corrected HEIST -> distance target for phase 1
#     (knn=7, t=15 as in bp542's plot.ipynb, for a smoother continuum) ---
print("[step2] computing PHATE for the distance target...")
phate_op = phate.PHATE(n_components=2, knn=7, t=15, n_jobs=-1, random_state=SEED, verbose=False)
phate_emb = phate_op.fit_transform(X).astype(np.float32)
adata.obsm['X_phate_heist'] = phate_emb
# Normalize PHATE target by its std (as bp542 does) so the latent lands at scale ~1,
# matching the regime MIOFlow's hyperparameters (lambda_density/energy, lr) were tuned for.
phate_norm = (phate_emb / phate_emb.std()).astype(np.float32)
D = squareform(pdist(phate_norm)).astype(np.float32)   # NxN pairwise distances (normalized)
print(f"[step2] distance matrix {D.shape}, mean={D.mean():.4f}  (phate std={phate_emb.std():.4g})")

# --- GAGA 2-stage training ---
loader = dataloader_from_pc(X, D, batch_size=1024, shuffle=True)
model = Autoencoder(input_dim=X.shape[1], latent_dim=LATENT_DIM, hidden_dims=[128, 64])
train_gaga_two_phase(
    model=model,
    train_loader=loader,
    encoder_epochs=300,
    decoder_epochs=300,
    learning_rate=1e-3,
    device=device,
    dist_weight_phase1=1.0,
    recon_weight_phase2=1.0,
)

# --- Encode all cells -> 2D latent ---
model.eval()
with torch.no_grad():
    Z = model.encode(torch.tensor(X, device=device)).cpu().numpy()
adata.obsm['X_gaga_2stage_heist'] = Z.astype(np.float32)
print(f"[step2] latent Z {Z.shape}  range x[{Z[:,0].min():.2f},{Z[:,0].max():.2f}] "
      f"y[{Z[:,1].min():.2f},{Z[:,1].max():.2f}]")

torch.save(model.state_dict(), WEIGHTS)
adata.write_h5ad(OUT_H5AD)
print(f"[step2] saved weights -> {WEIGHTS} ; wrote obsm['X_gaga_2stage_heist'] -> {OUT_H5AD}")
