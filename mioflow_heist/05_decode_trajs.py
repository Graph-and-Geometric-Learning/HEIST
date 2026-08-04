"""
Step 5: Decode the latent trajectories (ptrajs) into gene and L-R feature space.

- gtrajs: dec_gene(ptrajs) -> gene-PCA (200) -> genes (16556) via varm['pcs'] + mean(X)
          (mirrors bp542 analyze-trajs back-projection).
- strajs: dec_lr(ptrajs)   -> LR_feats (640).  strajs[...,72] = NCAN:SDC3 signaling.

Run: .venv/bin/python mioflow_heist/05_decode_trajs.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import paths

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import scipy.sparse as sp
import torch
import scanpy as sc

from aux_model import Decoder

RAN_PATH = paths.AXOLOTL_RAN
SUFFIX = os.environ.get("HEIST_SUFFIX", "")
PTRAJS = paths.traj(f"ptrajs{SUFFIX}.npy")
DEC = paths.ckpt(f"aux_decoders{SUFFIX}.pth")
GTRAJS = paths.traj(f"gtrajs{SUFFIX}.npy")
STRAJS = paths.traj(f"strajs{SUFFIX}.npy")

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[step5] device={device}")

ptrajs = np.load(PTRAJS).astype(np.float32)          # (500,100,2)
Ntraj, Nbins, Dlat = ptrajs.shape
print(f"[step5] ptrajs {ptrajs.shape}")

ck = torch.load(DEC, map_location=device)
dec_gene = Decoder(ck['latent_dim'], ck['gene_out']).to(device); dec_gene.load_state_dict(ck['dec_gene']); dec_gene.eval()
dec_lr = Decoder(ck['latent_dim'], ck['lr_out']).to(device); dec_lr.load_state_dict(ck['dec_lr']); dec_lr.eval()

# pristine adata for the gene back-projection (original log-norm X + varm['pcs'])
adata = sc.read_h5ad(RAN_PATH)
pcs = np.asarray(adata.varm['pcs'], dtype=np.float32)          # (16556, 200)
Xmean = np.asarray(adata.X.mean(axis=0)).ravel().astype(np.float32)  # (16556,)
print(f"[step5] pcs {pcs.shape} | Xmean {Xmean.shape}")

flat = torch.tensor(ptrajs.reshape(-1, Dlat), device=device)
with torch.no_grad():
    gene_pca = dec_gene(flat).cpu().numpy()          # (500*100, 200)  ~ X_pca_harmony scale
    lr = dec_lr(flat).cpu().numpy()                  # (500*100, 640)

# gene-PCA -> full gene space (bp542 math): pca @ pcs.T + mean(X)
genes = gene_pca @ pcs.T + Xmean[None, :]            # (500*100, 16556)
gtrajs = genes.reshape(Ntraj, Nbins, -1).astype(np.float32)
strajs = lr.reshape(Ntraj, Nbins, -1).astype(np.float32)
print(f"[step5] gtrajs {gtrajs.shape} | strajs {strajs.shape}")
assert gtrajs.shape[:2] == (Ntraj, Nbins) and strajs.shape == (Ntraj, Nbins, ck['lr_out'])

np.save(GTRAJS, gtrajs)
np.save(STRAJS, strajs)
print(f"[step5] saved {GTRAJS} and {STRAJS}")
