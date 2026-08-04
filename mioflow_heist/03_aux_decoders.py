"""
Step 3: Train auxiliary decoders from the 2D HEIST latent to feature spaces, so
MIOFlow trajectories can be decoded for the Fig-5 gene-trend and L-R panels.

- dec_gene: 2D latent -> X_pca_harmony (200D)   [-> genes via varm['pcs'] in step 5]
- dec_lr:   2D latent -> LR_feats (640D)         [strajs[...,72] = NCAN:SDC3]

Reports held-out R^2 (is gene/LR info recoverable from a 2D latent?).
Run: .venv/bin/python mioflow_heist/03_aux_decoders.py
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
import torch.nn as nn
import scanpy as sc
from sklearn.metrics import r2_score

from aux_model import Decoder

SUFFIX = os.environ.get("HEIST_SUFFIX", "")
H5AD = os.environ.get("HEIST_H5AD", paths.h5ad("axolotl_heist.h5ad"))
OUT = paths.ckpt(f"aux_decoders{SUFFIX}.pth")
SEED = 0
EPOCHS = 1500
LR = 1e-3

torch.manual_seed(SEED); np.random.seed(SEED)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"[step3] device={device}")

adata = sc.read_h5ad(H5AD)
Z = np.asarray(adata.obsm['X_gaga_2stage_heist'], dtype=np.float32)   # (N,2)
gene = np.asarray(adata.obsm['X_pca_harmony'], dtype=np.float32)      # (N,200)
lr = np.asarray(adata.obsm['LR_feats'], dtype=np.float32)             # (N,640)
print(f"[step3] Z{Z.shape} gene{gene.shape} lr{lr.shape}")


def train_decoder(Z, Y, name):
    n = Z.shape[0]
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(n)
    n_tr = int(0.9 * n)
    tr, te = perm[:n_tr], perm[n_tr:]
    Zt = torch.tensor(Z, device=device)
    Yt = torch.tensor(Y, device=device)
    dec = Decoder(Z.shape[1], Y.shape[1]).to(device)
    opt = torch.optim.Adam(dec.parameters(), lr=LR)
    lossf = nn.MSELoss()
    tr_idx = torch.tensor(tr, device=device)
    for ep in range(EPOCHS):
        dec.train(); opt.zero_grad()
        loss = lossf(dec(Zt[tr_idx]), Yt[tr_idx])
        loss.backward(); opt.step()
    dec.eval()
    with torch.no_grad():
        pred_te = dec(Zt[torch.tensor(te, device=device)]).cpu().numpy()
    r2 = r2_score(Y[te], pred_te, multioutput='variance_weighted')
    print(f"[step3] {name}: held-out R^2 = {r2:.4f}  (train MSE {loss.item():.4f})")
    return dec


dec_gene = train_decoder(Z, gene, "dec_gene (2D->200 gene-PCA)")
dec_lr = train_decoder(Z, lr, "dec_lr   (2D->640 LR-feats)")

torch.save({'dec_gene': dec_gene.state_dict(),
            'dec_lr': dec_lr.state_dict(),
            'gene_out': gene.shape[1], 'lr_out': lr.shape[1], 'latent_dim': Z.shape[1]}, OUT)
print(f"[step3] saved decoders -> {OUT}")
