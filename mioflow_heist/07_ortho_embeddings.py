"""
Task A: Compute cell embeddings with the `final_model_with_custom_layer_full_ortho`
checkpoint (first-commit architecture) on the SAME cached graphs used by the pretrained
HEIST run, so the two are directly comparable.

- Loads the first-commit GraphEncoder (local copy in ortho_model/) + the ortho checkpoint (strict).
- Reuses cached graphs mioflow_heist/data/axolotl_heist.pt (no re-preprocessing).
- Stores the 256D embedding under obsm['X_heist'] of a fresh axolotl_heist_ortho.h5ad
  (same key name as the pretrained run, so steps 2-6 work unchanged with HEIST_H5AD/HEIST_SUFFIX).

Run: .venv/bin/python mioflow_heist/07_ortho_embeddings.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
ORTHO = os.path.join(HERE, "ortho_model")
sys.path.insert(0, ORTHO)   # old `model` package (must precede repo root)
sys.path.insert(1, REPO)    # utils.*
sys.path.insert(2, HERE)    # paths
import paths

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import scanpy as sc
from torch_geometric.nn.pool import global_mean_pool

from model.model import GraphEncoder            # -> ortho_model/model/model.py
from utils.dataloader import create_dataloader

RAN_PATH = paths.AXOLOTL_RAN
CKPT = os.path.join(REPO, "saved_models", "final_model_with_custom_layer_full_ortho.pth")
GRAPHS_PT = paths.data("axolotl_heist.pt")
OUT_H5AD = paths.h5ad("axolotl_heist_ortho.h5ad")
BATCH_SIZE = 5000

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"[ortho] device={device}")
print(f"[ortho] model package: {sys.modules['model'].__file__}")

# --- Load ortho checkpoint + build first-commit GraphEncoder ---
ck = torch.load(CKPT, map_location='cpu', weights_only=False)
a = ck['args']
print(f"[ortho] args: pe_dim={a.pe_dim} init={a.init_dim} hidden={a.hidden_dim} "
      f"output={a.output_dim} layers={a.num_layers} heads={a.num_heads} "
      f"cross={a.cross_message_passing} pe={a.pe} blending={a.blending}")
model = GraphEncoder(a.pe_dim, a.init_dim, a.hidden_dim, a.output_dim, a.num_layers,
                     a.num_heads, a.cross_message_passing, a.pe, a.blending).to(device)
model.load_state_dict(ck['model_state_dict'], strict=True)
model.eval()
output_dim = model.norm.normalized_shape[0]
print(f"[ortho] loaded (strict) | output_dim={output_dim}")

# --- Reuse cached graphs ---
graphs = torch.load(GRAPHS_PT, weights_only=False)
print(f"[ortho] cached graphs: {len(graphs)} (1 high + {len(graphs)-1} low)")

adata = sc.read_h5ad(RAN_PATH)
adata.obs_names_make_unique()
assert graphs[0].num_nodes == adata.n_obs, (graphs[0].num_nodes, adata.n_obs)

# --- Extract embeddings (iterate underlying partition list; NO gene_mask) ---
dataloader = create_dataloader(graphs, BATCH_SIZE, permute=False)
partitions = list(dataloader.dataset)
print(f"[ortho] {len(partitions)} METIS partition(s)")
emb = torch.zeros((graphs[0].num_nodes, 2 * output_dim), device=device)
with torch.no_grad():
    for high_sub, low_batch, batch_idx in partitions:
        high_sub = high_sub.to(device)
        low_batch = low_batch.to(device)
        batch_idx = batch_idx.to(device).reshape(-1)
        high_emb, low_emb = model.encode(high_sub, low_batch)   # blending off, pe on
        emb[batch_idx] = torch.cat([high_emb, global_mean_pool(low_emb, low_batch.batch)], dim=1)

X = emb.cpu().numpy()
print(f"[ortho] X_heist_ortho shape={X.shape}")
assert X.shape == (adata.n_obs, 2 * output_dim) and np.isfinite(X).all()

# aliasing sanity (gene half varies per cell within a type)
gh = X[:, output_dim:]
ct = adata.obs['Annotation'].astype(str).to_numpy()
sub = gh[ct == ct[0]]
print(f"[ortho] aliasing check: type '{ct[0]}' {sub.shape[0]} cells, "
      f"{np.unique(np.round(sub,5),axis=0).shape[0]} distinct gene-half rows")

adata.obsm['X_heist'] = X   # same key as pretrained run
adata.write_h5ad(OUT_H5AD)
print(f"[ortho] wrote {OUT_H5AD}")
