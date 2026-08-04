"""
Preprocessing-level batch correction: compute HEIST embeddings PER SECTION (per Batch)
instead of one graph over all sections.

Why: the 18 axolotl sections are tiled into a shared coordinate frame (many overlap in
(x,y)), so a single Voronoi graph wires spurious edges between different sections and the
raw-coordinate PE differs by section -> a sequencing/section batch effect in the embedding.
Building one spatial graph per section (with per-section coordinate frames) removes it at the
root, and matches how HEIST is meant to be used (one graph per tissue sample).

Uses a GLOBAL top-200 HVG set (so gene-graph nodes are comparable across sections), but
per-section spatial graphs + per-section GRNs.

Output: obsm['X_heist'] on a fresh axolotl_heist_persec.h5ad (same key as the global run,
so step 08 works with HEIST_BASE_H5AD=.../axolotl_heist_persec.h5ad).

Run: .venv/bin/python mioflow_heist/01b_heist_embeddings_persection.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
import paths

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import torch
import scanpy as sc
from torch_geometric.nn.pool import global_mean_pool

from heist_preprocess import preprocess
from utils.dataloader import create_dataloader
from model.model import GraphEncoder

SECTION_KEY = "Batch"
OUT_H5AD = paths.h5ad("axolotl_heist_persec.h5ad")
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"[persec] device={device}")

adata = sc.read_h5ad(paths.AXOLOTL_RAN)
adata.obs_names_make_unique()

# HEIST-input copy (raw counts) + real cell types
ah = adata.copy()
ah.X = ah.layers['counts'].copy()
ah.obs['cell_type'] = ah.obs['Annotation'].astype(str)

# --- GLOBAL top-200 HVG (so gene nodes are comparable across sections) ---
tmp = ah.copy()
sc.pp.normalize_total(tmp, target_sum=1e4); sc.pp.log1p(tmp)
sc.pp.highly_variable_genes(tmp, n_top_genes=200)
hvg = tmp.var_names[tmp.var['highly_variable']].tolist()
ah = ah[:, hvg].copy()
print(f"[persec] global HVG genes: {ah.n_vars}")

model = GraphEncoder.from_pretrained("HirenMadhu/HEIST").to(device).eval()
output_dim = model.final_norm.normalized_shape[0]
print(f"[persec] model loaded | output_dim={output_dim}")

sections = list(map(str, adata.obs[SECTION_KEY].astype(str).unique()))
X_heist = np.zeros((adata.n_obs, 2 * output_dim), dtype=np.float32)
sec_arr = adata.obs[SECTION_KEY].astype(str).to_numpy()

for i, sec in enumerate(sections):
    idx = np.where(sec_arr == sec)[0]
    sub = ah[idx].copy()
    graphs = preprocess(sub, save_root=paths.DATA_DIR, save_file_name=f"axolotl_sec{i}",
                        max_genes=200, spatial="spatial", cell_type="cell_type")
    dl = create_dataloader(graphs, batch_size=len(idx), permute=False)
    partitions = list(dl.dataset)
    emb = torch.zeros((graphs[0].num_nodes, 2 * output_dim), device=device)
    with torch.no_grad():
        for high_sub, low_batch, bidx in partitions:
            high_sub = high_sub.to(device); low_batch = low_batch.to(device)
            bidx = bidx.to(device).reshape(-1)
            he, le = model.encode(high_sub, low_batch)
            emb[bidx] = torch.cat([he, global_mean_pool(le, low_batch.batch)], dim=1)
    X_heist[idx] = emb.cpu().numpy()
    print(f"[persec] {i+1}/{len(sections)} {sec}: {len(idx)} cells | parts={len(partitions)}")

assert np.isfinite(X_heist).all(), "non-finite embeddings"
adata.obsm['X_heist'] = X_heist
adata.write_h5ad(OUT_H5AD)
print(f"[persec] X_heist {X_heist.shape} -> wrote {OUT_H5AD}")
