"""
Step 1: Compute HEIST cell embeddings on the axolotl_ran subset.

- Loads the pristine axolotl_ran.h5ad (kept intact for downstream steps).
- Runs the CORRECTED heist_preprocess (per-cell gene graphs; no aliasing) on a COPY.
- Loads pretrained HEIST GraphEncoder, extracts [high(spatial) | pooled low(gene)] per cell.
- Stores adata.obsm['X_heist'] on the pristine adata and writes axolotl_heist.h5ad.

Run: .venv/bin/python mioflow_heist/01_heist_embeddings.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)   # for utils.*, model.*
sys.path.insert(0, HERE)   # for heist_preprocess, paths
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

RAN_PATH = paths.AXOLOTL_RAN
OUT_H5AD = paths.h5ad("axolotl_heist.h5ad")
PT_DIR = paths.DATA_DIR
BATCH_SIZE = 5000  # -> ceil(11716/5000)=3 METIS partitions (avoids nparts=1 edge case)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
print(f"[step1] device={device}")

# --- Load pristine adata (keep intact) and build a HEIST-only copy ---
adata = sc.read_h5ad(RAN_PATH)
adata.obs_names_make_unique()
print(f"[step1] loaded {adata.n_obs} cells x {adata.n_vars} genes")

ah = adata.copy()
ah.X = ah.layers['counts'].copy()            # HEIST expects raw counts (does its own norm/log/HVG)
ah.obs['cell_type'] = ah.obs['Annotation'].astype(str)
print("[step1] cell_type counts:\n", ah.obs['cell_type'].value_counts())

# --- Preprocess: build hierarchical graphs (heavy: Voronoi + per-type GRNs + MAGIC) ---
graphs = preprocess(
    adata=ah,
    save_root=PT_DIR,
    save_file_name="axolotl_heist",
    max_genes=200,
    spatial="spatial",
    cell_type="cell_type",
)
print(f"[step1] built {len(graphs)} graphs (1 high-level + {len(graphs)-1} low-level)")
assert graphs[0].num_nodes == adata.n_obs, (graphs[0].num_nodes, adata.n_obs)

# --- Load pretrained HEIST model ---
model = GraphEncoder.from_pretrained("HirenMadhu/HEIST").to(device)
model.eval()
output_dim = model.final_norm.normalized_shape[0]   # per-half embedding dim
print(f"[step1] model loaded | pe_dim={model.pe_dim} | output_dim={output_dim} | "
      f"positional_encoding={model.positional_encoding}")

# --- Extract embeddings: cat([high | mean-pooled low]) per cell ---
# Iterate the underlying partition list (NOT the collated DataLoader): PyG's
# batch_size=1 collater would re-batch low_level_batch and corrupt its .batch vector.
dataloader = create_dataloader(graphs, BATCH_SIZE, permute=False)
partitions = list(dataloader.dataset)
print(f"[step1] {len(partitions)} METIS partition(s)")
graph_embeddings = torch.zeros((graphs[0].num_nodes, 2 * output_dim), device=device)

with torch.no_grad():
    for high_level_subgraph, low_level_batch, batch_idx in partitions:
        high_level_subgraph = high_level_subgraph.to(device)
        low_level_batch = low_level_batch.to(device)
        batch_idx = batch_idx.to(device).reshape(-1)
        high_emb, low_emb = model.encode(high_level_subgraph, low_level_batch)   # NO gene_mask
        pooled_low = global_mean_pool(low_emb, low_level_batch.batch)
        graph_embeddings[batch_idx] = torch.cat([high_emb, pooled_low], dim=1)

X_heist = graph_embeddings.cpu().numpy()
print(f"[step1] X_heist shape = {X_heist.shape}")
assert X_heist.shape == (adata.n_obs, 2 * output_dim)
assert np.isfinite(X_heist).all(), "non-finite values in X_heist"

# --- Aliasing sanity check: gene half must vary per-cell within a cell type ---
gene_half = X_heist[:, output_dim:]
ct = adata.obs['Annotation'].astype(str).to_numpy()
example = ct[0]
sub = gene_half[ct == example]
n_unique = np.unique(np.round(sub, 5), axis=0).shape[0]
print(f"[step1] aliasing check: cell type '{example}' has {sub.shape[0]} cells, "
      f"{n_unique} distinct gene-half rows (must be >> 1 / not collapsed to a few)")

# --- Save onto pristine adata ---
adata.obsm['X_heist'] = X_heist
adata.write_h5ad(OUT_H5AD)
print(f"[step1] wrote {OUT_H5AD}")
