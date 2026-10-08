"""HEIST inference: load a pretrained model and embed a new spatial dataset.

    from utils.inference import load_heist, embed_adata
    model, vocab = load_heist("v2.0/marker_all")             # or a local folder, or "v1"
    emb = embed_adata(adata, model, vocab, save_root="data/preprocessed", name="my_sample")
    adata.obsm["X_heist"] = emb                              # (n_cells, 2 * output_dim)

The data goes through the SAME pipeline as the v2 pretraining data (utils/preprocess.py):
genes outside the vocabulary are ignored, large tissues are tiled with the same spatial partition,
and each tile gets top-200 HVGs, auto-resolution Leiden, MAGIC, a Voronoi cell graph and per-cluster
GPU KSG mutual-information gene networks. A CUDA GPU is required for the gene-network step.

Models on https://huggingface.co/HirenMadhu/HEIST:
    "v1"                 the original published checkpoint (repo root; no gene vocabulary)
    "v2.0/nomarker_all"  v2 data (Xenium + Vizgen + SEA-AD), no gene-identity embedding
    "v2.0/marker_all"    v2 data, learned per-gene embedding (gene vocabulary)
    "v2.0/marker_xenium" Xenium (v1 + Prime 5K) only, learned per-gene embedding
"""
import json
import os

import numpy as np
import torch
from torch_geometric.nn.pool import global_mean_pool
from tqdm import tqdm

from model.model import GraphEncoder
from utils.dataloader import create_dataloader
from utils.preprocess import filter_genes_to_vocab, preprocess, spatial_chunks

HF_REPO = "HirenMadhu/HEIST"
MODELS = ["v1", "v2.0/nomarker_all", "v2.0/marker_all", "v2.0/marker_xenium"]

# Settings the v2 pretraining chunks were built with (data_prep/build_chunks.py). Inference must match.
V2_PREPROCESS = dict(max_genes=200, leiden_resolution='auto', target_clusters=7, min_cluster_size=50)
BATCH_SIZE = 384  # cells per METIS partition, as in pretraining (train_v2.sh)


def _fetch(model, filename):
    """Path to `filename` for `model`: a local folder, or a subfolder of the HF repo."""
    if os.path.isdir(model):
        return os.path.join(model, filename)
    from huggingface_hub import hf_hub_download
    return hf_hub_download(HF_REPO, filename, subfolder=model)


def load_heist(model="v2.0/marker_all", device=None):
    """Load a HEIST encoder. Returns (model, gene_vocab); gene_vocab is None for "v1".

    `model` is one of MODELS or a local folder holding config.json + model.safetensors (+ the
    gene_vocab.json one level up, as written by upload_to_hf.py).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    if model == "v1":
        return GraphEncoder.from_pretrained(HF_REPO).to(device).eval(), None

    from safetensors.torch import load_file
    with open(_fetch(model, "config.json")) as f:
        config = json.load(f)
    enc = GraphEncoder(**config)
    enc.load_state_dict(load_file(_fetch(model, "model.safetensors")), strict=True)

    # One vocabulary per version (v2.0/gene_vocab.json), shared by its models.
    version_dir = os.path.dirname(model.rstrip("/"))
    with open(_fetch(version_dir, "gene_vocab.json") if version_dir else _fetch(model, "gene_vocab.json")) as f:
        vocab = json.load(f)
    return enc.to(device).eval(), vocab


@torch.no_grad()
def embed_graphs(model, graphs, batch_size=BATCH_SIZE):
    """Per-cell embeddings [cell-graph embedding | mean-pooled gene embedding] for one chunk."""
    device = next(model.parameters()).device
    n = graphs[0].num_nodes
    out = torch.zeros((n, 2 * model.final_norm.normalized_shape[0]), device=device)
    for hi, lo, idx in create_dataloader(graphs, batch_size, permute=False):
        hi, lo = hi.to(device), lo.to(device)
        high_emb, low_emb = model.encode(hi, lo)
        out[idx.to(device)] = torch.cat([high_emb, global_mean_pool(low_emb, lo.batch)], dim=1)
    return out.cpu().numpy()


def embed_adata(adata, model, gene_vocab, save_root, name, cell_type=None, spatial="spatial",
                batch_size=BATCH_SIZE):
    """Embed every cell of `adata`. Returns an (n_obs, 2*output_dim) array aligned to adata.obs_names.

    adata: raw counts in .X, coordinates in .obsm[spatial].
    cell_type: obs column to build one gene network per annotated type; None clusters with Leiden.
    gene_vocab: from load_heist(); genes outside it are ignored. None (v1) keeps all genes.
    Graphs are cached as <save_root>/<name>[_<tile>].pt; delete them to rebuild.
    """
    if spatial not in adata.obsm:
        raise KeyError(f"adata.obsm['{spatial}'] (cell coordinates) is required")
    a = adata.copy()
    if gene_vocab is not None:
        a, _ = filter_genes_to_vocab(a, gene_vocab)
    if cell_type is not None:
        a.obs["cell_type"] = a.obs[cell_type].astype(str).values
    if hasattr(a.X, "toarray"):
        a.X = a.X.toarray()
    a.X = np.asarray(a.X, dtype=np.float32)

    labels = spatial_chunks(np.asarray(a.obsm[spatial]))
    emb = None
    for tile in tqdm(sorted(set(labels)), desc="HEIST tiles", disable=len(set(labels)) == 1):
        rows = np.where(labels == tile)[0]
        graphs = preprocess(a[rows].copy(), save_root, f"{name}_{tile}" if tile else name,
                            spatial=spatial, cell_type=cell_type is not None, gene_vocab=gene_vocab,
                            **V2_PREPROCESS)
        if graphs[0].num_nodes != len(rows):
            raise RuntimeError(f"tile '{tile}': {len(rows)} cells in, {graphs[0].num_nodes} graph nodes out; "
                               f"a stale cache in {save_root}? delete {name}*.pt and rerun")
        e = embed_graphs(model, graphs, batch_size)
        if emb is None:
            emb = np.full((a.n_obs, e.shape[1]), np.nan, dtype=np.float32)
        emb[rows] = e
    return emb
