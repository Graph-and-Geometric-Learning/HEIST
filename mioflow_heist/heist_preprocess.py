"""
Local corrected copy of utils/preprocess.py for the MIOFlow-HEIST experiment.

The ONLY functional change vs the shipped utils/preprocess.py is the per-cell
low-level-graph assembly loop at the bottom. The original does:

    G_gene = gene_network_dict[adata.obs.cell_type[k]]   # shared object per type
    G_gene.X = torch.from_numpy(adata.X[k].reshape(NUM_GENES, 1))
    graphs.append(G_gene)

Because `gene_network_dict[type]` returns ONE shared PyG Data per cell type, every
cell of a given type ends up referencing the same object with the LAST cell's X, so
the gene half of the embedding collapses to one vector per cell type. (`copy.copy`
does NOT fix this: PyG Data shares its underlying `_store`, so setting `.X` on the
shallow copy still mutates the original.)

Fix: build a fresh `Data` per cell that SHARES the read-only GRN topology
(`edge_index`/`weight`/`edge_attr` of that cell type) but gets its OWN `X`.

Everything else is identical to the shipped preprocess (imports from utils.*).
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import magic
import networkx as nx
import torch
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

from utils.build_cell_graph import (
    calcualte_voronoi_from_coords,
    build_graph_from_cell_coords,
    assign_attributes,
)
from utils.create_coexpression_networks import build_gene_network_gpu


def add_self_loops(graph):
    if graph.edge_index.shape[1] == 0:
        num_nodes = graph.num_nodes
        self_loops = torch.arange(0, num_nodes, dtype=torch.long).repeat(2, 1)
        graph.edge_index = self_loops
        graph.edge_attr = torch.ones((num_nodes, 1), dtype=torch.float)
    return graph


def preprocess(adata, save_root, save_file_name, max_genes=200, spatial='spatial', cell_type=None):
    file_path = os.path.join(save_root, save_file_name + '.pt')
    if os.path.exists(file_path):
        graphs = torch.load(file_path, weights_only=False)
        return graphs

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if adata.n_vars > max_genes:
        sc.pp.highly_variable_genes(adata, n_top_genes=max_genes)
        adata = adata[:, adata.var.highly_variable].copy()   # materialize (avoid view-assignment issues)
    if cell_type:
        cell_types = adata.obs.cell_type
    else:
        sc.tl.leiden(adata, resolution=0.1)
        cell_types = adata.obs.leiden.unique()
        adata.obs.cell_type = adata.obs.leiden

    coordinates = adata.obsm[spatial]
    coordinates = coordinates - coordinates.min(axis=0)
    xmax, ymax = coordinates.max(axis=0)
    voronoi_polygons = calcualte_voronoi_from_coords(coordinates[:, 0], coordinates[:, 1])
    cell_data = pd.DataFrame(np.c_[adata.obs.index, coordinates], columns=['CELL_ID', 'X', 'Y'])
    G_cell, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, voronoi_polygons)
    G_cell = assign_attributes(G_cell, cell_data, node_to_cell_mapping)

    NUM_GENES = adata.X.shape[1]

    import scipy.sparse as sp
    magic_operator = magic.MAGIC()
    X_magic = magic_operator.fit_transform(adata.X)
    if hasattr(X_magic, 'values'):        # pandas DataFrame
        X_magic = X_magic.values
    if sp.issparse(X_magic):
        X_magic = X_magic.toarray()
    X_magic = np.asarray(X_magic, dtype=np.float32)   # dense (n_cells, n_genes)
    adata.X = X_magic

    print("Creating the GRNs using MI")

    cell_type_dict = {}
    for ct in cell_types:
        cell_type_data = adata[adata.obs.cell_type == ct]
        cell_type_dict[ct] = cell_type_data

    gene_network_dict = {}
    for ct, cdata in tqdm(cell_type_dict.items()):
        edges, weights, gene_names = build_gene_network_gpu(
            cdata,
            topk_per_gene=200,
            min_abs_corr=None,
            mi_bins=32,
            mi_batch_size=20000,
            device="cuda",
        )

        G = nx.Graph()
        G.add_nodes_from(range(len(gene_names)))
        G.add_weighted_edges_from(
            [(int(i), int(j), float(w)) for (i, j), w in zip(edges, weights)]
        )
        G = G.to_undirected()
        gene_network_dict[ct] = G

    for k in gene_network_dict:
        gene_network_dict[k] = from_networkx(gene_network_dict[k])

    print("Converting to PyG format")
    NUM_GENES = adata.X.shape[1]
    graphs = []
    G_cell = G_cell.to_undirected()
    G_cell = add_self_loops(from_networkx(G_cell))

    le = LabelEncoder()
    G_cell.cell_type = torch.from_numpy(le.fit_transform(adata.obs.cell_type))
    G_cell.cell_types = le.classes_

    scaler = StandardScaler()
    G_cell.X = torch.from_numpy(scaler.fit_transform(adata.obsm['spatial']))
    graphs.append(G_cell)

    cell_type_arr = adata.obs.cell_type.to_numpy()
    for k in tqdm(range(len(cell_type_arr))):
        base = gene_network_dict[cell_type_arr[k]]
        # Fresh Data per cell: share read-only GRN topology, own per-cell X.
        G_gene = Data(num_nodes=NUM_GENES)
        G_gene.edge_index = base.edge_index
        if getattr(base, 'weight', None) is not None:
            G_gene.weight = base.weight
        if getattr(base, 'edge_attr', None) is not None:
            G_gene.edge_attr = base.edge_attr
        G_gene.cell_type = G_cell.cell_type[k]
        G_gene.X = torch.from_numpy(X_magic[k].reshape(NUM_GENES, 1))
        graphs.append(G_gene)

    Path(save_root).mkdir(parents=True, exist_ok=True)
    torch.save(graphs, file_path)
    return graphs
