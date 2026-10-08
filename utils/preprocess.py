import scanpy as sc
import pandas as pd
import numpy as np
import scipy.sparse as sp
from pathlib import Path
import scanpy as sc
import magic
from utils.build_cell_graph import calcualte_voronoi_from_coords, build_graph_from_cell_coords, assign_attributes
from utils.create_coexpression_networks import build_gene_network_gpu
from sklearn.preprocessing import LabelEncoder
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data
import torch
import os
import pandas as pd

def add_self_loops(graph):
    if graph.edge_index.shape[1] == 0:
        # Add self-loops for all nodes
        num_nodes = graph.num_nodes
        self_loops = torch.arange(0, num_nodes, dtype=torch.long).repeat(2, 1)
        graph.edge_index = self_loops
        graph.edge_attr = torch.ones((num_nodes, 1), dtype=torch.float)
    return graph

# Spatial chunking used for the v2 pretraining data; inference must tile large tissues the same way.
MAX_CHUNK = 35000     # split tiles until each has < MAX_CHUNK cells...
SPLIT_ABOVE = 30000   # ...but only for samples with more than SPLIT_ABOVE cells


def partition(xy, max_size=MAX_CHUNK):
    """Recursive quadrant split at each tile's bounding-box midpoint (the v1/v2 spatial chunking).

    Labels are digit strings, one digit (1-4) per level, matching the pretraining file suffixes (e.g.
    '133'). Quadrants: 1 = (x<mid, y<mid), 2 = (x<mid, y>=mid), 3 = (x>=mid, y<mid), 4 = (x>=mid, y>=mid).
    Differs from v1 in one way: v1 stopped as soon as the FIRST tile was < max_size, which could leave
    other tiles far larger; here every tile is split until it is < max_size.
    """
    labels = np.full(len(xy), '', dtype=object)
    todo = [np.arange(len(xy))]
    while todo:
        idx = todo.pop()
        if len(idx) < max_size:
            continue
        p = xy[idx]
        mid = (p.min(0) + p.max(0)) / 2
        q = 1 + 2 * (p[:, 0] >= mid[0]) + (p[:, 1] >= mid[1])
        for k in range(1, 5):
            sub = idx[q == k]
            labels[sub] = labels[sub] + str(k)
            todo.append(sub)
    return labels


def spatial_chunks(xy):
    """Chunk labels for a whole sample: one chunk ('') unless it has more than SPLIT_ABOVE cells."""
    if len(xy) > SPLIT_ABOVE:
        return partition(np.asarray(xy))
    return np.full(len(xy), '', dtype=object)


def filter_genes_to_vocab(adata, gene_vocab, verbose=True):
    """Keep only genes in the pretraining vocabulary (matched by upper-case symbol); drop the rest.

    v2 models embed each gene by its vocabulary index, so a gene the model never saw has no embedding
    and is IGNORED rather than mapped to an arbitrary row. Control probes (BLANK-*, NegControl*,
    *Codeword*, ...) are never in the vocabulary, so they are dropped here too.
    Returns (filtered AnnData copy, list of dropped gene names).
    """
    names = adata.var_names.astype(str)
    known = np.array([g.upper() in gene_vocab for g in names])
    dropped = names[~known].tolist()
    if not known.any():
        raise ValueError('None of the genes in this dataset are in the HEIST gene vocabulary '
                         f'({len(gene_vocab)} human gene symbols). Mouse symbols must be converted '
                         'to human orthologs first.')
    if verbose and dropped:
        print(f'[HEIST] ignoring {len(dropped)}/{len(names)} genes not in the vocabulary, e.g. {dropped[:8]}')
    return adata[:, known].copy(), dropped


def preprocess(adata, save_root, save_file_name, max_genes = 200, spatial = 'spatial', cell_type = None,
               leiden_resolution = 0.5, gene_vocab = None, target_clusters = 7, min_kept_counts = 0,
               min_cluster_size = 0):
    file_path = os.path.join(save_root, save_file_name + '.pt')
    if os.path.exists(file_path):
        graphs = torch.load(file_path, weights_only = False)
        return graphs

    sc.pp.filter_genes(adata, min_cells=3)
    if min_kept_counts:
        adata.layers['counts'] = adata.X.copy()  # raw counts, for the kept-gene filter below
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if(adata.n_vars > max_genes):
        sc.pp.highly_variable_genes(adata, n_top_genes=max_genes)
        adata = adata[:, adata.var.highly_variable]
    if min_kept_counts:
        kc = adata.layers['counts'].sum(1)
        kc = np.asarray(kc).ravel()
        adata = adata[kc >= min_kept_counts].copy()
        del adata.layers['counts']
    if cell_type:
        cell_types = adata.obs.cell_type.unique()
    else:
        sc.pp.pca(adata, n_comps=min(50, adata.n_vars - 1))
        sc.pp.neighbors(adata)
        if leiden_resolution == 'auto':
            best = None
            for r in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
                sc.tl.leiden(adata, resolution=r, flavor='igraph', n_iterations=2, directed=False,
                             random_state=0)
                # count only clusters that survive the small-cluster merge below
                k = int((adata.obs.leiden.value_counts() >= max(min_cluster_size, 1)).sum())
                if best is None or abs(k - target_clusters) < abs(best[1] - target_clusters):
                    best = (r, k)
                if k >= target_clusters:
                    break  # cluster count grows with resolution; finer ones only move further away
            leiden_resolution = best[0]
        sc.tl.leiden(adata, resolution=leiden_resolution, flavor='igraph', n_iterations=2, directed=False,
                     random_state=0)
        adata.uns['leiden_resolution_used'] = leiden_resolution
        if min_cluster_size:
            lab = adata.obs['leiden'].astype(str).to_numpy()
            sizes = pd.Series(lab).value_counts()
            big = sizes.index[sizes >= min_cluster_size]
            small = np.isin(lab, sizes.index[sizes < min_cluster_size])
            if small.any() and len(big):
                onehot = sp.csr_matrix((np.ones((~small).sum()), (np.where(~small)[0],
                                        pd.Categorical(lab[~small], categories=big).codes)),
                                       shape=(len(lab), len(big)))
                w = (adata.obsp['connectivities'][small] @ onehot).toarray()
                # isolated cells (no link to any big cluster) go to the largest cluster
                lab[small] = np.where(w.sum(1) > 0, big[w.argmax(1)], big[0])
                adata.obs['leiden'] = pd.Categorical(lab)
        cell_types = adata.obs.leiden.unique()
        adata.obs['cell_type'] = adata.obs['leiden']

    coordinates = adata.obsm[spatial]
    coordinates = coordinates - coordinates.min(axis=0)
    xmax, ymax = coordinates.max(axis=0)
    voronoi_polygons = calcualte_voronoi_from_coords(coordinates[:, 0], coordinates[:, 1])
    cell_data = pd.DataFrame(np.c_[adata.obs.index, coordinates], columns=['CELL_ID', 'X', 'Y'])
    G_cell, node_to_cell_mapping = build_graph_from_cell_coords(cell_data, voronoi_polygons)
    G_cell = assign_attributes(G_cell, cell_data, node_to_cell_mapping)

    NUM_GENES = adata.X.shape[1]

    magic_operator = magic.MAGIC()
    adata.X = magic_operator.fit_transform(adata.X)

    print("Creating the GRNs using MI")

    cell_type_dict = {}
    for ct in cell_types:
        cell_type_data = adata[adata.obs.cell_type == ct]
        cell_type_dict[ct] = cell_type_data

    gene_network_dict = {}
    for ct, cell_data in tqdm(cell_type_dict.items()):
        edges, weights, gene_names = build_gene_network_gpu(
            cell_data,
            topk_per_gene=200,     # tune: higher -> more candidates (slower, more edges)
            min_abs_corr=None,     # OR set e.g. 0.2 and reduce topk usage
            mi_bins=32,            # tune: 16-64; more bins -> slower but more precise
            mi_batch_size=20000,   # tune to fit GPU memory
            device="cuda"
        )

        G = nx.Graph()
        G.add_nodes_from(range(len(gene_names)))
        G.add_weighted_edges_from(
            [(int(i), int(j), float(w)) for (i,j), w in zip(edges, weights)]
        )
        G = G.to_undirected()
        gene_network_dict[ct] = G

        
    for k in gene_network_dict:
        gene_network_dict[k] = from_networkx(gene_network_dict[k])
        
    print("Converting to PyG format")
    # The format should be as following
    #   List of PyG graphs [high_level_graph, low_level_graph_0, ...., low_level_graph_N]
    #   low_level_graph_i refers to low-level graph of ith cell
    #   The initial features for cell are the spatial location
    #   Initial features genes graph i, are just the gene-expression for cell i
    #       Remember to reshape them to (NUM_GENES, 1). 
    #   Initial features should be called X in the graphs.
    #Save this list of graphs.
    NUM_GENES = adata.X.shape[1]
    graphs = []
    G_cell = G_cell.to_undirected()
    G_cell = add_self_loops(from_networkx(G_cell))

    le = LabelEncoder()
    G_cell.cell_type = torch.from_numpy(le.fit_transform(adata.obs.cell_type))
    G_cell.cell_types = le.classes_

    # Gene identity of every gene-graph node (node i == column i of adata.X, after filtering/HVG),
    # so a model can learn gene embeddings shared across chunks with different panels/HVG sets.
    # gene_vocab maps upper-case symbol -> global index (data/pretraining_v2/gene_vocab.json).
    G_cell.gene_names = [str(g) for g in adata.var_names]
    G_cell.leiden_resolution = adata.uns.get('leiden_resolution_used')  # None when types were given
    if gene_vocab is not None:
        G_cell.gene_ids = torch.tensor([gene_vocab[g.upper()] for g in G_cell.gene_names], dtype=torch.long)

    scaler = StandardScaler()
    G_cell.X = torch.from_numpy(scaler.fit_transform(adata.obsm['spatial'])) 
    graphs.append(G_cell)
    # Positional access via numpy. `adata.obs.cell_type[k]` is LABEL-based indexing: with string
    # barcodes pandas currently falls back to positional (deprecated, removed in pandas 3.0), and with
    # an integer index it silently returns the wrong cell or raises KeyError.
    cell_type_arr = adata.obs.cell_type.to_numpy()
    X_all = adata.X
    for k in tqdm(range(len(cell_type_arr))):
        base = gene_network_dict[cell_type_arr[k]]
        # A fresh Data per cell. Indexing gene_network_dict returns the ONE shared graph for that
        # cell type, so assigning .X to it mutates that shared object -- every cell of a type would
        # end up holding the LAST cell's expression, collapsing the gene branch to one vector per
        # cell type. Topology is read-only and safe to share; only X/cell_type are per-cell.
        G_gene = Data(num_nodes=NUM_GENES)
        G_gene.edge_index = base.edge_index
        if getattr(base, 'weight', None) is not None:
            G_gene.weight = base.weight
        if getattr(base, 'edge_attr', None) is not None:
            G_gene.edge_attr = base.edge_attr
        G_gene.cell_type = G_cell.cell_type[k]
        if gene_vocab is not None:
            # Per-node GLOBAL gene index. GraphEncoder._marker_ids reads `marker_id` when present, so
            # --marker_embedding (num_markers = len(gene_vocab)) indexes by gene identity instead of
            # node position, which differs across chunks. One shared tensor: torch.save stores it once.
            G_gene.marker_id = G_cell.gene_ids
        G_gene.X = torch.from_numpy(np.asarray(X_all[k]).reshape(NUM_GENES, 1))
        graphs.append(G_gene)
    Path(save_root).mkdir(parents = True, exist_ok = True)
    torch.save(graphs, file_path)
    return graphs