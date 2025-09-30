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

def preprocess(adata, save_root, save_file_name, max_genes = 200, spatial = 'spatial', cell_type = None):
    file_path = os.path.join(save_root, save_file_name + '.pt')
    if os.path.exists(file_path):
        graphs = torch.load(file_path, weights_only = False)
        return graphs

    sc.pp.filter_genes(adata, min_cells=3)
    sc.pp.normalize_total(adata, target_sum=1e4)
    sc.pp.log1p(adata)
    if(adata.n_vars > max_genes):
        sc.pp.highly_variable_genes(adata, n_top_genes=max_genes)
        adata = adata[:, adata.var.highly_variable]
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

    magic_operator = magic.MAGIC()
    adata.X = magic_operator.fit_transform(adata.X)

    print("Creating the GRNs using MI")

    cell_type_dict = {}
    for cell_type in cell_types:
        cell_type_data = adata[adata.obs.cell_type == cell_type]
        cell_type_dict[cell_type] = cell_type_data

    gene_network_dict = {}
    for cell_type, cell_data in tqdm(cell_type_dict.items()):
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
        gene_network_dict[cell_type] = G

        
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

    scaler = StandardScaler()
    G_cell.X = torch.from_numpy(scaler.fit_transform(adata.obsm['spatial'])) 
    graphs.append(G_cell)
    for k in tqdm(range(len(adata.obs.cell_type))):
        G_gene = gene_network_dict[adata.obs.cell_type[k]]
        G_gene.num_nodes = NUM_GENES
        G_gene.cell_type = G_cell.cell_type[k]
        G_gene.X = torch.from_numpy(adata.X[k].reshape(NUM_GENES, 1))
        graphs.append(G_gene)
    Path(save_root).mkdir(parents = True, exist_ok = True)
    torch.save(graphs, file_path)
    return graphs