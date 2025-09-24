import scanpy as sc
import magic
from sklearn.feature_selection import mutual_info_regression
from itertools import combinations
import networkx as nx
import numpy as np
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from torch_geometric.utils import from_networkx
import torch
import os
from torch_geometric.data import Data, Dataset
from glob import glob
file = glob("data/sea_raw/*")[0]
file_name = file.split('/')[-1][:-5]
file_path = 'data/sea_raw/'+file_name+'.h5ad'
adata = sc.read_h5ad(file_path)
print(f"Loaded file {file_name}.\nNow pre-processing the data.")

braak_stages = adata.obs['Braak'].values
np.save("data/sea_braak_labels/"+file_name + '_' + str(i) + ".npy", braak_stages)
sc.pp.filter_genes(adata, min_cells=3)
sc.pp.normalize_total(adata, target_sum=1e4)
sc.pp.log1p(adata)
np.save(f"data/sea_matrix/{file_name}_{str(i)}.npy", adata.X)
sc.pp.neighbors(adata, n_neighbors=10, use_rep = 'spatial')
sc.tl.leiden(adata, resolution=0.1)
cell_types = adata.obs.leiden.unique()
root = "data/sea_preprocessed/"

NUM_GENES = adata.X.shape[1]

magic_operator = magic.MAGIC()
adata.X = magic_operator.fit_transform(adata.X)

cell_types = adata.obs.leiden.unique()

print("Creating the GRNs using MI")

cell_type_dict = {}
for cell_type in cell_types:
    cell_type_data = adata[adata.obs['leiden'] == cell_type]
    cell_type_dict[cell_type] = cell_type_data

gene_network_dict = {}
for cell_type, cell_data in tqdm(cell_type_dict.items()):
    X_dense = cell_data.X.toarray() #gene-expression per cell type

    gene_names = cell_data.var.index.tolist()

    G = nx.Graph()
    
    for gene_idx_1, gene_idx_2 in combinations(range(X_dense.shape[1]), 2):
        gene_data_1 = X_dense[:, gene_idx_1]
        gene_data_2 = X_dense[:, gene_idx_2]

        mi = mutual_info_regression(gene_data_1.reshape(-1, 1), gene_data_2)[0]

        if mi > 0.35: #thresholding
            G.add_edge(gene_idx_1, gene_idx_2, weight=mi)
            G.add_edge(gene_idx_2, gene_idx_1, weight=mi)

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
G_cell = nx.from_numpy_array(adata.obsp['connectivities'])
G_cell = G_cell.to_undirected()
G_cell = from_networkx(G_cell)
G_cell.cell_type = torch.LongTensor([int(i) for i in adata.obs.leiden.values])
scaler = StandardScaler()
G_cell.X = torch.from_numpy(scaler.fit_transform(adata.obsm['spatial'])) 
graphs.append(G_cell)
for k in tqdm(range(len(adata.obs['leiden']))):
    G_gene = gene_network_dict[adata.obs['leiden'][k]]
    G_gene.num_nodes = NUM_GENES
    G_gene.cell_type = G_cell.cell_type[k]
    G_gene.X = torch.from_numpy(adata.X[k].reshape(NUM_GENES, 1))
    graphs.append(G_gene)

root = 'data/sea_preprocessed/'
torch.save(graphs, os.path.join(root, file_name + '_' + str(i) + '.pt'))
dataset = torch.load(os.path.join(root, file_name + '_' + str(i) + '.pt'))
print("Data saved")

print(f"Number of graphs in the dataset: {len(dataset)}")
