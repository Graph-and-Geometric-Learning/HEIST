import torch
from torch_geometric.data import DataLoader, Dataset
import torch_geometric.transforms as T
import torch
import pymetis
from torch_geometric.utils import to_networkx, subgraph, add_self_loops, k_hop_subgraph
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import random
from torch.utils.data.distributed import DistributedSampler
import networkx as nx
import math
import os
 
def shuffle_node_indices(data):
    perm = torch.randperm(data.num_nodes)  # Random permutation of node indices
    data.X = data.X[perm]  # Shuffle node features
    # Shuffle edge indices
    row, col = data.edge_index
    row = perm[row]
    col = perm[col]
    data.edge_index = torch.stack([row, col], dim=0)
    return data, perm

class CustomDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class EmbeddingDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        embedding = self.embeddings[idx]
        label = self.labels[idx]
        return embedding, label

def create_dataloader(graphs, batch_size, permute = True):
    high_level_graph = graphs[0]
    num_nodes = high_level_graph.num_nodes
    low_level_graphs = graphs[1:]

    # Assign default edge weights if missing
    for idx in range(len(low_level_graphs)):
        if 'weight' not in low_level_graphs[idx] or low_level_graphs[idx].edge_attr is None:
            num_edges = low_level_graphs[idx].edge_index.size(1)
            low_level_graphs[idx].weight = torch.ones((num_edges, 1), dtype=torch.float)

    # Shuffle node indices if required
    if permute:
        high_level_graph, perm = shuffle_node_indices(high_level_graph)

    # Convert high-level graph to NetworkX for partitioning
    G = to_networkx(high_level_graph, to_undirected=True)

    # Partitioning using Metis
    num_partitions = math.ceil(num_nodes / batch_size)
    adjacency_list = [list(G.neighbors(node)) for node in range(G.number_of_nodes())]
    _, node_partitions = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

    # Create partitions
    partitions = [[] for _ in range(num_partitions)]
    for node_idx, part_idx in enumerate(node_partitions):
        partitions[part_idx].append(node_idx)

    # Create high-level subgraphs and low-level batches for each partition
    partitioned_batches = []
    for partition_node_indices in partitions:
        partition_node_indices = torch.tensor(partition_node_indices, dtype=torch.long)
        edge_index_partition, _ = subgraph(partition_node_indices, high_level_graph.edge_index, relabel_nodes=True)

        high_level_subgraph = Data(
            X=high_level_graph.X[partition_node_indices],
            edge_index=edge_index_partition,
            y=high_level_graph.y[partition_node_indices] if high_level_graph.y is not None else None,
            num_nodes=len(partition_node_indices),
            train_mask=high_level_graph.train_mask[partition_node_indices] if hasattr(high_level_graph, "train_mask") else None,
            test_mask=high_level_graph.test_mask[partition_node_indices] if hasattr(high_level_graph, "train_mask") else None,
            val_mask=high_level_graph.val_mask[partition_node_indices] if hasattr(high_level_graph, "train_mask") else None,
        )

        # Assign corresponding low-level graphs
        if permute:
            corresponding_low_level_graphs = [low_level_graphs[node_idx] for node_idx in perm[partition_node_indices]]
        else:
            corresponding_low_level_graphs = [low_level_graphs[node_idx] for node_idx in partition_node_indices]
        try:
        # Create a batch from low-level graphs
            low_level_batch = Batch.from_data_list(corresponding_low_level_graphs)
        except KeyError as e:
            # print(f"[Rank {rank}] Caught KeyError in low-level batch creation: {e}", flush=True)
            # print(f"[Rank {rank}] Low-level graphs: {corresponding_low_level_graphs}", flush=True)
            # raise  
            raise
        # Store high-level subgraph and corresponding low-level batch
        partitioned_batches.append((high_level_subgraph, low_level_batch, partition_node_indices))
    # Step 6: Define DataLoader for batched subgraphs
    batch_size = 1
    dataloader = DataLoader(partitioned_batches, batch_size=batch_size, shuffle=True)

    return dataloader


def create_dataloader_ddp(graphs, batch_size, rank, world_size, permute=True):
    """
    Creates a PyTorch DataLoader for distributed training with PyTorch's DistributedDataParallel (DDP).
    
    Args:
        graphs (list): List of PyG graphs [high_level_graph, low_level_graphs].
        batch_size (int): Number of nodes per partition.
        rank (int): Process rank in DDP.
        world_size (int): Total number of processes.
        permute (bool): Whether to shuffle node indices before partitioning.

    Returns:
        DataLoader: PyTorch DataLoader for the partitioned subgraphs.
    """

    # Extract high-level and low-level graphs
    high_level_graph = graphs[0]
    num_nodes = high_level_graph.num_nodes
    low_level_graphs = graphs[1:]

    # Assign default edge weights if missing
    for idx in range(len(low_level_graphs)):
        if 'weight' not in low_level_graphs[idx] or low_level_graphs[idx].edge_attr is None:
            num_edges = low_level_graphs[idx].edge_index.size(1)
            low_level_graphs[idx].weight = torch.ones((num_edges, 1), dtype=torch.float)

    # Shuffle node indices if required
    if permute:
        high_level_graph, perm = shuffle_node_indices(high_level_graph)

    # Convert high-level graph to NetworkX for partitioning
    G = to_networkx(high_level_graph, to_undirected=True)

    # Partitioning using Metis
    num_partitions = math.ceil(num_nodes / batch_size)
    adjacency_list = [list(G.neighbors(node)) for node in range(G.number_of_nodes())]
    _, node_partitions = pymetis.part_graph(num_partitions, adjacency=adjacency_list)

    # Create partitions
    partitions = [[] for _ in range(num_partitions)]
    for node_idx, part_idx in enumerate(node_partitions):
        partitions[part_idx].append(node_idx)

    # Create high-level subgraphs and low-level batches for each partition
    partitioned_batches = []
    for partition_node_indices in partitions:
        partition_node_indices = torch.tensor(partition_node_indices, dtype=torch.long)
        edge_index_partition, _ = subgraph(partition_node_indices, high_level_graph.edge_index, relabel_nodes=True)

        high_level_subgraph = Data(
            X=high_level_graph.X[partition_node_indices],
            edge_index=edge_index_partition,
            y=high_level_graph.y[partition_node_indices] if high_level_graph.y is not None else None,
            num_nodes=len(partition_node_indices)
        )

        # Assign corresponding low-level graphs
        if permute:
            corresponding_low_level_graphs = [low_level_graphs[node_idx] for node_idx in perm[partition_node_indices]]
        else:
            corresponding_low_level_graphs = [low_level_graphs[node_idx] for node_idx in partition_node_indices]
        try:
        # Create a batch from low-level graphs
            low_level_batch = Batch.from_data_list(corresponding_low_level_graphs)
        except KeyError as e:
            # print(f"[Rank {rank}] Caught KeyError in low-level batch creation: {e}", flush=True)
            # print(f"[Rank {rank}] Low-level graphs: {corresponding_low_level_graphs}", flush=True)
            # raise  
            raise
        # Store high-level subgraph and corresponding low-level batch
        partitioned_batches.append((high_level_subgraph, low_level_batch, partition_node_indices))

    # Use DistributedSampler to distribute data across GPUs
    sampler = DistributedSampler(partitioned_batches, num_replicas=world_size, rank=rank, shuffle=True)

    # Create DataLoader with batch_size=1 (each batch is a partition)
    dataloader = DataLoader(partitioned_batches, batch_size=1, shuffle=False, sampler=sampler)

    return dataloader
