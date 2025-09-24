# augment.py
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected, add_self_loops, dropout_edge
from torch_geometric.data import Batch
import random

class GraphAugmentor:
    def __init__(self, edge_drop_rate=0.2, feature_mask_rate=0.3):
        """
        Initialize the augmentor with specified rates.
        Args:
            edge_drop_rate (float): The probability of dropping each edge.
            feature_mask_rate (float): The probability of masking each feature.
        """
        self.edge_drop_rate = edge_drop_rate
        self.feature_mask_rate = feature_mask_rate
        self.transform_high = T.AddRandomWalkPE(walk_length=2, attr_name='pe')
        self.transform_low = T.AddRandomWalkPE(walk_length=1, attr_name='pe')

    def augment(self, graph):
        """
        Create two different augmentations of the input graph.
        Args:
            graph (torch_geometric.data.Data): The input graph to augment.
        Returns:
            (Data, Data): Two augmented versions of the input graph.
        """
        aug1 = self._drop_edges(graph)
        # aug1 = self._mask_features(aug1)
        aug2 = self._drop_edges(graph)
        # aug2 = self._mask_features(aug2)
        return aug1, aug2

    def _drop_edges(self, graph):
        """
        Drop edges from the graph.
        Args:
            graph (torch_geometric.data.Data): The input graph.
        Returns:
            Data: The graph with edges dropped.
        """
        edge_index, _ = dropout_edge(graph.edge_index, p=self.edge_drop_rate, force_undirected=True)
        aug_graph = Data(X=graph.X, edge_index=edge_index, num_nodes = graph.X.shape[0])#, cell_type = graph.cell_type)
        if(aug_graph.X.shape[1]==2):
            aug_graph = self.transform_high(aug_graph)
        else:
            aug_graph = self.transform_low(aug_graph)
        if hasattr(graph, 'y'):
            aug_graph.y = graph.y
        return aug_graph

    def _mask_features(self, graph):
        """
        Mask features of the graph.
        Args:
            graph (torch_geometric.data.Data): The input graph.
        Returns:
            Data: The graph with features masked.
        """
        x = graph.x.clone()
        mask = torch.rand(x.shape) < self.feature_mask_rate
        x[mask] = 0
        aug_graph = Data(X=x, edge_index=graph.edge_index)
        if hasattr(graph, 'y'):
            aug_graph.y = graph.y
        return aug_graph

    def augment_batch(self, batch):
        """
        Augment a batched graph without connecting the batches.
        Args:
            batch (torch_geometric.data.Batch): The input batched graph to augment.
        Returns:
            (Batch, Batch): Two augmented versions of the input batched graph.
        """
        aug_graphs_1 = []
        aug_graphs_2 = []
        batch_idx = batch.batch

        for graph in batch.to_data_list():
            aug1, aug2 = self.augment(graph)
            aug_graphs_1.append(aug1)
            aug_graphs_2.append(aug2)

        batch_aug_1 = Batch.from_data_list(aug_graphs_1)
        batch_aug_2 = Batch.from_data_list(aug_graphs_2)
        batch_aug_1.batch = batch_idx
        batch_aug_2.batch = batch_idx
        
        return batch_aug_1, batch_aug_2