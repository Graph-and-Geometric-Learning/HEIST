import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, GINConv
from torch_geometric.nn.pool import global_mean_pool
from model.layers import MultiLevelGraphLayer
from model.pe import calculate_sinusoidal_pe


class GraphEncoder(nn.Module):
    def __init__(
        self,
        pe_dim,
        init_dim,
        hidden_dim,
        output_dim,
        num_layers,
        num_heads,
        cross_message_passing,
        positional_encoding,
    ):
        super().__init__()
        self.pe_dim = pe_dim
        self.positional_encoding = positional_encoding
        self.cross_message_passing = cross_message_passing

        # Input projections (pe_dim when using PE with addition, otherwise raw input dim)
        high_in_dim = pe_dim if positional_encoding else 2
        low_in_dim = pe_dim if positional_encoding else 1
        self.mlp_high = nn.Sequential(nn.Linear(high_in_dim, init_dim), nn.GELU())
        self.mlp_low = nn.Sequential(nn.Linear(low_in_dim, init_dim), nn.GELU())

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(MultiLevelGraphLayer(init_dim, hidden_dim, num_heads, cross_message_passing))
        for _ in range(num_layers - 2):
            self.convs.append(MultiLevelGraphLayer(hidden_dim, hidden_dim, num_heads, cross_message_passing))
        self.convs.append(MultiLevelGraphLayer(hidden_dim, output_dim, num_heads, cross_message_passing))

        self.final_norm = nn.LayerNorm(output_dim)
        self.projection_head = nn.Sequential(nn.Linear(output_dim, output_dim), nn.GELU())

    def _prepare_inputs(self, high_level_graph, low_level_graphs):
        device = high_level_graph.X.device

        if self.positional_encoding:
            high_level_graph, low_level_graphs = calculate_sinusoidal_pe(
                high_level_graph, low_level_graphs, self.pe_dim
            )
            high_level_graph.pe = high_level_graph.pe.to(device)
            low_level_graphs.pe = low_level_graphs.pe.to(device)

            # Repeat [x, y] -> [x, x, ..., x, y, y, ..., y] (d/2 times each)
            half_dim = self.pe_dim // 2
            x_repeated = high_level_graph.X[:, 0:1].repeat(1, half_dim)
            y_repeated = high_level_graph.X[:, 1:2].repeat(1, half_dim)
            high_repeated = torch.cat([x_repeated, y_repeated], dim=1)
            high_emb = high_repeated.float() + high_level_graph.pe.float()

            # Low-level: add directly (broadcasting handles [N, 1] + [N, pe_dim])
            low_emb = low_level_graphs.X.float() + low_level_graphs.pe.float()
        else:
            high_emb = high_level_graph.X.float()
            low_emb = low_level_graphs.X.float()

        return high_emb, low_emb

    def _run_conv_layers(self, high_emb, high_level_graph, low_emb, low_level_graphs):
        high_emb, low_emb = self.convs[0](high_emb, high_level_graph, low_emb, low_level_graphs)

        for layer in self.convs[1:]:
            high_emb_new, low_emb_new = layer(high_emb, high_level_graph, low_emb, low_level_graphs)
            high_emb = high_emb_new + high_emb
            low_emb = low_emb_new + low_emb

        return self.final_norm(high_emb), self.final_norm(low_emb)
  
    def forward(self, high_level_graph, low_level_graphs, high_mask=None, low_mask=None):
        if high_mask is not None and low_mask is not None:
            high_level_graph.X = high_level_graph.X * high_mask
            low_level_graphs.X = low_level_graphs.X * low_mask

        high_emb, low_emb = self._prepare_inputs(high_level_graph, low_level_graphs)
        high_emb, low_emb = self.mlp_high(high_emb), self.mlp_low(low_emb)
        high_emb, low_emb = self._run_conv_layers(high_emb, high_level_graph, low_emb, low_level_graphs)
        return self.projection_head(high_emb), self.projection_head(low_emb)

    def encode(self, high_level_graph, low_level_graphs, gene_mask=None):
        if gene_mask is not None:
            low_level_graphs.X = low_level_graphs.X * gene_mask

        high_emb, low_emb = self._prepare_inputs(high_level_graph, low_level_graphs)
        high_emb, low_emb = self.mlp_high(high_emb), self.mlp_low(low_emb)

        return self._run_conv_layers(high_emb, high_level_graph, low_emb, low_level_graphs)


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        if num_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, X):
        for i in range(len(self.layers) - 1):
            X = F.relu(self.layers[i](X))
        return self.layers[-1](X)


class MLP_multihead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2, num_layers):
        super().__init__()
        self.bn = nn.BatchNorm1d(input_dim)
        if num_layers == 1:
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim1)])
            self.layers.append(nn.Linear(input_dim, output_dim2))
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for _ in range(num_layers - 2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim1))
            self.layers.append(nn.Linear(hidden_dim, output_dim2))

    def forward(self, X, batch):
        X = self.bn(X)
        for i in range(len(self.layers) - 2):
            X = F.relu(self.layers[i](X))
        return global_mean_pool(self.layers[-2](X), batch), self.layers[-1](X)


class GIN_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([GINConv(nn.Linear(input_dim, hidden_dim), train_eps=True)])
        for _ in range(num_layers - 1):
            self.layers.append(GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        self.high_mlp = nn.Linear(hidden_dim, 2)
        self.low_mlp = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward(self, high_emb, high_graph, low_emb, low_graph):
        for layer in self.layers:
            high_emb = layer(high_emb, high_graph.edge_index).relu()
            low_emb = layer(low_emb, low_graph.edge_index).relu()
        high_emb = F.dropout(high_emb, p=0.5, training=self.training)
        low_emb = F.dropout(low_emb, p=0.5, training=self.training)
        high_emb = self.high_mlp(high_emb)
        low_emb = torch.abs(self.low_mlp(low_emb))
        alpha = torch.sigmoid(self.alpha)
        return high_emb, low_emb, alpha


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([GINConv(nn.Linear(input_dim, hidden_dim), train_eps=True)])
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x, edge_index, batch):
        x = self.batch_norm(x)
        for i in range(len(self.layers) - 1):
            x = self.layers[i](x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        return x


class GraphTrans(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([TransformerConv(input_dim, hidden_dim // num_heads, heads=num_heads)])
        for _ in range(num_layers - 2):
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index, batch):
        x = self.layers[0](x, edge_index)
        for i in range(1, len(self.layers) - 1, 2):
            x = self.layers[i](x)
            x = self.layers[i + 1](x, edge_index).relu()
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        return x
