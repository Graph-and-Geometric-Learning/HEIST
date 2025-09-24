import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import TransformerConv, GINConv, GINEConv
from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.utils import get_laplacian
import numpy as np
from torch_scatter import scatter
import torch_geometric as tg
import matplotlib.pyplot as plt
from model.layers import MultiLevelGraphLayer, HierarchicalBlending, HeirarchicalBlendingWavelet
from model.pe import calculate_sinusoidal_pe, calculate_anchor_pe

def get_dirichlet_energy_normalised(hidden, edge_index): 
    deg = tg.utils.degree(edge_index[0])
    deg = torch.sqrt(deg + torch.ones_like(deg))
    deg = deg.to(hidden.device)
    
    diff = (hidden/deg[:,None])[edge_index[0]] - (hidden/deg[:,None])[edge_index[1]]
    dist = torch.norm(diff, dim=-1)**2
    neigh_sum = scatter(dist, edge_index[0], dim=0, reduce='sum')
    
    batch = torch.zeros(neigh_sum.size(0)).long()
    batch = batch.to(neigh_sum.device)
    
    neigh_sum = neigh_sum.unsqueeze(-1)
    
    return torch.sqrt(global_mean_pool(neigh_sum, batch=batch)).cpu().detach().item()

class GraphEncoder(nn.Module):
    def __init__(self, pe_dim, init_dim, hidden_dim, output_dim, num_layers, num_heads, cross_message_passing, positional_encoding, blending):
        super(GraphEncoder, self).__init__()
        self.blending = blending
        self.pe_dim = pe_dim
        self.cross_message_passing, self.positional_encoding = cross_message_passing, positional_encoding
        self.pe_fn = calculate_sinusoidal_pe
        if(self.positional_encoding):
            self.mlp_high = nn.Sequential(nn.Linear(2 + self.pe_dim, init_dim), nn.GELU())
            self.mlp_low = nn.Sequential(nn.Linear(1 + self.pe_dim, init_dim), nn.GELU())
        else:
            self.mlp_high = nn.Sequential(nn.Linear(2, init_dim), nn.GELU())
            self.mlp_low = nn.Sequential(nn.Linear(1, init_dim), nn.GELU())
        self.convs = nn.ModuleList()

        self.convs.append(MultiLevelGraphLayer(init_dim, hidden_dim, num_heads, self.cross_message_passing))
        
        for _ in range(num_layers - 2):
            self.convs.append(MultiLevelGraphLayer(hidden_dim, hidden_dim, num_heads, self.cross_message_passing))
        
        # Final layer
        self.convs.append(MultiLevelGraphLayer(hidden_dim, output_dim, num_heads, self.cross_message_passing))
        self.norm = nn.LayerNorm(output_dim)

        self.projection_head = nn.Sequential(nn.Linear(output_dim, output_dim), nn.GELU())
        self.beta = nn.Parameter(torch.tensor(0.0))  
        self.hierarchical_blending = HierarchicalBlending(pe_dim, num_heads)


    def forward(self, high_level_graph, low_level_graphs, high_mask=None, low_mask=None):# -> tuple:
        if high_mask is not None and low_mask is not None:
            high_level_graph.X = high_level_graph.X * high_mask
            low_level_graphs.X = low_level_graphs.X * low_mask
    
        if(self.positional_encoding):
            high_level_graph, low_level_graphs = self.pe_fn(high_level_graph, low_level_graphs, self.pe_dim)
            high_level_graph.pe = high_level_graph.pe.to(high_level_graph.X.device)
            low_level_graphs.pe = low_level_graphs.pe.to(high_level_graph.X.device)
        
            high_emb_in = torch.cat([high_level_graph.X.float(), high_level_graph.pe.float()], 1)
            low_emb_in = torch.cat([low_level_graphs.X.float(), low_level_graphs.pe.float()], 1)
        else:
            high_emb_in = high_level_graph.X.float()
            low_emb_in = low_level_graphs.X.float()
        high_emb_in, low_emb_in = self.mlp_high(high_emb_in), self.mlp_low(low_emb_in)
        if self.blending:
            high_emb_in, low_emb_in = self.hierarchical_blending(high_emb_in, low_emb_in, low_level_graphs.batch)# -> tuple
        high_emb, low_emb= self.convs[0](high_emb_in, high_level_graph, low_emb_in, low_level_graphs)

        for i, layer in enumerate(self.convs[1:]):
            high_emb_new, low_emb_new = layer(high_emb, high_level_graph, low_emb, low_level_graphs)
            
            high_emb = high_emb_new + high_emb
            low_emb = low_emb_new + low_emb
        # high_emb, low_emb = self.convs[-1](high_emb+high_emb_in, high_level_graph, low_emb+low_emb_in, low_level_graphs)
        return self.norm(high_emb), self.norm(low_emb)#, aux_loss

    def encode(self, high_level_graph, low_level_graphs, gene_mask = None):
        if gene_mask is not None:
            low_level_graphs.X = low_level_graphs.X * gene_mask
    
        if(self.positional_encoding):
            high_level_graph, low_level_graphs = self.pe_fn(high_level_graph, low_level_graphs, self.pe_dim)
            high_level_graph.pe = high_level_graph.pe.to(high_level_graph.X.device)
            low_level_graphs.pe = low_level_graphs.pe.to(high_level_graph.X.device)
        
            high_emb_in = torch.cat([high_level_graph.X.float(), high_level_graph.pe.float()], 1)
            low_emb_in = torch.cat([low_level_graphs.X.float(), low_level_graphs.pe.float()], 1)
        else:
            high_emb = high_level_graph.X.float()
            low_emb = low_level_graphs.X.float()
        if self.blending:
            high_emb, low_emb = self.hierarchical_blending(high_emb, low_emb, low_level_graphs.batch)# -> tuple
        high_emb_in, low_emb_in = self.mlp_high(high_emb_in), self.mlp_low(low_emb_in)
        high_emb, low_emb= self.convs[0](high_emb_in, high_level_graph, low_emb_in, low_level_graphs)

        for i, layer in enumerate(self.convs[1:]):
            high_emb_new, low_emb_new = layer(high_emb, high_level_graph, low_emb, low_level_graphs)
            
            high_emb = high_emb_new + high_emb
            low_emb = low_emb_new + low_emb
        # high_emb, low_emb = self.convs[-1](high_emb+high_emb_in, high_level_graph, low_emb+low_emb_in, low_level_graphs)
        return self.norm(high_emb), self.norm(low_emb)#, aux_loss

    
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.sf = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(input_dim)
        self.ln = nn.LayerNorm(input_dim)
        if(num_layers==1):
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for i in range(num_layers-2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, X):
        # X = self.bn(X)
        for i in range(len(self.layers)-1):
            X = F.relu(self.layers[i](X))
        return self.layers[-1](X)

class MLP_multihead(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim1, output_dim2, num_layers):
        super(MLP_multihead, self).__init__()
        self.sf = nn.Softmax(dim=1)
        self.bn = nn.BatchNorm1d(input_dim)
        self.ln = nn.LayerNorm(input_dim)
        if(num_layers==1):
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim1)])
            self.layers.append(nn.Linear(input_dim, output_dim2))
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for i in range(num_layers-2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim1))
            self.layers.append(nn.Linear(hidden_dim, output_dim2))
    
    def forward(self, X, batch):
        X = self.bn(X)
        for i in range(len(self.layers)-2):
            X = F.relu(self.layers[i](X))
        return global_mean_pool(self.layers[-2](X), batch), self.layers[-1](X)
   
class GIN_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3):
        super(GIN_decoder, self).__init__()
        self.layers = nn.ModuleList([GINConv(nn.Linear(input_dim, hidden_dim), train_eps=True)])
        for i in range(num_layers-1):
            self.layers.append(GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        self.high_mlp = nn.Linear(hidden_dim, 2)
        self.low_mlp = nn.Linear(hidden_dim, 1)
        self.alpha = nn.Parameter(torch.tensor(0.0))  

    def forward(self, high_emb, high_graph, low_emb, low_graph):
        for i in range(len(self.layers)):
            high_emb, low_emb = self.layers[i](high_emb, high_graph.edge_index), self.layers[i](low_emb, low_graph.edge_index)
            high_emb, low_emb = high_emb.relu(), low_emb.relu()
        high_emb = F.dropout(high_emb, p=0.5, training=self.training)
        low_emb = F.dropout(low_emb, p=0.5, training=self.training)
        high_emb = self.high_mlp(high_emb)
        low_emb = torch.abs(self.low_mlp(low_emb))
        alpha = torch.sigmoid(self.alpha)
        return high_emb, low_emb, alpha
 
class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GIN, self).__init__()
        self.layers = nn.ModuleList([GINConv(nn.Linear(input_dim, hidden_dim), train_eps=True)])
        for i in range(num_layers-2):
            self.layers.append(GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        self.batch_norm = nn.BatchNorm1d(input_dim)

    def forward(self, x, edge_index, batch):
        x = self.batch_norm(x)
        for i in range(len(self.layers)-1):
            x = self.layers[i](x, edge_index)
            x = x.relu()
        # x = global_mean_pool(x, batch)  
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        return x

class GraphTrans(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_heads, num_layers):
        super(GraphTrans, self).__init__()
        self.layers = nn.ModuleList([TransformerConv(input_dim, hidden_dim // num_heads, heads=num_heads)])
        for i in range(num_layers-2):
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(TransformerConv(hidden_dim, hidden_dim // num_heads, heads=num_heads))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x, edge_index, batch):
        x = self.layers[0](x, edge_index)
        for i in range(1, len(self.layers)-1, 2):
            x = self.layers[i](x)
            x = self.layers[i+1](x, edge_index)
            x = x.relu()
        
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.layers[-1](x)
        return x