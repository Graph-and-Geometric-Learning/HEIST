import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TransformerConv, GPSConv, GINEConv, PNAConv, GINConv

from model.GWT_model import GraphWaveletTransform


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
        self.sf = nn.Softmax(dim=1)
        if(num_layers==1):
            self.layers = nn.ModuleList([nn.Linear(input_dim, output_dim)])
        else:
            self.layers = nn.ModuleList([nn.Linear(input_dim, hidden_dim)])
            for i in range(num_layers-2):
                self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.Linear(hidden_dim, output_dim))
    
    def forward(self, X):
        for i in range(len(self.layers)-1):
            X = F.relu(self.layers[i](X))
        return self.layers[-1](X)
        
class GatingNetwork(nn.Module):
    def __init__(self, input_dim, num_experts):
        super(GatingNetwork, self).__init__()
        self.gate = nn.Linear(input_dim, num_experts)

    def forward(self, x):
        return F.softmax(self.gate(x), dim=-1)

# Define the Mixture of Experts Layer class
class MoETransformerConv(MessagePassing):
    def __init__(self, in_channels: int,
        out_channels: int,
        heads: int = 1,
        num_experts: int = 4,  # Number of experts
        top_k: int = 2,  # Activate only top-k experts per input
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.0,
        edge_dim: int = None,
        bias: bool = False,
        root_weight: bool = True,
        **kwargs
    ):
        super(MoETransformerConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)  # Ensure top_k ≤ num_experts
        self.beta = beta and root_weight
        self.root_weight = root_weight
        self.concat = concat
        self.dropout = dropout

        self.experts = nn.ModuleList([
            TransformerConv(in_channels, out_channels, heads, concat, beta, dropout, edge_dim, bias, root_weight)
            for _ in range(num_experts)
        ])
        self.gate = GatingNetwork(in_channels, num_experts)

    def forward(self, x: Tensor, edge_index: Tensor, edge_attr: Tensor = None):
        gating_scores = self.gate(x)
        topk_gating_scores, topk_indices = gating_scores.topk(self.top_k, dim=-1, sorted=False)
        # Create a mask to zero out the contributions of non-topk experts
        mask = torch.zeros_like(gating_scores).scatter_(-1, topk_indices, 1)
        # Use the mask to retain only the topk gating scores
        gating_scores = gating_scores * mask
        # Normalize the gating scores to sum to 1 across the selected top experts
        gating_scores = F.normalize(gating_scores, p=1, dim=-1)
        
        expert_outputs = torch.stack([expert(x, edge_index) for expert in self.experts], dim=1)
        # expert_outputs = expert_outputs.transpose(1, 2)
        output = torch.einsum('ne,neo->no', gating_scores, expert_outputs)  # Shape: [num_nodes, out_channels]
        return output, self.auxiliary_loss(gating_scores)
    
    def auxiliary_loss(self, gating_scores: Tensor, lambda_aux: float = 0.01):
        expert_prob = gating_scores.mean(dim=0)  # Shape: [num_experts]

        # Compute entropy-like auxiliary loss to encourage balanced selection
        aux_loss = -torch.sum(expert_prob * torch.log(expert_prob + 1e-10))  # Prevent log(0)

        return lambda_aux * aux_loss


class GINE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(GINE, self).__init__()
        self.layer = GINEConv(nn.Linear(input_dim, output_dim), train_eps=True)

    def forward(self, x, edge_index):
        x = self.layer(x, edge_index)
        x = x.relu()
        return x
    
class CrossMessagePassing(nn.Module):
    def __init__(self, d):
        super(CrossMessagePassing, self).__init__()
        self.Q = nn.Linear(d, d, bias = False)
        self.K = nn.Linear(d, d, bias = False)
        self.V = nn.Linear(d, d, bias = False)
        self.d = math.sqrt(d)
        
    def forward(self, to_emb, from_emb):
        Q = self.Q(to_emb)
        K = self.K(from_emb)
        V = self.V(to_emb)
        weights = (Q*K).sum(1)/self.d
        return weights.view(to_emb.shape[0],1)*V
    
class PNA(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(PNA, self).__init__()
        self.layer = PNAConv(input_dim, output_dim,  ['mean', 'max', 'min', 'std'], ["linear"], None)

    def forward(self, x, edge_index):
        x = self.layers(x, edge_index)
        x = x.relu()
        return x

class MultiLevelGraphLayer(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads, cross_message_passing):
        super(MultiLevelGraphLayer, self).__init__()
        self.conv_high = GINConv(nn.Linear(input_dim, output_dim), train_eps=True)
        self.multi_head = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.conv_low = TransformerConv(input_dim, output_dim // num_heads, heads=num_heads)

        self.norm_high_pre = nn.LayerNorm(output_dim)
        self.norm_high_post = nn.LayerNorm(output_dim)
        self.norm_low_pre = nn.LayerNorm(output_dim)
        self.norm_low_post = nn.LayerNorm(output_dim)

        self.MLP_high = MLP(output_dim, output_dim*4, output_dim, 3)
        self.MLP_low = MLP(output_dim, output_dim*4, output_dim, 3)

        self.cross_message_passing = cross_message_passing
        self.cross_lh = CrossMessagePassing(output_dim)
        self.cross_hl = CrossMessagePassing(output_dim)

    def forward(self, high_emb_in, high_level_graph, low_emb_in, low_level_graphs):
        high_emb_gin = self.conv_high(high_emb_in, high_level_graph.edge_index)
        high_emb_mh, _ = self.multi_head(high_emb_in, high_emb_in, high_emb_in)
        pre_high_emb = high_emb_mh + high_emb_gin
        high_emb = self.norm_high_pre(pre_high_emb)
        high_emb = self.MLP_high(high_emb)
        high_emb += pre_high_emb
        high_emb = self.norm_high_post(high_emb)

        pre_low_emb = self.conv_low(low_emb_in, low_level_graphs.edge_index)
        low_emb = self.norm_low_pre(pre_low_emb)
        low_emb = self.MLP_low(low_emb)
        low_emb += pre_low_emb
        low_emb = self.norm_low_post(low_emb)

        if(self.cross_message_passing):
            x = global_mean_pool(low_emb, low_level_graphs.batch)
            high_emb_per_node = high_emb[low_level_graphs.batch]  # (N_low_nodes, output_dim)
            _high_emb = self.cross_hl(high_emb, x)
            updated_low_emb = self.cross_lh(low_emb, high_emb_per_node)
            return F.gelu(_high_emb), F.gelu(updated_low_emb)
        else:
            return F.gelu(high_emb), F.gelu(low_emb)