import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TransformerConv, GPSConv, GINEConv, PNAConv, GINConv

from model.GWT_model import GraphWaveletTransform

class HierarchicalBlending(nn.Module):
    def __init__(self, 
                 d_model: int, 
                 n_heads: int = 1, 
                 p: float = 0.5):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.p = p
        
        self.attn_high = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.attn_low  = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def forward(self, high_emb, low_emb, batch):
        device = high_emb.device

        num_feats = high_emb.shape[1]
        perm = torch.randperm(num_feats, device=device)
        half = num_feats // 2
        anchor1, anchor2 = perm[:half], perm[half:]

        high_emb1 = high_emb.clone()
        high_emb1[:, anchor1] = 0
        high_emb2 = high_emb.clone()
        high_emb2[:, anchor2] = 0

        low_mask = torch.rand(low_emb.shape[0], device=device) < self.p
        low_emb1 = low_emb.clone()
        low_emb1[low_mask] = 0
        low_emb2 = low_emb.clone()
        low_emb2[~low_mask] = 0

        global_low_emb2 = global_mean_pool(low_emb2, batch)

        Q_high = high_emb1.unsqueeze(1)         # [B, 1, d_model]
        K_high = global_low_emb2.unsqueeze(1)   # [B, 1, d_model]
        V_high = global_low_emb2.unsqueeze(1)   # [B, 1, d_model]

        attn_high_out, _ = self.attn_high(Q_high, K_high, V_high)
        attn_high_out = attn_high_out.squeeze(1)

        Q_low = low_emb1.unsqueeze(1)         # [N, 1, d_model]
        K_low = high_emb2[batch].unsqueeze(1) # [N, 1, d_model]
        V_low = high_emb2[batch].unsqueeze(1) # [N, 1, d_model]

        attn_low_out, _ = self.attn_low(Q_low, K_low, V_low)
        attn_low_out = attn_low_out.squeeze(1)  # => [N, d_model]

        return attn_high_out, attn_low_out
    
def HeirarchicalBlendingWavelet(high_level_subgraph, low_level_graphs, high_emb, low_emb, J):# -> tuple:# -> tuple:
    num_genes = len(torch.where(low_level_graphs.batch==0)[0])
    max_genes = []
    for i in range(low_level_graphs.num_nodes//num_genes):
        max_genes.append(torch.argmax(low_level_graphs.X[i*num_genes:(i+1)*num_genes].view(-1))+high_level_subgraph.num_nodes + i*num_genes)
    new_edges = torch.stack([torch.arange(high_level_subgraph.num_nodes), torch.LongTensor(max_genes)])
    new_edges = torch.cat([new_edges, torch.stack([new_edges[1], new_edges[0]])], 1).to(high_level_subgraph.X.device)
    new_edge_index = torch.cat([high_level_subgraph.edge_index, new_edges, low_level_graphs.edge_index+high_level_subgraph.num_nodes],1)
    new_edge_index = torch.cat([new_edge_index, torch.arange(high_level_subgraph.num_nodes + low_level_graphs.num_nodes).repeat(2,1).to(new_edge_index.device)], 1)
    high_weights = (high_level_subgraph.X[high_level_subgraph.edge_index[0]] - high_level_subgraph.X[high_level_subgraph.edge_index[1]]).square().sum(1).pow(1/2)
    high_weights = high_weights/high_weights.sum()
    edge_weights = torch.cat([high_weights, torch.ones(2*high_level_subgraph.num_nodes, device = high_weights.device)*0.5, low_level_graphs.weight/low_level_graphs.weight.sum(), torch.ones(high_level_subgraph.num_nodes + low_level_graphs.num_nodes, device = high_weights.device)*0.5])
    emb = torch.cat([high_emb, low_emb], 0)
    gwt = GraphWaveletTransform(new_edge_index, edge_weights, emb, 4, high_weights.device)
    emb = gwt.generate_timepoint_features()
    return emb[:high_level_subgraph.num_nodes].float(), emb[high_level_subgraph.num_nodes:].float()

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
        # self.conv = GPSConv(input_dim, GINConv(nn.Linear(input_dim, output_dim), train_eps=True), heads=num_heads)
        self.norm = nn.LayerNorm(output_dim)
        self.cross_message_passing = cross_message_passing
        # self.attn_fclh = nn.Linear(output_dim * 2, 1, bias=False)
        # self.attn_fchl = nn.Linear(output_dim * 2, 1, bias=False)
        self.cross_lh = CrossMessagePassing(output_dim)
        self.cross_hl = CrossMessagePassing(output_dim)

    def forward(self, high_emb_in, high_level_graph, low_emb_in, low_level_graphs):
        high_emb = self.norm(high_emb_in)
        low_emb = self.norm(low_emb_in)

        high_emb_gin = self.conv_high(high_emb, high_level_graph.edge_index)
        high_emb_mh, _ = self.multi_head(high_emb, high_emb, high_emb)
        high_emb = high_emb_mh + high_emb_gin #+ high_emb_in
        
        low_emb = self.conv_low(low_emb, low_level_graphs.edge_index) #+ low_emb_in
        
        if(self.cross_message_passing):
            # _high_emb = high_emb#.clone().requires_grad_()
            x = global_mean_pool(low_emb, low_level_graphs.batch)
            high_emb_per_node = high_emb[low_level_graphs.batch]  # (N_low_nodes, output_dim)
            _high_emb = self.cross_hl(high_emb, x)
            updated_low_emb = self.cross_lh(low_emb, high_emb_per_node)
            # sim_score = torch.softmax(self.attn_fclh(torch.cat([_high_emb, x], dim=1)), dim=1)
            # _high_emb = sim_score[:,0].view(high_level_graph.num_nodes,1) * _high_emb + sim_score[:,1].view(high_level_graph.num_nodes,1)  * x
            # high_emb_per_node = high_emb[low_level_graphs.batch]  # (N_low_nodes, output_dim)
            # concat_hl = torch.cat([low_emb, high_emb_per_node], dim=1)  # (N_low_nodes, 2 * output_dim)
            # attn_scores_low = torch.softmax(self.attn_fchl(concat_hl), dim=1)  # (N_low_nodes, 1)
            # updated_low_emb = attn_scores_low[:,0].view(low_level_graphs.num_nodes,1) * low_emb + attn_scores_low[:,1].view(low_level_graphs.num_nodes,1) * high_emb_per_node  # (N_low_nodes, output_dim)
            
            # sim_score = torch.sigmoid(self.attn_fclh(torch.cat([_high_emb, x], dim=1)))
            # _high_emb = sim_score * _high_emb + (1 - sim_score) * x
            
            # # Gather high_emb[i] for each node in low_emb using the batch vector
            # high_emb_per_node = high_emb[low_level_graphs.batch]  # (N_low_nodes, output_dim)
            # # concat = torch.stack([low_emb, high_emb_per_node], dim=1)  # [N, 2, d_model]
            # # scores = self.attn_fchl(concat.view(concat.shape[0], -1))  # [N, 1]
            # # alpha = torch.softmax(torch.cat([scores, -scores], dim=1), dim=1)  # [N, 2]
            # # # alpha = torch.softmax(self.attn_fchl(concat.view(concat.shape[0], -1)), dim=1)  # [N, 2]
            # # updated_low_emb = alpha[:, 0:1] * low_emb + alpha[:, 1:2] * high_emb_per_node

            # # Concatenate high_emb and low_emb
            # concat_hl = torch.cat([low_emb, high_emb_per_node], dim=1)  # (N_low_nodes, 2 * output_dim)

            # # Compute attention scores in batch
            # attn_scores_low = torch.sigmoid(self.attn_fchl(concat_hl)+1)  # (N_low_nodes, 1)

            # # Weighted update
            # updated_low_emb = attn_scores_low * low_emb + (1 - attn_scores_low) * high_emb_per_node  # (N_low_nodes, output_dim)
            # import pdb; pdb.set_trace()
            return F.gelu(_high_emb), F.gelu(updated_low_emb)#, aux_loss_high+aux_loss_low
        else:
            return F.gelu(high_emb), F.gelu(low_emb)