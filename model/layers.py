import math

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.pool import global_mean_pool
from torch_geometric.nn import TransformerConv, GINConv
from torch_geometric.utils import remove_self_loops, scatter, softmax as pyg_softmax

from model.GWT_model import GraphWaveletTransform


class MLP(nn.Module):
    """The one MLP used across HEIST: num_layers Linear layers with ReLU between them (none after the
    last). num_layers=1 is a single Linear. Parameter names (layers.<i>.weight/bias) are what every
    saved checkpoint uses, so do not rename them."""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(MLP, self).__init__()
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
    
class NicheAttention(nn.Module):
    """
    Cross-attention from cells to a GLOBAL, learned niche vocabulary.

    `prototypes` are module parameters, so niche k denotes the same thing in every METIS partition,
    every tissue and every patient. That is the property per-tissue clustering cannot give, and it is
    what turns a niche picture into a per-patient composition vector comparable across a cohort.

    The query is built ONLY from the self-excluded spatial neighbourhood, never from the cell's own
    embedding. This is deliberate. HEIST's contrastive objective is supervised by `cell_type`
    (model/loss.py:108), so cell embeddings are strongly cell-type organised; a softmax head fed
    `h_i` would simply recover cell types and relabel them "niches". Feeding only the neighbourhood
    means two ADJACENT cells of DIFFERENT types land in the same niche, which is what "niche" has to
    mean. Same self-exclusion convention as blca_heist/05_niche.py:62-105 (`idx[:, kmin:]`).

    Neighbourhood pooling is multi-scale: the d channels are split across `num_heads` blocks, each
    smoothed with its own LEARNED bandwidth sigma over physical distance, so one head can look at a
    tight 20um ring while another sees a broad field.
    """

    def __init__(self, dim, num_niches=16, num_heads=4, detach=False, out_dim=None):
        """
        `dim` is the width of the feature the head reads, which MUST carry molecular content, not
        just the spatial half. `out_dim` is the width of the tensor written back (the cell embedding).

        Measured consequence of getting this wrong: fed only `high_emb`, niche labels were
        predictable from raw (x,y) at 0.925 balanced accuracy while neighbourhood marker content
        reached only 0.523 -- i.e. the head had discovered a spatial TILING of the core, not a
        microenvironment. HEIST's high-level node feature is literally the (x,y) coordinate plus a
        sinusoidal PE (model/model.py:115-119), with expression entering only indirectly through
        cross-level message passing, so a position-only tiling is the path of least resistance.
        Feeding [high_emb || mean-pooled gene embedding] puts real molecular content in the query.
        """
        super().__init__()
        if dim % num_heads:
            raise ValueError(f"dim {dim} must be divisible by num_heads {num_heads}")
        self.dim = dim
        self.out_dim = out_dim or dim
        self.num_niches = num_niches
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.detach = detach

        self.prototypes = nn.Parameter(torch.empty(num_niches, dim))
        nn.init.normal_(self.prototypes, std=0.02)

        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, self.out_dim, bias=False)
        self.O = nn.Linear(self.out_dim, self.out_dim, bias=False)
        self.norm = nn.LayerNorm(self.out_dim)
        # Whitens the pooled neighbourhood before the query projection. Averaging ~8 neighbours of a
        # barely-trained backbone leaves q_i nearly identical for every cell, which is precisely the
        # uniform-S basin; normalising restores the between-cell variance the assignment needs.
        self.q_norm = nn.LayerNorm(dim)

        # One bandwidth per head, parameterised in log space so it stays positive. Coordinates are
        # StandardScaler'd (utils/preprocess.py), and measured median Delaunay edge length is ~0.07,
        # so log_sigma=0 (sigma=1) starts broad and anneals down.
        self.log_sigma = nn.Parameter(torch.zeros(num_heads))

        # Cosine attention with a learned temperature, NOT raw dot-product / sqrt(dim).
        # With dim=128 and freshly initialised projections, q.k/sqrt(128) lands within a hair of 0
        # for every prototype, so softmax starts essentially uniform -- and uniform S is a genuine
        # optimum of the cut term, so there is no gradient pressure to leave it. Measured with the
        # dot-product form: after 3 epochs sharpness had crawled from 0.0625 (=1/K, i.e. nothing) to
        # only 0.169 and the ortho term had moved 1.2223 -> 1.2136 against a uniform value of ~1.22.
        # Normalising q and P onto the sphere bounds the logits to [-1/tau, 1/tau] and makes the
        # initial scale a knob rather than an accident.
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.1)))

    def neighborhood_pool(self, h, pos, edge_index, num_nodes):
        """Multi-scale, distance-weighted, self-excluded mean over spatial neighbours."""
        edge_index, _ = remove_self_loops(edge_index)
        src, dst = edge_index[0], edge_index[1]

        # Physical distance is recomputed from `pos` rather than read off graph.distance: the METIS
        # partitioner rebuilds each block as a fresh Data(X, edge_index, y, num_nodes) and DROPS the
        # distance attribute (utils/dataloader.py:96-104). pos IS the (standardized) coordinate, so
        # this is exact, not an approximation.
        d2 = (pos[src] - pos[dst]).pow(2).sum(-1, keepdim=True)          # [E, 1]
        sigma2 = self.log_sigma.exp().pow(2).clamp(min=1e-4)             # [H]
        logits = -d2 / sigma2.unsqueeze(0)                               # [E, H]
        alpha = pyg_softmax(logits, dst, num_nodes=num_nodes)            # [E, H]

        msg = h[src].view(-1, self.num_heads, self.head_dim) * alpha.unsqueeze(-1)
        out = scatter(msg, dst, dim=0, dim_size=num_nodes, reduce="sum")
        return out.reshape(num_nodes, self.dim)

    def forward(self, feat, pos, edge_index, h=None):
        """
        feat : [C, dim]      what the niche is computed FROM -- must include molecular content
        pos  : [C, 2]        coordinates (captured BEFORE any masking of graph.X)
        h    : [C, out_dim]  the tensor written back to; defaults to feat when they are the same
        Returns (h_updated [C, out_dim], S [C, K]).
        """
        if h is None:
            h = feat
        num_nodes = feat.size(0)
        # detach shields the pretrained backbone from the niche losses, so the head is strictly
        # additive and cannot regress the published benchmarks.
        f_in = feat.detach() if self.detach else feat

        h_nb = self.neighborhood_pool(f_in, pos, edge_index, num_nodes)   # [C, dim]

        q = F.normalize(self.Q(self.q_norm(h_nb)), dim=-1)                # [C, d]
        k = F.normalize(self.K(self.prototypes), dim=-1)                  # [K, d]
        tau = self.log_tau.exp().clamp(min=1e-2, max=10.0)
        S = torch.softmax(q @ k.t() / tau, dim=-1)                        # [C, K]

        if self.detach:
            # Pure read-out mode. The write-back is skipped ON PURPOSE, not merely to protect the
            # backbone: routing niche context into high_emb puts the head on the reconstruction
            # path, and the measured recon loss is ~1e8 against order-1 niche terms, so recon
            # gradient would dominate the head's parameters completely and train it toward an
            # identity map. Detached, the head is optimised by cut + ortho alone.
            # V/O receive no gradient here; DDP already runs with find_unused_parameters=True.
            return h, S

        denom = S.sum(0).unsqueeze(-1).clamp(min=1e-6)                    # [K, 1]
        niche_emb = (S.t() @ f_in) / denom                                # [K, dim]
        ctx = S @ self.V(niche_emb)                                       # [C, out_dim]
        return self.norm(h + self.O(ctx)), S

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