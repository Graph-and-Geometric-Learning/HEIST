import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import PyTorchModelHubMixin
from torch_geometric.nn import TransformerConv, GINConv
from torch_geometric.nn.pool import global_mean_pool
from model.layers import MultiLevelGraphLayer, NicheAttention  # MLP re-exported for eval_*.py
from model.pe import calculate_sinusoidal_pe


class GraphEncoder(
    nn.Module,
    PyTorchModelHubMixin,
    repo_url="https://huggingface.co/HirenMadhu/HEIST",
    pipeline_tag="feature-extraction",
    license="mit",
):
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
        *,
        marker_embedding=False,
        num_markers=None,
        niche_attention=False,
        num_niches=16,
        niche_heads=4,
        niche_detach=False,
        niche_feat="low",
        rank_pe_fixed=False,
    ):
        super().__init__()
        self.pe_dim = pe_dim
        self.positional_encoding = positional_encoding
        self.cross_message_passing = cross_message_passing
        self.marker_embedding = marker_embedding
        self.num_markers = num_markers
        self.rank_pe_fixed = rank_pe_fixed

        # Input projections (pe_dim when using PE with addition, otherwise raw input dim)
        high_in_dim = pe_dim if positional_encoding else 2
        low_in_dim = pe_dim if positional_encoding else 1
        self.mlp_high = nn.Sequential(nn.Linear(high_in_dim, init_dim), nn.GELU())
        self.mlp_low = nn.Sequential(nn.Linear(low_in_dim, init_dim), nn.GELU())

        if marker_embedding:
            if num_markers is None:
                raise ValueError("marker_embedding=True requires num_markers")
            if not positional_encoding:
                raise ValueError("marker_embedding=True requires positional_encoding=True")
            self.marker_id_emb = nn.Embedding(num_markers, pe_dim)
            nn.init.normal_(self.marker_id_emb.weight, std=0.02)
            self.mask_token = nn.Parameter(torch.zeros(pe_dim))
            nn.init.normal_(self.mask_token, std=0.02)

        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.convs.append(MultiLevelGraphLayer(init_dim, hidden_dim, num_heads, cross_message_passing))
        for _ in range(num_layers - 2):
            self.convs.append(MultiLevelGraphLayer(hidden_dim, hidden_dim, num_heads, cross_message_passing))
        self.convs.append(MultiLevelGraphLayer(hidden_dim, output_dim, num_heads, cross_message_passing))

        self.final_norm = nn.LayerNorm(output_dim)
        self.projection_head = nn.Sequential(nn.Linear(output_dim, output_dim), nn.GELU())

        self.niche_attention = niche_attention
        self.num_niches = num_niches
        self.niche_feat = niche_feat
        if niche_feat == "markers" and num_markers is None:
            raise ValueError("niche_feat='markers' requires num_markers (the panel size)")
        _nf_dim = {"high": output_dim, "low": output_dim,
                   "concat": 2 * output_dim, "markers": num_markers}[niche_feat]
        self.niche = (
            NicheAttention(_nf_dim, num_niches=num_niches, num_heads=niche_heads,
                           detach=niche_detach, out_dim=output_dim)
            if niche_attention
            else None
        )

    @staticmethod
    def _marker_ids(low_level_graphs):
        batch = low_level_graphs.batch
        mid = getattr(low_level_graphs, "marker_id", None)
        if mid is not None:
            return mid.view(-1).long()
        counts = torch.bincount(batch)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)[:-1]])
        return (torch.arange(batch.numel(), device=batch.device) - offsets[batch]).long()

    def _prepare_inputs(self, high_level_graph, low_level_graphs):
        device = high_level_graph.X.device

        if self.positional_encoding:
            high_level_graph, low_level_graphs = calculate_sinusoidal_pe(
                high_level_graph, low_level_graphs, self.pe_dim,
                rank_pe_fixed=getattr(self, "rank_pe_fixed", False),
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
            if self.marker_embedding:
                mid = self._marker_ids(low_level_graphs)
                if int(mid.max()) >= self.num_markers:
                    raise ValueError(
                        f"marker index {int(mid.max())} >= num_markers={self.num_markers}. "
                        f"The batch has cells with {int(mid.max()) + 1} markers -- check that "
                        f"--data_dir points at ONE cohort (charville=40, dfci=41, upmc=22)."
                    )
                low_emb = low_emb + self.marker_id_emb(mid)   # = x + rank_pe + gene identity
                lm = getattr(low_level_graphs, "_low_mask", None)
                if lm is not None:                      # learned [MASK] token at masked positions
                    lm = lm.view(-1, 1).float()
                    low_emb = low_emb * lm + self.mask_token.unsqueeze(0) * (1.0 - lm)
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
  
    def _apply_niche(self, high_emb, low_emb, high_level_graph, low_level_graphs, pos):
        if self.niche is None:
            return high_emb, None
        if self.niche_feat == "markers":
            feat = self._raw_markers
        else:
            pooled = global_mean_pool(low_emb, low_level_graphs.batch)
            feat = {"high": high_emb,
                    "low": pooled,
                    "concat": torch.cat([high_emb, pooled], dim=1)}[self.niche_feat]
        return self.niche(feat, pos, high_level_graph.edge_index, h=high_emb)

    def _capture_markers(self, low_level_graphs):
        if self.niche_feat != "markers" or self.niche is None:
            return None
        n_mk = int(torch.bincount(low_level_graphs.batch).max())
        return low_level_graphs.X.view(-1, n_mk).float().detach().clone()

    def forward(self, high_level_graph, low_level_graphs, high_mask=None, low_mask=None,
                return_niche=False):
        pos = high_level_graph.X.float()
        self._raw_markers = self._capture_markers(low_level_graphs)
        if high_mask is not None and low_mask is not None:
            high_level_graph.X = high_level_graph.X * high_mask
            low_level_graphs.X = low_level_graphs.X * low_mask
            low_level_graphs._low_mask = low_mask

        high_emb, low_emb = self._prepare_inputs(high_level_graph, low_level_graphs)
        high_emb, low_emb = self.mlp_high(high_emb), self.mlp_low(low_emb)
        high_emb, low_emb = self._run_conv_layers(high_emb, high_level_graph, low_emb, low_level_graphs)
        high_emb, S = self._apply_niche(high_emb, low_emb, high_level_graph, low_level_graphs, pos)
        out = (self.projection_head(high_emb), self.projection_head(low_emb))
        return (*out, S) if return_niche else out

    def encode(self, high_level_graph, low_level_graphs, gene_mask=None, return_niche=False):
        pos = high_level_graph.X.float()
        self._raw_markers = self._capture_markers(low_level_graphs)
        if gene_mask is not None:
            low_level_graphs.X = low_level_graphs.X * gene_mask

        high_emb, low_emb = self._prepare_inputs(high_level_graph, low_level_graphs)
        high_emb, low_emb = self.mlp_high(high_emb), self.mlp_low(low_emb)

        high_emb, low_emb = self._run_conv_layers(high_emb, high_level_graph, low_emb, low_level_graphs)
        high_emb, S = self._apply_niche(high_emb, low_emb, high_level_graph, low_level_graphs, pos)
        return (high_emb, low_emb, S) if return_niche else (high_emb, low_emb)


class GIN_decoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers=3, nonneg_recon=True):
        """
        nonneg_recon: apply torch.abs to the low-level (per-marker) reconstruction.

        That is right for raw counts, but WRONG for standardized input: with per-core z-scored
        markers roughly half the targets are negative, so an abs() output can never reach them —
        the best possible prediction for any t<0 is 0, and |x| has a sign-flipping gradient that
        parks those units at ~0. On z-scored data set nonneg_recon=False.
        """
        super().__init__()
        self.nonneg_recon = nonneg_recon
        self.layers = nn.ModuleList([GINConv(nn.Linear(input_dim, hidden_dim), train_eps=True, aggr='mean')])
        for _ in range(num_layers - 1):
            self.layers.append(GINConv(nn.Linear(hidden_dim, hidden_dim), train_eps=True, aggr='mean'))
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
        low_emb = self.low_mlp(low_emb)
        if self.nonneg_recon:
            low_emb = torch.abs(low_emb)
        alpha = torch.sigmoid(self.alpha)
        return high_emb, low_emb, alpha
