import torch
import torch.nn as nn
import torch.nn.functional as F

from huggingface_hub import PyTorchModelHubMixin
from torch_geometric.nn import TransformerConv, GINConv
from torch_geometric.nn.pool import global_mean_pool
from model.layers import MultiLevelGraphLayer
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
    ):
        """
        marker_embedding / num_markers are KEYWORD-ONLY on purpose. Several stale call sites
        (main.py:128, calculate_rep.py:73, eval_gene_imputation_fine_tune.py:83) already pass 10
        POSITIONAL args to this 8-parameter signature, which raises a loud TypeError today. If these
        were positional they would instead silently bind args.anchor_pe -> marker_embedding and
        args.blending -> num_markers. Keyword-only params are still captured by
        PyTorchModelHubMixin.__new__, so config serialisation is unaffected, and omitting them from a
        published config.json means from_pretrained() falls back to these defaults (no new module,
        so the existing HirenMadhu/HEIST checkpoint still loads unchanged).
        """
        super().__init__()
        self.pe_dim = pe_dim
        self.positional_encoding = positional_encoding
        self.cross_message_passing = cross_message_passing
        self.marker_embedding = marker_embedding
        self.num_markers = num_markers

        # Input projections (pe_dim when using PE with addition, otherwise raw input dim)
        high_in_dim = pe_dim if positional_encoding else 2
        low_in_dim = pe_dim if positional_encoding else 1
        self.mlp_high = nn.Sequential(nn.Linear(high_in_dim, init_dim), nn.GELU())
        self.mlp_low = nn.Sequential(nn.Linear(low_in_dim, init_dim), nn.GELU())

        if marker_embedding:
            if num_markers is None:
                raise ValueError("marker_embedding=True requires num_markers")
            if not positional_encoding:
                # low_in_dim would be 1, but the embedding is pe_dim wide.
                raise ValueError("marker_embedding=True requires positional_encoding=True")
            # Factorised marker token (scGPT/Geneformer style): an IDENTITY vector plus a per-marker
            # VALUE direction. The previous `X + pe` broadcast one scalar across all pe_dim channels,
            # so through mlp_low the expression value occupied a single fixed direction SHARED by
            # every marker — the model had no per-marker way to represent magnitude.
            self.marker_id_emb = nn.Embedding(num_markers, pe_dim)
            self.marker_val_emb = nn.Embedding(num_markers, pe_dim)
            nn.init.normal_(self.marker_id_emb.weight, std=0.02)
            nn.init.normal_(self.marker_val_emb.weight, std=0.02)
            # Learned [MASK] token: multiplicative masking sets a value to 0.0, which for z-scored
            # input is the MODE, so the model cannot tell a masked entry from an average one.
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

    @staticmethod
    def _marker_ids(low_level_graphs):
        """
        Within-cell node position == marker index, derived from the batch vector.

        Node order is guaranteed marker-aligned: GRNs are built with add_nodes_from(range(n_markers))
        and X is reshaped in column order, and shuffle_node_indices only ever permutes the HIGH-level
        graph. Derived rather than stored so no cached .pt graph needs regenerating.
        Uses bincount, NOT `ptr` — the DataLoader(batch_size=1) re-collation in utils/dataloader.py
        drops `ptr`.
        """
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
            if self.marker_embedding:
                mid = self._marker_ids(low_level_graphs)
                x = low_level_graphs.X.float().view(-1, 1)
                low_emb = low_emb + self.marker_id_emb(mid) + self.marker_val_emb(mid) * x
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
  
    def forward(self, high_level_graph, low_level_graphs, high_mask=None, low_mask=None):
        if high_mask is not None and low_mask is not None:
            high_level_graph.X = high_level_graph.X * high_mask
            low_level_graphs.X = low_level_graphs.X * low_mask
            # Stash the mask so _prepare_inputs can substitute a learned [MASK] token. This MUST be
            # applied inside forward(): touching self.mask_token / marker embeddings from the training
            # loop would put them outside the autograd graph DDP traverses from forward()'s output,
            # so with find_unused_parameters=True their gradients would never be synchronised and the
            # tables would silently diverge across ranks.
            low_level_graphs._low_mask = low_mask

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
        low_emb = self.low_mlp(low_emb)
        if self.nonneg_recon:
            low_emb = torch.abs(low_emb)
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
