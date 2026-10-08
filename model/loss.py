import math

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.pool import global_add_pool
from torch_geometric.utils import degree, remove_self_loops


def niche_cut_loss(S, edge_index, num_nodes=None):
    """
    MinCutPool relaxed normalised-cut term, -Tr(S^T A S) / Tr(S^T D S), in [-1, 0].

    This is what turns "the attention happens to look spatially coherent" (the Figure 4 claim) into
    a trained objective: neighbouring cells are pushed to share a niche.

    Computed sparsely from edge_index. Do NOT densify — a METIS partition is 384 cells here but the
    same code runs at inference with batch_size=4096 (one partition per region, so no METIS seams
    cut through the niche map), where a dense [C, C] adjacency is wasteful.

    Minimised (-1) by assigning every cell to ONE niche, so it MUST be paired with
    niche_balance_loss or it collapses.
    """
    if num_nodes is None:
        num_nodes = S.size(0)
    edge_index, _ = remove_self_loops(edge_index)
    src, dst = edge_index[0], edge_index[1]
    if src.numel() == 0:
        return S.sum() * 0.0

    num = (S[src] * S[dst]).sum()                                  # Tr(S^T A S)
    deg = degree(dst, num_nodes=num_nodes, dtype=S.dtype)           # [C]
    den = (deg.unsqueeze(-1) * S.pow(2)).sum()                      # Tr(S^T D S)
    return -num / (den + 1e-9)


def niche_balance_loss(S):
    """
    NEGATIVE entropy of the mean assignment, normalised to [-1, 0]. REQUIRED, not optional.

    There are TWO degenerate corners and no single term rules out both. Measured at K=16:

                     cut      ortho    balance
        uniform     -1.0000   1.2247   -1.0000
        collapsed   -1.0000   1.2247    0.0000
        good        -0.8600   0.0000   -1.0000

    `niche_cut_loss` is blind to both. `niche_ortho_loss` scores uniform and collapsed IDENTICALLY,
    so it rules out both relative to a good partition but is FLAT between them -- a head that falls
    into collapse gets no gradient out of it. This term is the only one that separates the two
    (collapsed 0.0 vs uniform -1.0), so it supplies the escape direction.

    Observed live: Charville epoch 1 with w_balance=0 sat at usage entropy 0.096 and sharpness 0.93,
    i.e. every cell confidently assigned to ONE niche, with ortho pinned at the 1.2240 plateau.
    Use ortho AND balance together.

    Sign still matters. `MoETransformerConv.auxiliary_loss`
    (model/layers.py:87-93) computes the same entropy but returns it POSITIVE and adds it to the
    loss, which MINIMISES entropy and therefore actively drives collapse onto a single expert. That
    module is dead code, so the sign was never exercised — do not copy it. Here we return -H so that
    minimising the loss MAXIMISES entropy, i.e. spreads mass across niches.
    """
    p = S.mean(0)
    entropy = -(p * (p + 1e-10).log()).sum()
    return -entropy / math.log(S.size(1))


def niche_ortho_loss(S):
    """
    MinCutPool orthogonality term, || S^T S / ||S^T S||_F  -  I_K / sqrt(K) ||_F.

    This is the term that makes the niche head learn anything at all, and it is NOT
    interchangeable with the entropy penalty above. `niche_cut_loss` sits at its -1.0 floor for ANY
    assignment that is constant across cells — collapsed onto one niche, and equally the fully
    UNIFORM S_ik = 1/K. Entropy of the mean assignment cannot see that either: it constrains only
    the column marginal, which uniform S maximises. So cut + entropy are BOTH perfectly satisfied by
    S_ik = 1/K, which carries zero information.

    Measured on the first smoke run with cut+entropy only: row entropy 0.9999, mean max-probability
    0.0648 against 1/16 = 0.0625. The head had learned exactly nothing.

    This term reads the GRAM matrix rather than the marginal, so it separates those cases:
      - uniform S           -> S^T S is rank-1, far from diagonal      -> ~1.22 at K=16 (penalised)
      - collapsed S         -> S^T S is rank-1                         -> penalised
      - balanced hard split -> S^T S = diag(N/K), i.e. exactly I/sqrt(K) -> 0.0 (optimal)
    Paired with the cut term it yields clusters that are simultaneously spatially coherent,
    balanced and near-one-hot: MinCutPool as published (Bianchi et al., 2020).
    """
    K = S.size(1)
    SS = S.t() @ S
    SS = SS / (SS.norm(p="fro") + 1e-9)
    target = torch.eye(K, device=S.device, dtype=S.dtype) / math.sqrt(K)
    return (SS - target).norm(p="fro")


def aucpr_hinge_loss(y_pred, y_true, margin=1.0):
    """
    AUCPR hinge loss function to optimize for area under the precision-recall curve.

    Args:
        y_pred (torch.Tensor): Predicted scores for each instance. Shape: (batch_size,)
        y_true (torch.Tensor): Ground truth binary labels for each instance. Shape: (batch_size,)
        margin (float): Margin for the hinge loss, default is 1.0.
    
    Returns:
        torch.Tensor: Calculated AUCPR hinge loss.
    """
    # Separate positive and negative samples
    pos_pred = y_pred[y_true == 1]
    neg_pred = y_pred[y_true == 0]
    
    # Pairwise difference: positive predictions should be higher than negative ones
    pairwise_diff = pos_pred.view(-1, 1) - neg_pred.view(1, -1)
    
    # Apply hinge loss with margin
    hinge_loss = F.relu(margin - pairwise_diff)
    
    # Average over all pairs
    loss = hinge_loss.mean()
    
    return loss


class AUCPRHingeLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(AUCPRHingeLoss, self).__init__()
        self.margin = margin

    def forward(self, y_pred, y_true):
        # Ensure labels are binary
        assert torch.all((y_true == 0) | (y_true == 1)), "y_true should be binary (0 or 1)."

        # Get positive and negative samples
        pos_mask = y_true == 1
        neg_mask = y_true == 0

        pos_pred = y_pred[pos_mask]
        neg_pred = y_pred[neg_mask]

        # If there are no positive or negative samples, return zero loss
        if pos_pred.numel() == 0 or neg_pred.numel() == 0:
            return torch.tensor(0.0, requires_grad=True).to(y_pred.device)

        # Calculate pairwise differences between positive and negative scores
        pairwise_diff = neg_pred.unsqueeze(0) - pos_pred.unsqueeze(1)  # Shape: (num_pos, num_neg)

        # Apply hinge loss with margin
        hinge_loss = torch.relu(self.margin + pairwise_diff)  # Hinge loss: max(0, margin + (neg - pos))

        # Average the loss
        loss = hinge_loss.mean()

        return loss

def cross_contrastive_loss(z_1, z_2, cell_type, N, temperature=0.5):
    # Normalize the embeddings
    z_1 = F.normalize(z_1, dim=1)
    z_2 = F.normalize(z_2, dim=1)

    batch_size = z_1.size(0)
    loss = 0.0
    epsilon = 1e-8  # Small constant for numerical stability

    for i in range(batch_size):
        # Positive pair (z_1[i], z_2[i])
        pos_sim = torch.mm(z_1[i].unsqueeze(0), z_2[i].unsqueeze(1)) / temperature

        # Select N negative samples that do not have the same cell_type
        mask = cell_type != cell_type[i]
        negative_indices = torch.nonzero(mask, as_tuple=False).squeeze()
        if(len(negative_indices.shape)):
            negative_indices = torch.nonzero(mask, as_tuple=False).squeeze()
            neg_indices = torch.randperm(len(negative_indices))[:N]  # Randomly select N negative samples
            neg_samples_1 = z_1[negative_indices[neg_indices]]
            neg_samples_2 = z_2[negative_indices[neg_indices]]

                # Compute similarity with negative samples
            neg_sim_1 = torch.mm(z_1[i].unsqueeze(0), neg_samples_1.t()) / temperature
            neg_sim_2 = torch.mm(z_2[i].unsqueeze(0), neg_samples_2.t()) / temperature

                # Compute denominator with positive and negative similarities
            denominator_1 = torch.cat([pos_sim, neg_sim_1], dim=1).sum() + epsilon
            denominator_2 = torch.cat([pos_sim, neg_sim_2], dim=1).sum() + epsilon

                # Compute loss for the positive pair
            loss_1 = -torch.log(pos_sim / denominator_1 + epsilon).mean()
            loss_2 = -torch.log(pos_sim / denominator_2 + epsilon).mean()

                # Accumulate loss
            loss += (loss_1 + loss_2) / 2
    # Final loss is averaged over the batch
    loss /= batch_size
    return loss

def contrastive_loss_cell(cell_types, high_emb, low_level_batch, low_emb, N):
    high_emb = F.normalize(high_emb, p=2, dim=-1)
    low_emb = F.normalize(low_emb, p=2, dim=-1)

    num_cells = cell_types.shape[0]
    device = high_emb.device

    arange = torch.arange(num_cells, device=device)
    positive_indices = []
    negative_indices = []

    for i in range(num_cells):
        current_type = cell_types[i]

        pos_idx = torch.where((cell_types == current_type) & (arange != i))[0]
        if len(pos_idx) > 0:
            positive_indices.append(pos_idx[torch.randint(0, len(pos_idx), (1,))].item())
        else:
            positive_indices.append(i)

        neg_idx = torch.where(cell_types != current_type)[0]
        if len(neg_idx) > 0:
            negative_indices.append(neg_idx[torch.randint(0, len(neg_idx), (N,))].tolist())
        else:
            # No negatives for this cell type — fall back to all other cells
            other_idx = torch.where(arange != i)[0]
            negative_indices.append(other_idx[torch.randint(0, len(other_idx), (N,))].tolist())

    positive_indices = torch.LongTensor(positive_indices).to(device)
    negative_indices = torch.LongTensor(negative_indices).to(device)

    positive_similarities_high = F.cosine_similarity(high_emb, high_emb[positive_indices]).clamp(min=1e-6)
    negative_similarities_high = F.cosine_similarity(high_emb.unsqueeze(1), high_emb[negative_indices], dim=-1)
    # high_level_loss = -torch.mean(
    #     torch.log(positive_similarities_high + 1e-8) - torch.logsumexp(negative_similarities_high, dim=-1)
    # )
    log_pos = torch.log(positive_similarities_high + 1e-8)
    lse_neg = torch.logsumexp(negative_similarities_high, dim=-1).clamp(max=30)  # ~exp(30)=1e13
    high_level_loss = -torch.mean(log_pos - lse_neg)


    pooled_low_level = F.normalize(global_add_pool(low_emb, low_level_batch.batch))
    positive_similarities_cross = F.cosine_similarity(pooled_low_level, high_emb[positive_indices]).clamp(min=1e-6)
    negative_similarities_cross = F.cosine_similarity(
        pooled_low_level.unsqueeze(1), high_emb[negative_indices], dim=-1
    )
    los_pos_cross = torch.log(positive_similarities_cross + 1e-8)
    lse_neg_cross = torch.logsumexp(negative_similarities_cross, dim=-1).clamp(max=30)
    cross_level_loss = -torch.mean(los_pos_cross - lse_neg_cross)
    # cross_level_loss = -torch.mean(
    #     torch.log(positive_similarities_cross + 1e-8) - torch.logsumexp(negative_similarities_cross, dim=-1)
    # )

    positive_similarities_low = F.cosine_similarity(pooled_low_level, pooled_low_level[positive_indices]).clamp(min=1e-6)
    negative_similarities_low = F.cosine_similarity(
        pooled_low_level.unsqueeze(1), pooled_low_level[negative_indices], dim=-1
    )
    log_pos_low = torch.log(positive_similarities_low + 1e-8)
    lse_neg_low = torch.logsumexp(negative_similarities_low, dim=-1).clamp(max=30)  # ~exp(30)=1e13
    low_level_loss = -torch.mean(log_pos_low - lse_neg_low)

    # low_level_loss = -torch.mean(
    #     torch.log(positive_similarities_low + 1e-8) - torch.logsumexp(negative_similarities_low, dim=-1)
    # )

    # Each component above is ALREADY a -torch.mean(...) over cells; dividing by num_cells again made
    # this term ~1/num_cells too small (~3e-3 at 384 cells vs a reconstruction loss of order 1). With
    # the learned sigmoid(alpha) blend — whose gradient drives alpha toward whichever term is SMALLER
    # — that silently annihilated the reconstruction objective within ~1 epoch. Do not reintroduce it.
    return high_level_loss + cross_level_loss + low_level_loss

def mae_loss_cell(high_emb, low_emb, decoded_high, decoded_low, high_mask, low_mask):
    if(high_mask.sum()):
        high_recon_loss = (1-F.cosine_similarity(high_emb*high_mask,decoded_high*high_mask)).sum()/high_mask.sum()
    else:
        high_recon_loss = 0
    low_recon_loss = (1-F.cosine_similarity(low_emb*low_mask,decoded_low*low_mask)).sum()/low_mask.sum()
    return (high_recon_loss + low_recon_loss)/2

def contrastive_loss_cell_single_view(cell_types, high_emb, low_emb, N):
    num_cells = cell_types.shape[0]
    device = high_emb.device

    arange = torch.arange(num_cells, device=device)
    positive_indices = []
    negative_indices = []

    for i in range(num_cells):
        current_type = cell_types[i]

        pos_idx = torch.where((cell_types == current_type) & (arange != i))[0]
        if len(pos_idx) > 0:
            positive_indices.append(pos_idx[torch.randint(0, len(pos_idx), (1,))].item())
        else:
            positive_indices.append(i)

        neg_idx = torch.where(cell_types != current_type)[0]
        if len(neg_idx) > 0:
            negative_indices.append(neg_idx[torch.randint(0, len(neg_idx), (N,))].tolist())
        else:
            other_idx = torch.where(arange != i)[0]
            negative_indices.append(other_idx[torch.randint(0, len(other_idx), (N,))].tolist())

    positive_indices = torch.LongTensor(positive_indices).to(device)
    negative_indices = torch.LongTensor(negative_indices).to(device)

    positive_similarities_high = F.cosine_similarity(high_emb, high_emb[positive_indices]).clamp(min=1e-6)
    negative_similarities_high = F.cosine_similarity(high_emb.unsqueeze(1), high_emb[negative_indices], dim=-1)
    high_level_loss = -torch.mean(
        torch.log(positive_similarities_high + 1e-8) - torch.logsumexp(negative_similarities_high, dim=-1)
    )

    pooled_low_level = F.normalize(low_emb)
    positive_similarities_cross = F.cosine_similarity(pooled_low_level, high_emb[positive_indices]).clamp(min=1e-6)
    negative_similarities_cross = F.cosine_similarity(
        pooled_low_level.unsqueeze(1), high_emb[negative_indices], dim=-1
    )
    cross_level_loss = -torch.mean(
        torch.log(positive_similarities_cross + 1e-8) - torch.logsumexp(negative_similarities_cross, dim=-1)
    )

    positive_similarities_low = F.cosine_similarity(pooled_low_level, pooled_low_level[positive_indices]).clamp(min=1e-6)
    negative_similarities_low = F.cosine_similarity(
        pooled_low_level.unsqueeze(1), pooled_low_level[negative_indices], dim=-1
    )
    low_level_loss = -torch.mean(
        torch.log(positive_similarities_low + 1e-8) - torch.logsumexp(negative_similarities_low, dim=-1)
    )

    # Each component above is ALREADY a -torch.mean(...) over cells; dividing by num_cells again made
    # this term ~1/num_cells too small (~3e-3 at 384 cells vs a reconstruction loss of order 1). With
    # the learned sigmoid(alpha) blend — whose gradient drives alpha toward whichever term is SMALLER
    # — that silently annihilated the reconstruction objective within ~1 epoch. Do not reintroduce it.
    return high_level_loss + cross_level_loss + low_level_loss
