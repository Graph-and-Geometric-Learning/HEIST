import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.pool import global_add_pool

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

def infoNCE_loss(embedding_1, embedding_2, temperature=2):
    # Normalize the embeddings to help stabilize similarity calculations
    embedding_1 = F.normalize(embedding_1, dim=1)
    embedding_2 = F.normalize(embedding_2, dim=1)

    epsilon = 1e-8  # Small constant for numerical stability

    # Compute similarity matrices
    refl_sim1 = torch.mm(embedding_1, embedding_1.t()) / temperature
    refl_sim2 = torch.mm(embedding_2, embedding_2.t()) / temperature
    between_sim = torch.mm(embedding_1, embedding_2.t()) / temperature

    # Compute the denominators with epsilon for numerical stability
    denominator1 = refl_sim1.sum(1) + between_sim.sum(1) - refl_sim1.diag() + epsilon
    denominator2 = refl_sim2.sum(1) + between_sim.sum(1) - refl_sim2.diag() + epsilon

    # Ensure that the diagonal of between_sim is safe for log computation
    safe_between_sim_diag = torch.clamp(between_sim.diag(), min=epsilon)

    # Calculate the losses
    loss1 = -torch.log(safe_between_sim_diag / denominator1).mean()
    loss2 = -torch.log(safe_between_sim_diag / denominator2).mean()
    
    # Final loss is the average of both directions
    loss = (loss1 + loss2) / 2
    return loss

def cross_contrastive_loss(z_1, z_2, cell_type, N, temperature=2):
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


def cca_loss(z1, z2, lambd=0.5):
        c = torch.mm(z1.T, z2)
        c1 = torch.mm(z1.T, z1)
        c2 = torch.mm(z2.T, z2)
        N = z1.shape[0]
        c = c / N
        c1 = c1 / N
        c2 = c2 / N

        loss_inv = -torch.diagonal(c).sum()
        iden = torch.tensor(np.eye(c.shape[0])).to(c.device)
        loss_dec1 = (iden - c1).pow(2).sum()
        loss_dec2 = (iden - c2).pow(2).sum()

        loss = loss_inv + lambd * (loss_dec1 + loss_dec2)
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
            negative_indices.append([])

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

    return (high_level_loss + cross_level_loss + low_level_loss) / num_cells

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
            negative_indices.append([])

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

    return (high_level_loss + cross_level_loss + low_level_loss) / num_cells
