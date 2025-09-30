import torch
import numpy as np
import networkx as nx

@torch.no_grad()
def _corr_prefilter_gpu(X, topk=200, min_abs_r=None):
    """
    X: (cells, genes) float32 CUDA tensor (standardized)
    Returns candidate pair indices (i_idx, j_idx) after correlation prefilter.
    Use either topk per gene or an absolute threshold.
    """
    # corrcoef wants (features, samples)
    G = X.shape[1]
    C = torch.corrcoef(X.T)  # (genes, genes), symmetric, diag=1
    C.fill_diagonal_(0.0)

    if min_abs_r is not None:
        mask = C.abs() >= min_abs_r
        i_idx, j_idx = mask.nonzero(as_tuple=True)
        # Keep only upper triangle to avoid duplicates
        keep = i_idx < j_idx
        return i_idx[keep], j_idx[keep]
    else:
        # topk per gene (symmetric dedup later)
        k = min(topk, G-1)
        vals, idxs = torch.topk(C.abs(), k=k, dim=1, largest=True, sorted=False)  # per row (gene)
        # Build candidate set (i, j) and dedup by i<j
        i_idx = torch.arange(G, device=X.device).repeat_interleave(k)
        j_idx = idxs.reshape(-1)
        # Remove duplicates by always ordering (min, max) and unique
        a = torch.minimum(i_idx, j_idx)
        b = torch.maximum(i_idx, j_idx)
        pairs = torch.stack([a, b], dim=1)
        pairs = torch.unique(pairs, dim=0)
        return pairs[:,0], pairs[:,1]

@torch.no_grad()
def _mutual_information_binned_gpu(x, y, bins=32, eps=1e-12):
    """
    x, y: (N, B) where B is number of pairs in the batch (each column a variable)
    Returns MI for each column pair (B,)
    Implementation detail:
    - Compute bin indices (N,B) in [0, bins-1]
    - Joint histogram via scatter_add into (B, bins*bins)
    - Convert to probabilities and compute MI
    """
    N, B = x.shape
    # Bin edges per column: use global min/max per column
    # Normalize to [0,1] first, then bucketize
    def _normalize(z):
        zmin = z.min(dim=0, keepdim=True).values
        zmax = z.max(dim=0, keepdim=True).values
        zrange = (zmax - zmin).clamp_min(1e-6)
        return (z - zmin) / zrange

    xN = _normalize(x)
    yN = _normalize(y)

    # Bin indices in [0, bins-1]
    # Avoid edge case where value == 1.0 -> clamp to bins-1
    xi = torch.clamp((xN * bins).long(), 0, bins-1)  # (N,B)
    yi = torch.clamp((yN * bins).long(), 0, bins-1)  # (N,B)

    # Flatten per-column joint index: ji = xi*bins + yi  in [0, bins*bins-1]
    ji = xi * bins + yi  # (N,B)

    # Build joint hist via scatter_add per column
    nbins2 = bins * bins
    joint = torch.zeros(B, nbins2, device=x.device, dtype=torch.float32)  # (B, bins*bins)
    # For scatter, make row indices for columns and flatten N*B
    col_idx = torch.arange(B, device=x.device).unsqueeze(0).expand(N, B)  # (N,B)
    joint.scatter_add_(1, ji.T, torch.ones_like(ji, dtype=torch.float32).T)

    joint = joint / float(N) + eps  # probabilities
    px = joint.view(B, bins, bins).sum(dim=2) + 0.0  # (B, bins)
    py = joint.view(B, bins, bins).sum(dim=1) + 0.0  # (B, bins)

    # Entropies
    Hx = -(px * (px + eps).log()).sum(dim=1)
    Hy = -(py * (py + eps).log()).sum(dim=1)
    Hxy = -(joint * (joint).log()).sum(dim=1)

    MI = Hx + Hy - Hxy
    return MI

def build_gene_network_gpu(cell_data,
                           topk_per_gene=200,
                           min_abs_corr=None,
                           mi_bins=32,
                           mi_batch_size=20000,
                           std_coeff = 1,
                           device="cuda"):
    """
    cell_data: AnnData-like with .X sparse (cells x genes), .var.index as gene names
    Returns: edges (np.ndarray Nx2), weights (np.ndarray N,), gene_names (list)
    """
    # 1) Move to GPU
    import scipy.sparse as sp
    X = cell_data.X
    if sp.issparse(X):
        X = X.tocoo()
        X = torch.sparse_coo_tensor(
            torch.stack([torch.tensor(X.row), torch.tensor(X.col)]),
            torch.tensor(X.data, dtype=torch.float32),
            size=X.shape,
        ).to(device).to_dense()
    else:
        X = torch.tensor(X, dtype=torch.float32, device=device)


    # 2) Correlation pre-filter to get candidate pairs
    i_idx, j_idx = _corr_prefilter_gpu(X, topk=topk_per_gene, min_abs_r=min_abs_corr)

    # 3) Compute MI for candidate pairs in batches
    Ncells = X.shape[0]
    pairs = torch.stack([i_idx, j_idx], dim=1)  # (P,2)
    P = pairs.shape[0]

    all_mi = []
    edges_out = []
    for start in range(0, P, mi_batch_size):
        end = min(start + mi_batch_size, P)
        batch = pairs[start:end]
        xi = X[:, batch[:,0]]  # (Ncells, B)
        xj = X[:, batch[:,1]]  # (Ncells, B)
        mi = _mutual_information_binned_gpu(xi, xj, bins=mi_bins)  # (B,)
        all_mi.append(mi)
        edges_out.append(batch)

    mi_all = torch.cat(all_mi, dim=0)  # (P,)
    edges_all = torch.cat(edges_out, dim=0)  # (P,2)

    k = int(0.25 * mi_all.numel())
    if k > 0:
        topk_vals, topk_idx = torch.topk(mi_all, k=k, largest=True, sorted=False)
        edges_keep = edges_all[topk_idx].detach().cpu().numpy()
        mi_keep = topk_vals.detach().cpu().numpy()
    else:
        edges_keep = np.empty((0, 2), dtype=int)
        mi_keep = np.empty((0,), dtype=float)

    gene_names = cell_data.var.index.tolist()
    return edges_keep, mi_keep, gene_names



