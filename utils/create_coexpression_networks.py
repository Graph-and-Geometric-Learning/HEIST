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

@torch.no_grad()
def _mutual_information_ksg_gpu(xi, xj, k=3, eps_noise=1e-10):
    """
    GPU Kraskov-Stoegbauer-Grassberger (KSG) estimator I for continuous-continuous MI,
    matching sklearn.feature_selection.mutual_info_regression semantics.

    xi, xj: (N, B) columns are the two variables of B candidate pairs over N samples.
    Returns MI (B,), clamped >= 0.

    For each pair/point: kth nearest-neighbor radius in the JOINT space under the
    Chebyshev (max) metric, then marginal counts n_x, n_y of OTHER points strictly
    within that radius; MI = psi(N) + psi(k) - <psi(n_x+1) + psi(n_y+1)>.
    """
    from torch.special import digamma
    N, B = xi.shape
    device = xi.device
    # tie-breaking noise (as sklearn does): scale ~ 1e-10 * max(1, mean|x|)
    xi = xi + eps_noise * xi.abs().mean(0, keepdim=True).clamp_min(1.0) * torch.randn_like(xi)
    xj = xj + eps_noise * xj.abs().mean(0, keepdim=True).clamp_min(1.0) * torch.randn_like(xj)
    xiT = xi.T.contiguous()                                  # (B, N)
    xjT = xj.T.contiguous()
    dxi = (xiT[:, :, None] - xiT[:, None, :]).abs()          # (B, N, N) x-marginal dists
    dxj = (xjT[:, :, None] - xjT[:, None, :]).abs()          # (B, N, N) y-marginal dists
    djoint = torch.maximum(dxi, dxj)                         # (B, N, N) Chebyshev joint
    # kth neighbor EXCLUDING self = (k+1)th smallest including self (self dist = 0)
    kth = torch.topk(djoint, k + 1, dim=2, largest=False).values[:, :, k]   # (B, N)
    radius = torch.nextafter(kth, torch.zeros_like(kth))     # strictly-less radius (KSG)
    nx = (dxi <= radius[:, :, None]).sum(dim=2) - 1          # (B, N) exclude self
    ny = (dxj <= radius[:, :, None]).sum(dim=2) - 1
    psiN = digamma(torch.tensor(float(N), device=device))
    psik = digamma(torch.tensor(float(k), device=device))
    mi = psiN + psik - (digamma((nx + 1).float()) + digamma((ny + 1).float())).mean(dim=1)
    return mi.clamp_min(0.0)


def build_gene_network_gpu(cell_data,
                           topk_per_gene=200,
                           min_abs_corr=None,
                           mi_bins=32,
                           mi_batch_size=20000,
                           ksg_k=3,
                           ksg_batch=48,
                           std_coeff=1.0,
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
    # KSG builds a (B,N,N) tensor per batch, so batch over PAIRS in small chunks (ksg_batch),
    # independent of mi_batch_size (which sized the old binned estimator).
    for start in range(0, P, ksg_batch):
        end = min(start + ksg_batch, P)
        batch = pairs[start:end]
        xi = X[:, batch[:, 0]]  # (Ncells, B)
        xj = X[:, batch[:, 1]]  # (Ncells, B)
        mi = _mutual_information_ksg_gpu(xi, xj, k=ksg_k)  # (B,) correct KSG estimator
        all_mi.append(mi)
        edges_out.append(batch)

    mi_all = torch.cat(all_mi, dim=0)  # (P,)
    edges_all = torch.cat(edges_out, dim=0)  # (P,2)

    # Adaptive edge selection: keep pairs with MI > mean(MI) + std_coeff * std(MI).
    if mi_all.numel() >= 2:
        thresh = mi_all.mean() + std_coeff * mi_all.std()
        keep = mi_all > thresh
        edges_keep = edges_all[keep].detach().cpu().numpy()
        mi_keep = mi_all[keep].detach().cpu().numpy()
    else:
        edges_keep = np.empty((0, 2), dtype=int)
        mi_keep = np.empty((0,), dtype=float)

    gene_names = cell_data.var.index.tolist()
    return edges_keep, mi_keep, gene_names



