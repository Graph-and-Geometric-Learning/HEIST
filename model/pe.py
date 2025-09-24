import torch

import torch

def calculate_sinusoidal_pe(high_level_graph, low_level_graphs, pe_dim):
    # Step 1: Calculate cell positional encodings (Dist_i)
    num_nodes = high_level_graph.num_nodes
    cell_locations = high_level_graph.X  # Shape: [num_cells, 2]
    anchor_nodes = torch.randint(0, num_nodes, (pe_dim,))
    
    # Compute distance vectors (Dist_i) between each cell and the anchors
    # dist_matrix = torch.cdist(cell_locations, cell_locations[anchor_nodes])  # Shape: [num_cells, num_cells]
    x, y = cell_locations[:, 0], cell_locations[:, 1]  # Extract x and y coordinates
    half_dim = pe_dim // 2  # Half for x, half for y
    i = torch.arange(half_dim // 2, device=cell_locations.device)  # i indices
    j = torch.arange(half_dim // 2, device=cell_locations.device)  # j indices

    # Compute denominator 10000^(4i/d) and 10000^(4j/d)
    div_term_x = 10000 ** (4 * i / half_dim)
    div_term_y = 10000 ** (4 * j / half_dim)

    # Compute positional encodings
    pe_x = torch.zeros((cell_locations.shape[0], half_dim), device=cell_locations.device)
    pe_y = torch.zeros((cell_locations.shape[0], half_dim), device=cell_locations.device)

    pe_x[:, 0::2] = torch.sin(x[:, None] / div_term_x)  # sin terms for x
    pe_x[:, 1::2] = torch.cos(x[:, None] / div_term_x)  # cos terms for x

    pe_y[:, 0::2] = torch.sin(y[:, None] / div_term_y)  # sin terms for y
    pe_y[:, 1::2] = torch.cos(y[:, None] / div_term_y)  # cos terms for y

    # Concatenate positional encodings for x and y
    pe = torch.cat([pe_x, pe_y], dim=-1)

    # Use the distance matrix as positional encoding for the high-level graph
    high_level_graph.pe = pe

    # Step 2: Calculate gene positional encodings (RankNorm * Dist_i)
    gene_expressions = low_level_graphs.X.squeeze(-1)  # Shape: [num_genes]
    gene_batches = low_level_graphs.batch  # Shape: [num_genes]

    # Filter by batch and calculate RankNorm for each gene batch
    rank_norm = torch.zeros_like(gene_expressions).to(high_level_graph.X.device).float()
    unique_batches = gene_batches.unique()
    for b in unique_batches:
        batch_mask = (gene_batches == b)
        batch_gene_expressions = gene_expressions[batch_mask]
        batch_ranks = torch.argsort(-batch_gene_expressions, dim=0)  # Descending order
        batch_rank_norm = torch.linspace(0, 1, steps=batch_ranks.size(0), device=batch_gene_expressions.device)
        rank_norm[torch.where(batch_mask)[0]] = batch_rank_norm[torch.argsort(batch_ranks)]

    div_term = 10000 * (2 * torch.arange(0, pe_dim, 2, device=rank_norm.device).float() /half_dim)
    gene_sinusoidal_pe = torch.zeros(rank_norm.size(0), pe_dim, device=rank_norm.device)
    gene_sinusoidal_pe[:, 0::2] = torch.sin(rank_norm.unsqueeze(-1) * div_term)
    gene_sinusoidal_pe[:, 1::2] = torch.cos(rank_norm.unsqueeze(-1) * div_term)
    # Set low-level graph PE
    low_level_graphs.pe = gene_sinusoidal_pe.to(high_level_graph.X.device)

    return high_level_graph, low_level_graphs

def calculate_anchor_pe(high_level_graph, low_level_graphs, pe_dim):
    # Step 1: Calculate cell positional encodings (Dist_i)
    num_nodes = high_level_graph.num_nodes
    cell_locations = high_level_graph.X  # Shape: [num_cells, 2]
    anchor_nodes = torch.randint(0, num_nodes, (pe_dim,))
    
    # Compute distance vectors (Dist_i) between each cell and the anchors
    dist_matrix = torch.cdist(cell_locations, cell_locations[anchor_nodes])  # Shape: [num_cells, num_cells]

    high_level_graph.pe = dist_matrix

    # Step 2: Calculate gene positional encodings (RankNorm * Dist_i)
    gene_expressions = low_level_graphs.X.squeeze(-1)  # Shape: [num_genes]
    gene_batches = low_level_graphs.batch  # Shape: [num_genes]

    # Filter by batch and calculate RankNorm for each gene batch
    rank_norm = torch.zeros_like(gene_expressions).to(high_level_graph.X.device).float()
    unique_batches = gene_batches.unique()
    for b in unique_batches:
        batch_mask = (gene_batches == b)
        batch_gene_expressions = gene_expressions[batch_mask]
        batch_ranks = torch.argsort(-batch_gene_expressions, dim=0)  # Descending order
        batch_rank_norm = torch.linspace(0, 1, steps=batch_ranks.size(0), device=batch_gene_expressions.device)
        rank_norm[torch.where(batch_mask)[0]] = batch_rank_norm[torch.argsort(batch_ranks)]

    # Apply sinusoidal encoding to the gene positional encodings
    div_term = torch.exp(torch.arange(0, pe_dim, 2, device=rank_norm.device).float() * -(torch.log(torch.tensor(10000.0)) / pe_dim))
    gene_sinusoidal_pe = torch.zeros(rank_norm.size(0), pe_dim, device=rank_norm.device)
    gene_sinusoidal_pe[:, 0::2] = torch.sin(rank_norm.unsqueeze(-1) * div_term)
    gene_sinusoidal_pe[:, 1::2] = torch.cos(rank_norm.unsqueeze(-1) * div_term)
    # Set low-level graph PE
    low_level_graphs.pe = gene_sinusoidal_pe.to(high_level_graph.X.device)

    return high_level_graph, low_level_graphs