# HEIST preprocessing

HEIST turns a spatial transcriptomics sample into a two-level graph: one **cell graph** (Voronoi
neighbors) and, for every cell, a **gene graph** whose topology is a mutual-information gene network
of that cell's cluster and whose node features are that cell's own expression.

Code: `utils/preprocess.py` (pipeline) and `utils/create_coexpression_networks.py` (gene networks).
The same functions build the v2 pretraining data (`data_prep/build_chunks.py`) and run inference
(`utils/inference.py`, `embed.py`, `cell_embeddings.ipynb`), so new data is processed exactly like the
data the models were trained on.

## Input

An AnnData with raw counts in `.X` and cell coordinates in `.obsm['spatial']`. Cell types in
`.obs` are optional.

## Steps

1. **Gene vocabulary** (v2 models): genes not in `gene_vocab.json` (6,495 human symbols, matched
   case-insensitively) are dropped, including control probes (`filter_genes_to_vocab`).
2. **Spatial tiling**: samples with more than 30,000 cells are split recursively into quadrants at
   the bounding-box midpoint until every tile has fewer than 35,000 cells (`partition`). Each tile
   is processed independently.
3. **Expression**: drop genes seen in fewer than 3 cells, normalize to 10,000 counts per cell,
   log1p, and keep the top 200 highly variable genes (all genes if 200 or fewer).
4. **Clusters**: the given cell types, or Leiden on a 50-PC kNN graph. The resolution is picked per
   tile so that about 7 clusters of 50 or more cells remain (`leiden_resolution='auto'`). Clusters
   under 50 cells are merged into their most connected neighbor.
5. **Denoising**: MAGIC on the expression matrix.
6. **Cell graph**: Voronoi tessellation of the cell coordinates; standardized (x, y) as cell features.
7. **Gene networks**: one per cluster. MI is estimated on the GPU for every gene pair with the KSG
   estimator (k = 3), identical to `sklearn.feature_selection.mutual_info_regression`. Pairs with
   MI > mean + 1 SD over all pairs become edges.
8. **Per-cell gene graphs**: each cell gets its cluster's gene network with its own expression as
   node features, plus `marker_id` (vocabulary index per gene node) for v2 models.

Output: `[cell_graph, gene_graph_cell_0, ..., gene_graph_cell_N]`, cached to `<save_root>/<name>.pt`.
`cell_graph` also stores `gene_names`, `gene_ids`, `cell_types` and the Leiden resolution used.

A CUDA GPU is required for step 7.

## Pretraining data (v2)

`data_prep/` rebuilds the pretraining corpus from public raw data: `download_*.sh` fetches
Xenium, Vizgen and SEA-AD; `build_gene_vocab.py` writes the vocabulary; `build_chunks.py` (one SLURM
task per sample, `build_chunks.sh`) writes `data/pretraining_v2/<source>_preprocessed/*.pt`. Training
chunks also drop cells with fewer than 10 transcripts; inference keeps every cell.
