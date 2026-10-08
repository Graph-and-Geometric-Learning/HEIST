"""Embed a spatial transcriptomics dataset with a pretrained HEIST model.

    python embed.py --adata sample.h5ad --out sample_heist.h5ad                    # v2.0/marker_all
    python embed.py --adata sample.h5ad --out sample_heist.h5ad --model v2.0/marker_xenium
    python embed.py --adata sample.h5ad --out sample_heist.h5ad --cell_type cell_type

Input: raw counts in .X and cell coordinates in .obsm['spatial'].
Output: the same AnnData with .obsm['X_heist'] = [cell-graph embedding | gene embedding] per cell.
Genes outside the HEIST vocabulary are ignored (listed in .uns['heist']['ignored_genes']).
Needs a CUDA GPU (gene networks are built with GPU mutual information).
"""
import argparse

import scanpy as sc

from utils.inference import MODELS, embed_adata, load_heist


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adata", required=True, help="input .h5ad (raw counts, obsm['spatial'])")
    p.add_argument("--out", required=True, help="output .h5ad with obsm['X_heist']")
    p.add_argument("--model", default="v2.0/marker_all", help=f"one of {MODELS} or a local model folder")
    p.add_argument("--cell_type", default=None, help="obs column with cell types (default: Leiden)")
    p.add_argument("--spatial", default="spatial", help="obsm key with cell coordinates")
    p.add_argument("--cache", default="data/preprocessed", help="folder for the cached graphs")
    p.add_argument("--name", default=None, help="cache file prefix (default: input file name)")
    args = p.parse_args()

    adata = sc.read_h5ad(args.adata)
    model, vocab = load_heist(args.model)
    name = args.name or args.adata.rsplit("/", 1)[-1].removesuffix(".h5ad")
    adata.obsm["X_heist"] = embed_adata(adata, model, vocab, args.cache, name,
                                        cell_type=args.cell_type, spatial=args.spatial)
    ignored = [] if vocab is None else [g for g in adata.var_names if str(g).upper() not in vocab]
    adata.uns["heist"] = {"model": args.model, "ignored_genes": ignored}
    adata.write_h5ad(args.out)
    print(f"[HEIST] {adata.n_obs} cells -> obsm['X_heist'] {adata.obsm['X_heist'].shape}, saved {args.out}")


if __name__ == "__main__":
    main()
