---
license: mit
tags:
  - single-cell
  - spatial-transcriptomics
  - graph-neural-network
library_name: pytorch
---

# HEIST

Pre-trained checkpoint for **HEIST: Hierarchical Embeddings for Integrated Spatial Transcriptomics** ([ICLR 2026](https://openreview.net/forum?id=lK82jpa8jr)).

Model code, training scripts, and tutorials live on GitHub: **https://github.com/Graph-and-Geometric-Learning/HEIST**

## Models

| Model | Path in this repo | Pretraining data | Gene identity |
|---|---|---|---|
| `v2.0/marker_all` | `v2.0/marker_all/` | Xenium (v1 + Prime 5K) + Vizgen (FFPE showcase + MERFISH 2.0) + SEA-AD MERFISH | learned per-gene embedding |
| `v2.0/marker_xenium` | `v2.0/marker_xenium/` | Xenium (v1 + Prime 5K) | learned per-gene embedding |
| `v2.0/nomarker_all` | `v2.0/nomarker_all/` | same as `marker_all` | none |
| `v1` | repo root | original release | none |

**v2.0** was pretrained on rebuilt data that fixes a preprocessing bug in v1, where every cell of a
cell type shared one gene-expression vector. v2 models embed each gene by its index in
`v2.0/gene_vocab.json` (6,495 human gene symbols). **Genes outside this vocabulary are ignored**, so
datasets with partly unknown panels still work. Mouse symbols need mapping to human orthologs first.

## Usage

Clone the repo and install dependencies:

```bash
git clone https://github.com/Graph-and-Geometric-Learning/HEIST
cd HEIST
pip install -e .
```

Embed a dataset (raw counts in `.X`, coordinates in `.obsm['spatial']`; needs a CUDA GPU):

```bash
python embed.py --adata sample.h5ad --out sample_heist.h5ad --model v2.0/marker_all
```

or from Python:

```python
from utils.inference import load_heist, embed_adata

model, gene_vocab = load_heist("v2.0/marker_all")   # "v1" loads the original model
adata.obsm["X_heist"] = embed_adata(adata, model, gene_vocab, save_root="data/preprocessed", name="sample")
```

`embed_adata` runs the same preprocessing as the v2 pretraining data: spatial tiling of large tissues,
top-200 HVGs, Leiden clusters, MAGIC, a Voronoi cell graph, and GPU KSG mutual-information gene networks.
For a walkthrough with PHATE visualization, see [`cell_embeddings.ipynb`](https://github.com/HirenMadhu/HEIST/blob/main/cell_embeddings.ipynb).

## Citation

```bibtex
@inproceedings{madhu2026heist,
  title={{HEIST}: A Graph Foundation Model for Spatial Transcriptomics and Proteomics Data},
  author={Madhu, Hiren and Rocha, Jo{\~a}o Felipe and Huang, Tinglin and Viswanath, Siddharth and Krishnaswamy, Smita and Ying, Rex},
  booktitle={The Fourteenth International Conference on Learning Representations},
  year={2026},
  url={https://openreview.net/forum?id=lK82jpa8jr}
}
```
