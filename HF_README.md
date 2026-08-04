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

## Usage

Clone the repo and install dependencies:

```bash
git clone https://github.com/Graph-and-Geometric-Learning/HEIST
cd HEIST
pip install -e .
```

Then load the pre-trained weights:

```python
from model.model import GraphEncoder

model = GraphEncoder.from_pretrained("HirenMadhu/HEIST")
model.eval()
```

For a full embedding-extraction tutorial (preprocessing, graph construction, PHATE visualization), see [`cell_embeddings.ipynb`](https://github.com/HirenMadhu/HEIST/blob/main/cell_embeddings.ipynb) in the GitHub repo.

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
