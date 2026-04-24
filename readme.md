
# 📚 **HEIST**

This project implements a distributed training pipeline for learning representations over spatial cell graphs using PyTorch, PyTorch Geometric, and Distributed Data Parallel (DDP).

---


## Setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

Install uv if you don't have it:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then create the environment and install dependencies:

```bash
uv sync
```

This will create a `.venv/` in the project root with all dependencies from [pyproject.toml](pyproject.toml) / `uv.lock`. Prefix commands with `uv run` (e.g. `uv run python main_ddp.py ...`) or activate the venv directly:

```bash
source .venv/bin/activate
```

---

## 📂 **Data Preparation**

* Place your dataset in the `data/pretraining/` directory.
* The data should be preprocessed and saved as PyTorch objects using `torch.save`.

---

## ⚙️ **Training the Model**

Run the the full HEIST model on all the available GPUs, use the following command to start distributed training:

```bash
python main_ddp.py --data_dir data/pretraining/ --pe --cross_message_passing
```

If you want to train only on single GPU, change `main_ddp` with `main`:

```bash
python main.py --data_dir data/pretraining/ --pe --cross_message_passing
```


### 💡 **Important Arguments:**

| Argument       | Description                 | Default           |
| -------------- | --------------------------- | ----------------- |
| `--data_dir`   | Path to preprocessed data   | data/pretraining/ |
| `--pe_dim`     | Positional Encoding Dim     | 128               |
| `--init_dim`   | Initial MLP Hidden Dim      | 128               |
| `--hidden_dim` | Hidden Dimension            | 128               |
| `--output_dim` | Output Dimension            | 128               |
| `--num_layers` | Number of MLP Layers        | 10                |
| `--num_heads`  | Number of Transformer Heads | 8                 |
| `--batch_size` | Batch Size                  | 128                |
| `--lr`         | Learning Rate               | 1e-3              |
| `--wd`         | Weight Decay                | 3e-3              |
| `--num_epochs` | Number of Training Epochs   | 20                |

---

## 📈 **Model Checkpoints**

* Checkpoints are saved automatically under the `saved_models/` directory.
* The best model is saved as `HEIST.pth`.

---

## 🤗 **Loading Pre-trained Models from Hugging Face**

Pre-trained HEIST checkpoints are hosted on the [Hugging Face Hub](https://huggingface.co/HirenMadhu). Load any of them with three lines:

```python
from model.model import GraphEncoder
model = GraphEncoder.from_pretrained("HirenMadhu/HEIST").to(device)
model.eval()
```

For an end-to-end tutorial on extracting cell embeddings from a pre-trained model — including preprocessing, graph construction, and visualization with PHATE — see [`cell_embeddings.ipynb`](cell_embeddings.ipynb).

---

## 🛠 **Resuming Training**

If you want to resume from a saved checkpoint, ensure that the model and optimizer state dictionaries are correctly loaded in the script.

---


## 📊 **Model Evaluation**

After training, you can evaluate the model using the provided evaluation script.

### ✅ **Run the Evaluation**

```bash
bash eval.sh
```

This script will:

1. Activate the uv-managed `.venv` environment.
2. Run a series of evaluations across multiple datasets and tasks:

   * **Representation Space Calculation**

     * `dfci`, `upmc`, `charville`, `sea`, `melanoma`, `placenta`, `lung`
   * **Tissue Classification**

     * Predict primary outcomes and recurrence for clinical datasets.
   * **Melanoma and Cell Clustering Evaluations**
   * **Placenta Dataset Analysis**
   * **Gene Imputation Tasks**

     * Both standard and fine-tuned versions.

---

### 📁 **Generated Outputs**

* Evaluation results will be stored in the corresponding directories or logged to the console.
* Ensure that the trained model checkpoint is available and named correctly (default: `HEIST`).

---

## License

This work is licensed under a [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{
madhu2026heist,
title={{HEIST}: A Graph Foundation Model for Spatial Transcriptomics and Proteomics Data},
author={Hiren Madhu and Jo{\~a}o Felipe Rocha and Tinglin Huang and Siddharth Viswanath and Smita Krishnaswamy and Rex Ying},
booktitle={The Fourteenth International Conference on Learning Representations},
year={2026},
url={https://openreview.net/forum?id=lK82jpa8jr}
}
```
