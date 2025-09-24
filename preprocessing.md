
# 🧬 **HEIST Preprocessing Guide**

This script performs preprocessing on raw spatial transcriptomics data, constructing cell-level and gene regulatory networks (GRNs) and saving them in PyTorch Geometric (PyG) format for downstream tasks.

---

## 📂 **Input Requirements**

* Raw `.h5ad` files should be placed in the directory:

  ```
  data/sea_raw/
  ```
* Example file structure:

  ```
  data/sea_raw/sample1.h5ad
  data/sea_raw/sample2.h5ad
  ```

---

## 📦 **Dependencies**

```bash
conda activate HEIST  # Or your preferred environment
pip install scanpy magic-impute torch-geometric scikit-learn tqdm networkx
```

---

## 🚀 **How to Run**

```bash
python utils/preprocess.py  # Replace with the actual filename
```

---

## ⚙️ **What This Script Does**

1. **Loads Raw Data**:
   Loads `.h5ad` files and extracts metadata like Braak stages and spatial locations.

2. **Preprocesses Data**:

   * Filters genes expressed in fewer than 3 cells.
   * Normalizes and log-transforms expression values.
   * Applies MAGIC for denoising.

3. **Constructs Graphs**:

   * **Cell-Level Graph**: Based on spatial proximity using `sc.pp.neighbors()`.
   * **Gene Regulatory Networks (GRNs)**:

     * Built using Mutual Information (MI) between gene pairs.
     * Edges are thresholded at MI > 0.35.
     * Converted to PyG format using `from_networkx()`.

4. **Saves Preprocessed Graphs**:

   * Output is saved to:

     ```
     data/sea_preprocessed/
     ```
   * Each file contains a list of graphs:

     * One high-level cell graph.
     * Multiple low-level gene graphs (one per cell).

---

## 📁 **Generated Output Example**

* Cell and gene graphs are saved as `.pt` files:

  ```
  data/sea_preprocessed/sample1_0.pt
  ```



---

## 📈 **Expected Console Output**

```
Loaded file sample1.
Now pre-processing the data.
Creating the GRNs using MI
Converting to PyG format
Data saved
Number of graphs in the dataset: 3500
```

