# mioflow_heist

Running MIOFlow trajectory inference on the axolotl regeneration data (bp542's `axolotl_ran.h5ad`
subset: 11,716 injured neurogenic-lineage cells) using **HEIST** embeddings, to reproduce Fig 5.

## Layout

```
mioflow_heist/
  paths.py                 central path config — import in every script (no hard-coded paths)
  heist_preprocess.py      corrected copy of utils/preprocess.py (per-cell gene graph; no aliasing)
  aux_model.py             small MLP Decoder shared by steps 03 & 05
  01_heist_embeddings.py   pretrained HEIST GraphEncoder -> obsm['X_heist'] (256D)
  02_gaga_latent.py        Harmony(Batch) + PHATE(X_heist) -> GAGA 2D latent   [older approach]
  03_aux_decoders.py       2D latent -> gene-PCA (200) and LR_feats (640) decoders
  04_mioflow_train.py      train MIOFlow ODE over time_bins -> ptrajs
  05_decode_trajs.py       decode ptrajs -> gtrajs (genes) + strajs (LR)
  06_plot.py               render Fig 5 -> outputs/figures/fig5_heist{SUFFIX}.png
  07_ortho_embeddings.py   run the first-commit "ortho" checkpoint on cached graphs -> X_heist_ortho
  08_phate_heist_latent.py CURRENT latent: gene-PHATE + lambda*HEIST-spatial-half -> GAGA 2D  [best]
  run_sweep.sh             lambda sweep driver (regular | ortho)
  reorganize.sh            one-time: move root artifacts into outputs/
  plot.ipynb               notebook version of the Fig 5 plot
  src/                     vendored MIOFlow libs (gaga.py, mioflow.py)
  ortho_model/             first-commit HEIST model files (to load the ortho checkpoint)
  data/                    preprocessed PyG graphs (axolotl_heist.pt)
  outputs/{h5ad,trajectories,checkpoints,figures}/   all artifacts (by SUFFIX)
```

`SUFFIX` (env `HEIST_SUFFIX`) tags every artifact for a run, e.g. `_l0p25` (lambda=0.25, pretrained)
or `_ortho_l0p5`. `HEIST_H5AD` / `HEIST_BASE_H5AD` / `HEIST_LAMBDA` override the defaults.

## Run

```bash
# one-time embeddings (pretrained + ortho)
.venv/bin/python mioflow_heist/01_heist_embeddings.py
.venv/bin/python mioflow_heist/07_ortho_embeddings.py
# lambda sweep -> Fig 5 per lambda
mioflow_heist/run_sweep.sh regular 0 0.25 0.5 1 2
mioflow_heist/run_sweep.sh ortho   0.25 0.5 1 2
```

## Findings

- The current latent (step 08) = **gene-PHATE base + lambda-weighted HEIST spatial half**. Best result at
  **lambda = 0.25** (pretrained): clean pseudotime gradient, coherent trajectories, monotonic NCAN:SDC3
  signaling. Larger lambda lets the (batch-neutral) HEIST spatial half blend in; too large washes out the
  lineage (spearman(latent, pseudotime): 0.72 @0 -> 0.70 @0.25 -> 0.23 @1).
- GRN (`utils/create_coexpression_networks.py`) uses a GPU KSG (Kraskov k=3) MI estimator + adaptive
  `MI > mean+std` edge threshold (validated vs sklearn).
- The ortho checkpoint underperforms the pretrained HEIST here (weaker decoders / embedding).
```
