#!/usr/bin/env bash
# One-time (idempotent) reorganization of the mioflow_heist/ folder:
# moves scattered artifacts at the root into outputs/{h5ad,trajectories,checkpoints,figures}.
# Step scripts, helpers (paths.py, heist_preprocess.py, aux_model.py), src/, ortho_model/,
# data/ and plot.ipynb stay at the root. Safe to re-run.
set -e
cd "$(dirname "$0")"

mkdir -p outputs/h5ad outputs/trajectories outputs/checkpoints outputs/figures data

shopt -s nullglob
mv -f axolotl_heist*.h5ad          outputs/h5ad/          2>/dev/null || true
mv -f *trajs*.npy phate_gene.npy   outputs/trajectories/  2>/dev/null || true
mv -f *.pth                        outputs/checkpoints/   2>/dev/null || true
mv -f fig5_*.png diag_*.png        outputs/figures/       2>/dev/null || true

echo "reorganized mioflow_heist/ -> outputs/{h5ad,trajectories,checkpoints,figures}"
ls -1 outputs/*/ | sed 's/^/  /' | head -60
