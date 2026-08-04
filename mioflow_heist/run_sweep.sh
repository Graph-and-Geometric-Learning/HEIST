#!/usr/bin/env bash
# Lambda sweep for the gene-PHATE + weighted HEIST-spatial-half MIOFlow latent.
#   latent = GAGA2D( concat[ blocknorm(PHATE(gene PCA)) , lambda * blocknorm(HEIST spatial half) ] )
# For each lambda: 08 (latent) -> 03 (decoders) -> 04 (MIOFlow) -> 05 (decode) -> 06 (Fig 5).
#
# Usage:
#   ./run_sweep.sh regular 0 0.25 0.5 1 2      # pretrained HEIST spatial half
#   ./run_sweep.sh ortho   0.25 0.5 1 2        # ortho-model spatial half (skip 0; model-independent)
set -e
cd "$(dirname "$0")/.."                        # repo root
REPO=$(pwd); MH="$REPO/mioflow_heist"; PY="$REPO/.venv/bin/python"; H5="$MH/outputs/h5ad"

MODEL=${1:-regular}; shift || true
LAMBDAS=("$@"); [ ${#LAMBDAS[@]} -eq 0 ] && LAMBDAS=(0 0.25 0.5 1 2)
if [ "$MODEL" = ortho ]; then BASE="$H5/axolotl_heist_ortho.h5ad"; TAG="_ortho"; else BASE="$H5/axolotl_heist.h5ad"; TAG=""; fi
lamtag(){ echo "l$1" | sed 's/\./p/'; }       # 0.25 -> l0p25 ; 1 -> l1

for lam in "${LAMBDAS[@]}"; do
  suf="${TAG}_$(lamtag "$lam")"
  echo "========== MODEL=$MODEL LAMBDA=$lam SUFFIX=$suf =========="
  export HEIST_SUFFIX="$suf" HEIST_LAMBDA="$lam" HEIST_BASE_H5AD="$BASE" HEIST_H5AD="$H5/axolotl_heist${suf}.h5ad"
  $PY "$MH/08_phate_heist_latent.py"
  $PY "$MH/03_aux_decoders.py"
  $PY "$MH/04_mioflow_train.py"
  $PY "$MH/05_decode_trajs.py"
  $PY "$MH/06_plot.py"
done
echo "========== SWEEP DONE ($MODEL) =========="
