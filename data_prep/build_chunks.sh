#!/bin/bash
# One array task per sample. Submit per source (indices = sorted samples, see build_chunks.py):
#   sbatch --array=0-41 data_prep/build_chunks.sh 10x     (35 v1 Xenium + 7 Xenium Prime 5K)
#   sbatch --array=0-26 data_prep/build_chunks.sh sea     (27 SEA-AD donors, 2024-12-11 release)
#   sbatch --array=0-N  data_prep/build_chunks.sh vizgen  (17 v1 FFPE showcase + MERFISH 2.0 regions)
# Finished chunks are skipped, so failed tasks can be resubmitted as-is.
# Requires data/pretraining_v2/{manifest.parquet,gene_vocab.json}; rebuild the vocab (and every
# chunk) if samples are added, since gene_ids index into it.
#SBATCH --job-name=heist_prep
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --account=pi_sk2433
#SBATCH --partition=gpu_h200,gpu_b200
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/prep/%x_%A_%a.out
#SBATCH --error=./logs/slurm/prep/%x_%A_%a.err

cd /home/hm638/project_pi_zy286/hm638/HEIST
ml uv
source .venv/bin/activate
python data_prep/build_chunks.py --source "$1"
