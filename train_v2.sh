#!/bin/bash
# HEIST v2 pretraining. Usage: sbatch --job-name=<RUN> train_v2.sh <RUN> <DATA_DIR> [extra main_ddp flags]
#   sbatch --job-name=HEIST_v2.0_nomarker_all  train_v2.sh HEIST_v2.0_nomarker_all  data/pretraining_v2/
#   sbatch --job-name=HEIST_v2.0_marker_all    train_v2.sh HEIST_v2.0_marker_all    data/pretraining_v2/        --marker_embedding --num_markers 6495
#   sbatch --job-name=HEIST_v2.0_marker_xenium train_v2.sh HEIST_v2.0_marker_xenium data/pretraining_v2_xenium/ --marker_embedding --num_markers 6495
# Resubmitting the same command resumes from saved_models/<RUN>_last.pth.
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --account=pi_sk2433
#SBATCH --partition=gpu_h200,gpu_b200
#SBATCH --gpus=2
#SBATCH --nodes=1
#SBATCH --mem=128G
#SBATCH --output=./logs/slurm/pretraining/%x_%j.out
#SBATCH --error=./logs/slurm/pretraining/%x_%j.err
#SBATCH --mail-type=FAIL,TIME_LIMIT

RUN=$1; DATA_DIR=$2; shift 2
cd /home/hm638/project_pi_zy286/hm638/HEIST
ml uv
source .venv/bin/activate

# main_ddp.setup() reads HEIST_MASTER_PORT (not MASTER_PORT); a per-job port stops runs that land on
# the same node from colliding on the default 29500.
export HEIST_MASTER_PORT=$((20000 + SLURM_JOB_ID % 20000))
export NCCL_SOCKET_IFNAME=lo

SAVE=saved_models/${RUN}.pth
RESUME=()
[ -f "saved_models/${RUN}_last.pth" ] && RESUME=(--resume "saved_models/${RUN}_last.pth")

python main_ddp.py --data_dir "$DATA_DIR" --batch_size 384 --pe --cross_message_passing --rank_pe_fixed \
    --save_path "$SAVE" --wandb --run_name "$RUN" "${RESUME[@]}" "$@"
