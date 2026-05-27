#!/bin/bash
#SBATCH --job-name=dino-run-a-full
#SBATCH --account=rbw@v100
#SBATCH --partition=gpu_p2
#SBATCH --constraint=v100-32g
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e

module purge
module load pytorch-gpu/py3/2.1.1

# ── Paths (override via environment before sbatch) ───────────────────────────
# DATA_PATH: ImageFolder root for AQUA20 (with class subdirectories directly, no /train append)
DATA_PATH=${DATA_PATH:-/lustre/fshomisc/home/rech/genlpd01/ucw75ke/work/datasets/aqua20/data/aqua20/}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/dino_runs/run_a_aqua20_full_$SLURM_JOB_ID}
CODE_DIR=${CODE_DIR:-$WORK/projects/dinoDD}

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# ── Launch ───────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOSTNAME:29500 \
    $CODE_DIR/main_dino_aqua.py \
    --arch resnet18 \
    --optimizer lars \
    --use_fp16 false \
    --use_bn_in_head true \
    --data_path "$DATA_PATH" \
    --epochs 100 \
    --batch_size_per_gpu 64 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --saveckp_freq 20 \
    --lp_eval_freq 10 \
    --lp_train_frac 0.05 \
    --lp_split_seed 42 \
    --lp_real_data_path "$DATA_PATH" \
    --knn_eval_freq 0 \
    --use_wandb false \
    --dist_url "env://"

# Usage:
#   sbatch slurm/run_a_aqua20_full.slurm
#   DATA_PATH=/custom/path sbatch slurm/run_a_aqua20_full.slurm
