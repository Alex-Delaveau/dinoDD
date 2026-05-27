#!/bin/bash
#SBATCH --job-name=dino-run-a-full-dev
#SBATCH --account=rbw@v100
#SBATCH --constraint=v100-32g
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e

module purge
module load pytorch-gpu/py3/2.1.1

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_PATH=${DATA_PATH:-/lustre/fshomisc/home/rech/genlpd01/ucw75ke/work/datasets/aqua20/data/aqua20/}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/dino_runs/run_a_aqua20_full_dev_$SLURM_JOB_ID}
CODE_DIR=${CODE_DIR:-$WORK/projects/dinoDD}

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# ── Launch (5 epochs, lp_eval_freq=1 to trigger LP on epoch 1) ───────────────
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
    --epochs 5 \
    --warmup_epochs 1 \
    --batch_size_per_gpu 64 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --saveckp_freq 5 \
    --lp_eval_freq 1 \
    --lp_train_frac 0.05 \
    --lp_split_seed 42 \
    --lp_real_data_path "$DATA_PATH" \
    --knn_eval_freq 0 \
    --use_wandb false \
    --dist_url "env://"

# Usage:
#   sbatch slurm/run_a_aqua20_full_dev.slurm
