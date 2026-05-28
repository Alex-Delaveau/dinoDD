#!/bin/bash
#SBATCH --job-name=dino-run-d-no-pretrain
#SBATCH --account=rbw@v100
#SBATCH --constraint=v100-32g
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e

module purge
module load pytorch-gpu/py3/2.1.1

# ── Paths (override via environment before sbatch) ───────────────────────────
DATA_PATH=${DATA_PATH:-/lustre/fshomisc/home/rech/genlpd01/ucw75ke/work/datasets/aqua20/data/aqua20/}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/dino_runs/run_d_no_pretrain_$SLURM_JOB_ID}
CODE_DIR=${CODE_DIR:-$WORK/projects/dinoDD}

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# ── Launch ───────────────────────────────────────────────────────────────────
# No --pretrained_weights: ResNet18 keeps its random PyTorch initialization,
# giving a baseline that shows what linear probing achieves without DINO.
srun torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOSTNAME:29500 \
    $CODE_DIR/eval_linear.py \
    --arch resnet18 \
    --data_path "$DATA_PATH" \
    --train_fraction 1.0 \
    --eval_split auto \
    --num_labels 20 \
    --epochs 100 \
    --lr 0.001 \
    --batch_size_per_gpu 128 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --use_wandb false \
    --dist_url "env://"

# Usage:
#   sbatch slurm/run_d_no_pretrain_baseline.slurm
#   DATA_PATH=/custom/path sbatch slurm/run_d_no_pretrain_baseline.slurm
