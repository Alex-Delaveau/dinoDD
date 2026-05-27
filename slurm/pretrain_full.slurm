#!/bin/bash
#SBATCH --job-name=dino-pretrain-full
#SBATCH --partition=gpu_p2
#SBATCH --account=rbw@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load pytorch-gpu/py3/2.1.1

# ── Paths (override via environment before sbatch) ───────────────────────────
# DATA_PATH must point to the ImageFolder root (with class subdirectories directly).
# main_dino_aqua.py calls datasets.ImageFolder(args.data_path) without appending /train.
DATA_PATH=${DATA_PATH:-/lustre/fshomisc/home/rech/genlpd01/ucw75ke/work/datasets/aqua20/data/aqua20/train}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/projects/dino/pretrain_full/$SLURM_JOB_ID}
CODE_DIR=${CODE_DIR:-$WORK/dinoDD}

mkdir -p "$OUTPUT_DIR"

# ── Launch ───────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOSTNAME:29500 \
    $CODE_DIR/main_dino_aqua.py \
    --arch vit_small \
    --patch_size 16 \
    --data_path "$DATA_PATH" \
    --epochs 100 \
    --batch_size_per_gpu 64 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --saveckp_freq 20 \
    --use_wandb false \
    --dist_url "env://"

# Usage:
#   DATA_PATH=$SCRATCH/aqua20/train sbatch slurm/pretrain_full.slurm
