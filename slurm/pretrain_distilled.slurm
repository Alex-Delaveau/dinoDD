#!/bin/bash
#SBATCH --job-name=dino-pretrain-distilled
#SBATCH --partition=gpu_p4
#SBATCH --account=<ACCOUNT>@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load pytorch-gpu/py3/2.1.1 cpuarch/amd

# ── Paths (override via environment before sbatch) ───────────────────────────
DISTILLED_DATA=${DISTILLED_DATA:?Must set DISTILLED_DATA (path to data.pth)}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/dino/pretrain_distilled/$SLURM_JOB_ID}
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
    --distilled_data_path "$DISTILLED_DATA" \
    --num_dataset_repeats 100 \
    --epochs 100 \
    --batch_size_per_gpu 64 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --saveckp_freq 20 \
    --use_wandb false \
    --dist_url "env://"

# Usage:
#   DISTILLED_DATA=$SCRATCH/aqua20/data.pth sbatch slurm/pretrain_distilled.slurm
