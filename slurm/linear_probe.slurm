#!/bin/bash
#SBATCH --job-name=dino-linear-probe
#SBATCH --partition=gpu_p13
#SBATCH --account=<ACCOUNT>@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load pytorch-gpu/py3/2.1.1 cpuarch/amd

# ── Paths (override via environment before sbatch) ───────────────────────────
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:?Must set PRETRAINED_WEIGHTS (path to DINO checkpoint.pth)}
DATA_PATH=${DATA_PATH:?Must set DATA_PATH (root with train/ and test/ subdirs)}
TRAIN_FRACTION=${TRAIN_FRACTION:-1.0}   # 0.01 = 1%, 0.05 = 5%, 1.0 = full
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/dino/linear_probe/frac${TRAIN_FRACTION}_$SLURM_JOB_ID}
CODE_DIR=${CODE_DIR:-$WORK/dinoDD}

mkdir -p "$OUTPUT_DIR"

# ── Launch ───────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOSTNAME:29500 \
    $CODE_DIR/eval_linear.py \
    --arch vit_small \
    --patch_size 16 \
    --pretrained_weights "$PRETRAINED_WEIGHTS" \
    --checkpoint_key teacher \
    --data_path "$DATA_PATH" \
    --train_fraction "$TRAIN_FRACTION" \
    --eval_split auto \
    --num_labels 20 \
    --epochs 100 \
    --lr 0.001 \
    --batch_size_per_gpu 128 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR" \
    --use_wandb false \
    --dist_url "env://"

# Usage examples:
#   1% probe:   TRAIN_FRACTION=0.01 PRETRAINED_WEIGHTS=... DATA_PATH=... sbatch slurm/linear_probe.slurm
#   5% probe:   TRAIN_FRACTION=0.05 PRETRAINED_WEIGHTS=... DATA_PATH=... sbatch slurm/linear_probe.slurm
#   Full probe: PRETRAINED_WEIGHTS=... DATA_PATH=... sbatch slurm/linear_probe.slurm
