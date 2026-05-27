#!/bin/bash
#SBATCH --job-name=dino-knn-eval
#SBATCH --partition=gpu_p13
#SBATCH --account=<ACCOUNT>@v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=01:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load pytorch-gpu/py3/2.1.1 cpuarch/amd

# ── Paths (override via environment before sbatch) ───────────────────────────
# DATA_PATH must point to a directory with train/ and test/ (or val/) subdirs (ImageFolder layout).
PRETRAINED_WEIGHTS=${PRETRAINED_WEIGHTS:?Must set PRETRAINED_WEIGHTS (path to DINO checkpoint.pth)}
DATA_PATH=${DATA_PATH:-/lustre/fshomisc/home/rech/genlpd01/ucw75ke/work/datasets/aqua20/data/aqua20/}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/dino/knn_eval/$SLURM_JOB_ID}
CODE_DIR=${CODE_DIR:-$WORK/dinoDD}

mkdir -p "$OUTPUT_DIR"

# ── Launch ───────────────────────────────────────────────────────────────────
torchrun \
    --nproc_per_node=1 \
    --nnodes=1 \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOSTNAME:29500 \
    $CODE_DIR/eval_knn.py \
    --arch vit_small \
    --patch_size 16 \
    --pretrained_weights "$PRETRAINED_WEIGHTS" \
    --checkpoint_key teacher \
    --data_path "$DATA_PATH" \
    --nb_knn 10 20 \
    --num_workers 8 \
    --dist_url "env://"

# Usage:
#   PRETRAINED_WEIGHTS=... DATA_PATH=$SCRATCH/aqua20 sbatch slurm/knn_eval.slurm
