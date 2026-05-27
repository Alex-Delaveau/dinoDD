#!/bin/bash
#SBATCH --job-name=dino-run-b-distilled-dev
#SBATCH --partition=gpu_p13
#SBATCH --account=<COMPTE_A_REMPLIR>@v100
#SBATCH --qos=qos_gpu-dev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module purge
module load pytorch-gpu/py3/2.1.1 cpuarch/amd

# ── Paths ─────────────────────────────────────────────────────────────────────
DISTILLED_DATA=${DISTILLED_DATA:?Must set DISTILLED_DATA (path to distilled .pth file)}
REAL_DATA_PATH=${REAL_DATA_PATH:-/lustre/fshomisc/home/rech/genlpd01/ucw75ke/work/datasets/aqua20}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/dino_runs/run_b_distilled_dev_$SLURM_JOB_ID}
CODE_DIR=${CODE_DIR:-$WORK/dinoDD}

mkdir -p "$OUTPUT_DIR"

# ── Launch (200 epochs, lp_eval_freq=50 → 4 LP evals to validate pipeline) ───
torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOSTNAME:29500 \
    $CODE_DIR/main_dino_aqua.py \
    --arch resnet18 \
    --distilled_data_path "$DISTILLED_DATA" \
    --num_dataset_repeats 50 \
    --epochs 200 \
    --batch_size_per_gpu 20 \
    --optimizer lars \
    --use_fp16 false \
    --use_bn_in_head true \
    --num_workers 4 \
    --output_dir "$OUTPUT_DIR" \
    --saveckp_freq 50 \
    --lp_eval_freq 50 \
    --lp_train_frac 0.05 \
    --lp_split_seed 42 \
    --lp_real_data_path "$REAL_DATA_PATH" \
    --knn_eval_freq 0 \
    --use_wandb false \
    --dist_url "env://"

# Usage:
#   DISTILLED_DATA=<PATH_TO_DISTILLED_PT> sbatch slurm/run_b_distilled_dev.slurm
