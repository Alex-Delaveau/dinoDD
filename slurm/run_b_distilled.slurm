#!/bin/bash
#SBATCH --job-name=dino-run-b-distilled
#SBATCH --account=rbw@v100
#SBATCH --constraint=v100-32g
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=20:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
set -e

module purge
module load pytorch-gpu/py3/2.1.1

# ── Paths ─────────────────────────────────────────────────────────────────────
# DISTILLED_DATA: path to the distilled .pth file (dict with "images" and "labels" keys)
DISTILLED_DATA=${DISTILLED_DATA:?Must set DISTILLED_DATA (path to distilled .pth file)}
REAL_DATA_PATH=${REAL_DATA_PATH:-/lustre/fshomisc/home/rech/genlpd01/ucw75ke/work/datasets/aqua20/data/aqua20/}
OUTPUT_DIR=${OUTPUT_DIR:-$WORK/dino_runs/run_b_distilled_$SLURM_JOB_ID}
CODE_DIR=${CODE_DIR:-$WORK/projects/dinoDD}

mkdir -p "$OUTPUT_DIR"
mkdir -p logs

# ── Launch ───────────────────────────────────────────────────────────────────
# ResNet-18 on 20-image distilled dataset:
#   - num_dataset_repeats=50: each epoch = 20*50 = 1000 effective images
#   - batch_size_per_gpu=20: = dataset size, 1 batch per repetition
#   - epochs=1000: ~50k grad updates; resume from checkpoint if more needed
#   - optimizer=lars, use_fp16=false, use_bn_in_head=true: DINO defaults for ResNets
#   - lp_eval_freq=100: 10 LP evals over 1000 epochs
srun torchrun \
    --nproc_per_node=$SLURM_GPUS_ON_NODE \
    --nnodes=$SLURM_NNODES \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$HOSTNAME:29500 \
    $CODE_DIR/main_dino_aqua.py \
    --arch resnet18 \
    --distilled_data_path "$DISTILLED_DATA" \
    --num_dataset_repeats 50 \
    --epochs 1000 \
    --batch_size_per_gpu 20 \
    --optimizer lars \
    --use_fp16 false \
    --use_bn_in_head true \
    --num_workers 4 \
    --output_dir "$OUTPUT_DIR" \
    --saveckp_freq 100 \
    --lp_eval_freq 100 \
    --lp_train_frac 0.05 \
    --lp_split_seed 42 \
    --lp_real_data_path "$REAL_DATA_PATH" \
    --knn_eval_freq 0 \
    --use_wandb false \
    --dist_url "env://"

# Usage:
#   DISTILLED_DATA=<PATH_TO_DISTILLED_PT> sbatch slurm/run_b_distilled.slurm
#   DISTILLED_DATA=<PATH> REAL_DATA_PATH=/custom/aqua20 sbatch slurm/run_b_distilled.slurm
#
# To resume from checkpoint (auto-detected by main_dino_aqua.py):
#   OUTPUT_DIR=$WORK/dino_runs/run_b_distilled_<old_job_id> DISTILLED_DATA=<PATH> sbatch slurm/run_b_distilled.slurm
