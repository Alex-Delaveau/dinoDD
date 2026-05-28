#!/bin/bash
#SBATCH --job-name=dino-run-d-no-pretrain
#SBATCH --account=rbw@v100
#SBATCH --constraint=v100-32g
#SBATCH --hint=nomultithread
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=00:30:00
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
# Random ResNet18 backbone — no pretrained weights.
# Uses the exact same sklearn LP protocol as run_lp_eval() in main_dino_aqua.py:
#   LogisticRegression(max_iter=1000, C=1.0), 5% stratified train data, L2-normalised features.
python $CODE_DIR/eval_lp_sklearn.py \
    --arch resnet18 \
    --data_path "$DATA_PATH" \
    --lp_train_frac 0.05 \
    --lp_split_seed 42 \
    --batch_size 128 \
    --num_workers 8 \
    --output_dir "$OUTPUT_DIR"

# Usage:
#   sbatch slurm/run_d_no_pretrain_baseline.slurm
#   DATA_PATH=/custom/path sbatch slurm/run_d_no_pretrain_baseline.slurm
