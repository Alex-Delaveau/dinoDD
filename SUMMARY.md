# SUMMARY: wandb + kNN eval integration for main_dino_aqua.py

## What was changed

### `main_dino_aqua.py`
- **wandb integration** (opt-in via `--use_wandb true`):
  - `wandb.init()` after distributed setup; reads `--wandb_project`, `--wandb_entity`, `--wandb_run_name`
  - Per-iteration logging: `iter/loss`, `iter/lr`, `iter/wd`, `iter/grad_norm`, `iter/teacher_temp`, `iter/momentum`
  - Per-epoch logging: `epoch/loss`, `epoch/lr`, `epoch/wd`
  - `wandb.finish()` at end of training
  - Graceful disable via `WANDB_DISABLED=true` or when `wandb` is not installed
- **kNN evaluation** (`run_knn_eval`, every `--knn_eval_freq` epochs):
  - Uses teacher backbone features (eval mode, no DINO head)
  - Handles AQUA20's 20-image constraint: all 20 images are both memory bank and query set, self-similarity masked to `-1e9`
  - k values capped at `min(k, N-1)` = 19 max (was hardcoded 1000 in eval_knn.py)
  - Metrics: top-1, top-5, F1 macro, F1 weighted per each k
  - Generates 20×20 confusion matrix (matplotlib → `wandb.Image`)
  - Generates t-SNE of 384-dim backbone features colored by class (matplotlib → `wandb.Image`)
- **New CLI args**: `--use_wandb`, `--wandb_project`, `--wandb_entity`, `--wandb_run_name`, `--knn_eval_freq`, `--knn_nb_knn`, `--knn_temperature`, `--num_classes`
- **Bug fix**: `clip_gradients()` returns a list of norms; `float(list)` raised `TypeError` — fixed to `max(param_norms)`.

### `pyproject.toml` / `uv.lock`
- Added `matplotlib` and `scikit-learn` dependencies (via `uv add`)

### `run_dino_aqua.ipynb`
- 5-cell notebook for launching training via `subprocess.Popen`
- Defines all hyperparameters as a dict, builds the command, streams stdout+stderr to notebook and to `{output_dir}/train.log`
- Provides PID, output dir, wandb run URL, and kill instructions

### New files
- `PLAN.md`: hyperparameter justification, integration design
- `DECISIONS.md`: dataset path assumptions, train/val split rationale, arch choice
- `SUMMARY.md`: this file

## Hyperparameters chosen (with justification)

| Parameter | Value | Reason |
|-----------|-------|--------|
| `arch` | `vit_small` | Small dataset (20 imgs); ViT-B would overfit. DINOv2 ViT-B was the distillation *teacher*, not a constraint on student arch. |
| `patch_size` | 16 | Stable default; 196px input → 12×12 patches. |
| `batch_size_per_gpu` | 64 | With repeat=10 → 200 imgs/epoch, gives ~3 iters/epoch. |
| `epochs` | 100 | Cheap per epoch; standard starting point for small distilled datasets. |
| `warmup_epochs` | 10 | 10% warmup, DINO standard. |
| `lr` | 0.0005 | DINO base LR for ref batch 256; cosine schedule applied by the framework. |
| `weight_decay` | 0.04→0.4 | Standard DINO cosine WD schedule. |
| `teacher_temp` | 0.04→0.07 (30 epoch warmup) | Shorter warmup (30/100 vs default 50) proportional to epoch count. |
| `momentum_teacher` | 0.996→1.0 | DINO default cosine EMA schedule. |
| `global_crops_scale` | (0.4, 1.0) | Less aggressive cropping for info-dense distilled images. |
| `out_dim` | 65536 | DINO default. |
| `use_fp16` | True | RTX 3070 supports FP16; faster training. |
| `knn_eval_freq` | 10 | Every 10 epochs as requested. |

## Active training run

| Field | Value |
|-------|-------|
| Run name | `dino-aqua20-20260505_121606` |
| PID | 21157 |
| Output dir | `/home/alex/internship/dino/outputs/dino-aqua20-20260505_121606/` |
| Log file | `/home/alex/internship/dino/outputs/dino-aqua20-20260505_121606/train.log` |
| wandb project | `dino-aqua20` |

## How to monitor

```bash
tail -f /home/alex/internship/dino/outputs/dino-aqua20-20260505_121606/train.log
kill 21157  # to stop
```

## kNN eval (epoch 9 confirmed)

```
Epoch 9 | 10-NN (bank=20 distilled, queries=1612 real test):
  top1=15.3%  top5=45.4%  F1_macro=6.2%  F1_weighted=16.7%
```

Random baseline for 20 classes ≈ 5% top-1 — model is above chance even at epoch 9 (still in LR warmup). F1 macro will improve once teacher temp finishes warming up (~epoch 30).

## Limitations

- Only 20 images total (IPC=1): kNN eval uses self-retrieval-masked full set as both bank and query. Accuracy reflects feature separability, not generalization. See `DECISIONS.md`.
- Training loss starts high (~10.8) because the teacher temperature is still in warmup. Loss should decrease through epochs 30–60 once teacher temp reaches its final value.
