# PLAN: wandb + kNN eval for main_dino_aqua.py

## Dataset Context

- **Distilled dataset**: 20 images total, 1 per class (IPC=1), 196×196 px, float32 [0,1]
- **Labels**: 0–19 (AQUA20 = 20 aquatic species classes)
- **Effective dataset size**: `RepeatDataset(repeat=10)` → 200 images/epoch
- **Teacher used for distillation**: DINOv2 ViT-B (feature extractor)
- **Student being trained**: DINOv1 (this repo)

## Hyperparameter Choices & Justification

| Parameter | Value | Justification |
|-----------|-------|---------------|
| `arch` | `vit_small` | Faster than ViT-B; distilled dataset is tiny (20 imgs), so ViT-B would overfit. DINOv2 ViT-B was the distillation *teacher* — no constraint on student arch. |
| `patch_size` | 16 | Images are 196×196 → 196/16 = 12.25 (rounded to 12), giving 144 tokens. Still fine for ViT-S. Patch 14 gives 14×14=196 tokens (cleaner), but 16 is more stable and the standard. |
| `batch_size_per_gpu` | 64 | With 200 effective images/epoch, batch 64 gives ~3 iters/epoch. Enough gradient signal with RepeatDataset. (cmd.sh uses 50; we slightly increase for efficiency.) |
| `epochs` | 100 | Distilled dataset is tiny; 100 epochs × 3 iters is modest. Can extend to 200 if loss hasn't converged. |
| `warmup_epochs` | 10 | 10% warmup is standard. |
| `lr` | 0.0005 | DINO default for reference batch 256. With effective batch=64, linear scaling gives 0.0005×64/256=0.000125 effective LR. Kept as the "base" per DINO convention (scheduler applies scaling). |
| `weight_decay` | 0.04 → 0.4 | Standard DINO schedule. |
| `teacher_temp` | 0.04 → 0.07 | warmup over 30 epochs (shorter than default 50 since only 100 total epochs). |
| `warmup_teacher_temp_epochs` | 30 | 30% of 100 epochs. |
| `student_temp` | 0.1 | DINO default. |
| `momentum_teacher` | 0.996 → 1.0 | DINO default cosine schedule. |
| `global_crops_scale` | (0.4, 1.0) | Distilled images are info-dense (distilled by DINOv2 teacher); less aggressive crop preserves more signal. |
| `local_crops_scale` | (0.05, 0.4) | Standard. |
| `local_crops_number` | 8 | DINO default. |
| `out_dim` | 65536 | DINO default. Works well for any dataset size. |
| `norm_last_layer` | True | Recommended for ViT-S (stabilizes training). |
| `use_fp16` | True | RTX 3070 has good FP16 support; speeds up training. |
| `num_workers` | 4 | Small dataset, low I/O bound; 4 is sufficient. |
| `knn_eval_freq` | 10 | Every 10 epochs as requested. |
| `saveckp_freq` | 20 | Save every 20 epochs. |
| `drop_path_rate` | 0.1 | Default. |
| `freeze_last_layer` | 1 | Standard for stability. |

## wandb Integration Points

1. **Initialization**: After `utils.init_distributed_mode(args)`, if `args.use_wandb` and `is_main_process()`: call `wandb.init(project, entity, name, config=vars(args))`.

2. **Per-iteration logging** (inside `train_one_epoch`):
   - `iter/loss`, `iter/lr`, `iter/wd`, `iter/grad_norm`
   - `iter/teacher_temp`, `iter/momentum` (from `dino_loss.teacher_temp_schedule[epoch]` and `m`)
   - Logged at every iteration with `step=global_step`

3. **Per-epoch logging** (in `train_dino` main loop):
   - All keys from `train_stats` (loss, lr, wd)
   - `epoch/epoch`
   - Logged at `step=epoch`

4. **kNN eval results** (every `knn_eval_freq` epochs):
   - `knn/{k}nn_top1`, `knn/{k}nn_top5`, `knn/{k}nn_f1_macro`, `knn/{k}nn_f1_weighted`
   - `knn/confusion_matrix` (wandb.Image of 20×20 confusion matrix plot)
   - `knn/tsne` (wandb.Image of t-SNE features colored by class)

5. **Finish**: `wandb.finish()` at end of training.

## kNN Evaluator Design

### AQUA20 Class Handling (Critical Fixes)
- `num_classes=1000` (ImageNet default) in `knn_classifier` → change to `args.num_classes` (default 20)
- Cap top-k at `min(k, num_train_samples - 1)` to avoid self-retrieval with a single memory bank
- `retrieval_one_hot.resize_(batch_size * k, num_classes)` uses `num_classes` correctly once fixed

### Data Split Strategy
With only 20 images (1 per class), 80/20 stratified split leaves 0 or 1 sample per class per split — not viable. Strategy:
- Use all 20 images as both **memory bank** (train) and **query set** (test)
- Exclude self-similarity by zeroing the diagonal of the similarity matrix
- Effectively: each image queries the other 19 images → picks 1 neighbor per class
- With k=10: uses the 10 most similar non-self images
- Document in DECISIONS.md

### Feature Extraction
- Use `teacher_without_ddp.backbone` in eval mode (teacher features are more stable)
- Apply standard eval transform: Resize(224) + CenterCrop(224) + Normalize(ImageNet stats)
- No RepeatDataset for eval — use raw 20 images

### Metrics
- top-1 accuracy, top-5 accuracy (capped at min(5, k))
- F1 macro (preferred by Alex over raw accuracy)
- F1 weighted
- 20×20 confusion matrix → matplotlib plot → wandb.Image
- t-SNE of 384-dim features (ViT-S embed_dim) → scatter plot colored by class → wandb.Image

## Commit Plan

1. `PLAN.md` (this file) — commit: `plan: add wandb + kNN eval plan`
2. `wandb integration` in `main_dino_aqua.py` — commit: `feat: add wandb logging to main_dino_aqua`
3. `kNN eval integration` in `main_dino_aqua.py` — commit: `feat: add periodic kNN eval to main_dino_aqua`
4. `plotting (confusion matrix, t-SNE)` — commit: `feat: add confusion matrix and t-SNE wandb plots`
5. `run_dino_aqua.ipynb` notebook — commit: `feat: add training launch notebook`
6. `SUMMARY.md` + `DECISIONS.md` — commit: `docs: add summary and decisions`

## File Changes

- **`main_dino_aqua.py`**: wandb args, wandb init/logging, kNN eval function, plotting
- **`run_dino_aqua.ipynb`**: new notebook for launching training
- **`PLAN.md`**: this file
- **`SUMMARY.md`**: post-implementation summary
- **`DECISIONS.md`**: documented assumptions and decisions
- **No changes** to: `eval_knn.py`, `utils.py`, `vision_transformer.py` (self-contained)
