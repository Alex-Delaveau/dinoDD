# DECISIONS.md

Assumptions and decisions made during implementation that could not be inferred from the code or task description alone.

## Dataset path

**Assumed path**: `/home/alex/internship/GradientDistillation/logged_files/distillation/aqua20/dinov2_vitb/dinov2_vitb_distill_196_ipc1_augs3/data.pth`

Identified by searching for `.pt`/`.pth` files in the internship directory. The file name `dinov2_vitb_distill_196_ipc1_augs3` encodes:
- Teacher: DINOv2 ViT-B
- Resolution: 196px
- IPC=1 (images per class = 1)
- 3 augmentations per sample during distillation

This matches `cmd.sh` which already references this exact path.

## Image resolution

Distilled images are **196×196 px** (not 224). We resize to 224 in the eval transform (kNN feature extraction) using `Resize(224, BICUBIC) + CenterCrop(224)` to match ImageNet normalization expected by the ViT-S backbone.

During training, `DataAugmentationDINO` already handles arbitrary input resolution (RandomResizedCrop to 224 and 96).

## Train/val split for kNN evaluation

**Memory bank (train)**: The 20 distilled images from `--distilled_data_path` (1 per class, IPC=1).

**Query set (test)**: The real AQUA20 test images at `--knn_test_data_path` (`/home/alex/internship/datasets/aqua20/data/aqua20/test`), 1612 images across 20 classes — a proper held-out evaluation set.

**Class ordering**: `datasets.ImageFolder` sorts class directories alphabetically. The 20 alphabetical class names map to integer labels 0–19, which matches the distilled dataset's integer labels (both derived from the same AQUA20 class ordering).

**k capping**: All k values are capped at `min(k, N_train) = min(k, 20)` since the memory bank only has 20 images.

## Student arch choice

ViT-S/16 chosen over ViT-B/16. The distillation *teacher* was DINOv2 ViT-B (for feature extraction during dataset creation), but the *student* being trained here (DINOv1) has no architectural constraint. ViT-S is faster and less prone to overfitting on a 20-image dataset.

## RepeatDataset

`RepeatDataset(repeat=10)` was already present in the codebase (repeats 20 images → 200 effective per epoch). This is left unchanged.

## wandb run name format

Default: `dino-aqua20-{timestamp}` (set in the notebook). Can be overridden with `--wandb_run_name`.

## Feature extraction model for kNN

Using `teacher_without_ddp.backbone` (teacher, not student) for kNN feature extraction. The teacher's EMA-averaged features are more stable and are what the original DINO paper uses for downstream evaluation. The backbone has `fc` and `head` set to `nn.Identity()` by `MultiCropWrapper.__init__`.
