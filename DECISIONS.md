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

**Problem**: With only 1 image per class (IPC=1, 20 total images), an 80/20 stratified split would leave 0.2 images per class per split — impossible.

**Decision**: Use all 20 images as **both** the memory bank and the query set.

**Self-retrieval mitigation**: The diagonal of the similarity matrix is masked to `-1e9` before top-k selection. This means each image's nearest neighbor is drawn from the other 19 images.

**Interpretation caveat**: Because the memory bank and query set are identical (minus self-retrieval), high kNN accuracy does not necessarily imply good generalization. The metric's primary use here is to track whether features are *separable* (i.e., whether training is progressing usefully), not to measure generalization.

**k capping**: All k values in `--knn_nb_knn` are capped at `N - 1 = 19` since there are only 19 non-self neighbors available.

## Student arch choice

ViT-S/16 chosen over ViT-B/16. The distillation *teacher* was DINOv2 ViT-B (for feature extraction during dataset creation), but the *student* being trained here (DINOv1) has no architectural constraint. ViT-S is faster and less prone to overfitting on a 20-image dataset.

## RepeatDataset

`RepeatDataset(repeat=10)` was already present in the codebase (repeats 20 images → 200 effective per epoch). This is left unchanged.

## wandb run name format

Default: `dino-aqua20-{timestamp}` (set in the notebook). Can be overridden with `--wandb_run_name`.

## Feature extraction model for kNN

Using `teacher_without_ddp.backbone` (teacher, not student) for kNN feature extraction. The teacher's EMA-averaged features are more stable and are what the original DINO paper uses for downstream evaluation. The backbone has `fc` and `head` set to `nn.Identity()` by `MultiCropWrapper.__init__`.
