"""Standalone sklearn linear probe — replicates exactly run_lp_eval() from main_dino_aqua.py.

Key choices that must stay in sync with run_lp_eval():
  - _build_lp_subset: stratified sampling with min_per_class=5 floor
  - L2-normalised features
  - LogisticRegression(max_iter=1000, C=1.0)
  - eval-only transform (no augmentation) for both train and test splits
"""
import argparse
import os
import json
import random as _rnd
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

try:
    from sklearn.linear_model import LogisticRegression
except ImportError:
    raise SystemExit("sklearn is required: pip install scikit-learn")

import utils


def _build_lp_subset(dataset, fraction, seed, min_per_class=5):
    """Copy of _build_lp_subset from main_dino_aqua.py — must stay identical."""
    rng = _rnd.Random(seed)
    targets = [s[1] for s in dataset.samples]
    class_to_indices = defaultdict(list)
    for idx, label in enumerate(targets):
        class_to_indices[label].append(idx)

    indices, per_class_counts = [], {}
    for cls, idx_list in sorted(class_to_indices.items()):
        n = len(idx_list)
        k = max(min(min_per_class, n), round(fraction * n))
        sampled = rng.sample(idx_list, k)
        indices.extend(sampled)
        per_class_counts[cls] = k

    return torch.utils.data.Subset(dataset, sorted(indices)), per_class_counts


@torch.no_grad()
def _extract_features(model, data_loader):
    """Extract L2-normalised features from a raw backbone (fc already removed)."""
    all_feats, all_labels = [], []
    for imgs, lbls in data_loader:
        imgs = imgs.cuda(non_blocking=True)
        feats = model(imgs)
        all_feats.append(F.normalize(feats, dim=1, p=2).cpu())
        all_labels.append(lbls)
    return torch.cat(all_feats), torch.cat(all_labels)


def main(args):
    cudnn.benchmark = True

    # ── Build backbone ────────────────────────────────────────────────────────
    model = torchvision_models.__dict__[args.arch]()
    model.fc = nn.Identity()
    model.cuda().eval()

    if args.pretrained_weights:
        utils.load_pretrained_weights(
            model, args.pretrained_weights, args.checkpoint_key, args.arch, args.patch_size
        )
        print(f"Loaded pretrained weights from {args.pretrained_weights}")
    else:
        print("No pretrained weights — random backbone (baseline)")

    # ── Transforms (identical to run_lp_eval) ────────────────────────────────
    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    # ── Training subset ───────────────────────────────────────────────────────
    train_full = datasets.ImageFolder(
        os.path.join(args.data_path, "train"), transform=eval_transform
    )
    train_subset, per_class_counts = _build_lp_subset(
        train_full, args.lp_train_frac, args.lp_split_seed
    )
    print(f"Train subset: {len(train_subset)} imgs | per-class: {per_class_counts}")

    # ── Test set ──────────────────────────────────────────────────────────────
    test_dir = os.path.join(args.data_path, "test")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(args.data_path, "val")
    test_set = datasets.ImageFolder(test_dir, transform=eval_transform)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.num_workers,
        pin_memory=True, drop_last=False,
    )

    # ── Feature extraction ────────────────────────────────────────────────────
    print("Extracting train features...")
    train_feats, train_labels = _extract_features(model, train_loader)
    print("Extracting test features...")
    test_feats, test_labels = _extract_features(model, test_loader)

    X_tr, y_tr = train_feats.numpy(), train_labels.numpy()
    X_te, y_te = test_feats.numpy(), test_labels.numpy()

    # ── sklearn LogisticRegression (identical params to run_lp_eval) ──────────
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.lp_split_seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    train_acc = clf.score(X_tr, y_tr) * 100.0

    proba = clf.predict_proba(X_te)
    top5_idx = np.argsort(proba, axis=1)[:, -5:]
    test_acc_top1 = (proba.argmax(axis=1) == y_te).mean() * 100.0
    test_acc_top5 = np.mean([y_te[i] in top5_idx[i] for i in range(len(y_te))]) * 100.0

    print(f"LP results | train={train_acc:.1f}%  top1={test_acc_top1:.1f}%  top5={test_acc_top5:.1f}%")

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        result = {
            "arch": args.arch,
            "pretrained_weights": args.pretrained_weights,
            "lp_train_frac": args.lp_train_frac,
            "lp_split_seed": args.lp_split_seed,
            "n_train": len(train_subset),
            "n_test": len(test_set),
            "train_acc": train_acc,
            "test_acc_top1": test_acc_top1,
            "test_acc_top5": test_acc_top5,
        }
        with open(Path(args.output_dir) / "lp_sklearn_results.json", "w") as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to {args.output_dir}/lp_sklearn_results.json")


def get_args_parser():
    parser = argparse.ArgumentParser("Sklearn linear probe on frozen backbone")
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--patch_size", default=16, type=int)
    parser.add_argument("--pretrained_weights", default="", type=str,
                        help="Path to DINO checkpoint. Leave empty for random backbone baseline.")
    parser.add_argument("--checkpoint_key", default="teacher", type=str)
    parser.add_argument("--data_path", required=True, type=str)
    parser.add_argument("--lp_train_frac", default=0.05, type=float,
                        help="Fraction of training data (same default as run_lp_eval).")
    parser.add_argument("--lp_split_seed", default=42, type=int)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument("--output_dir", default="", type=str)
    return parser


if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
