# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

try:
    from sklearn.metrics import f1_score, confusion_matrix as sk_confusion_matrix
    from sklearn.manifold import TSNE
    from sklearn.linear_model import LogisticRegression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


from torch.utils.data import Dataset

class TransformedTensorDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]  # (C, H, W) tensor
        if self.transform is not None:
            img = transforms.functional.to_pil_image(img)
            img = self.transform(img)
        return img, self.labels[idx]

class RepeatDataset(Dataset):
    def __init__(self, dataset, repeat=100):
        self.dataset = dataset
        self.repeat = repeat

    def __len__(self):
        return len(self.dataset) * self.repeat

    def __getitem__(self, idx):
        return self.dataset[idx % len(self.dataset)]

torchvision_archs = sorted(name for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision_models.__dict__[name]))

def get_args_parser():
    parser = argparse.ArgumentParser('DINO', add_help=False)

    # Model parameters
    parser.add_argument('--arch', default='vit_small', type=str,
        choices=['vit_tiny', 'vit_small', 'vit_base', 'xcit', 'deit_tiny', 'deit_small'] \
                + torchvision_archs,
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""")
    parser.add_argument('--patch_size', default=16, type=int, help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""")
    parser.add_argument('--out_dim', default=65536, type=int, help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""")
    parser.add_argument('--norm_last_layer', default=True, type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""")
    parser.add_argument('--momentum_teacher', default=0.996, type=float, help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""")
    parser.add_argument('--use_bn_in_head', default=False, type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)")

    # Temperature teacher parameters
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""")
    parser.add_argument('--teacher_temp', default=0.04, type=float, help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=0, type=int,
        help='Number of warmup epochs for the teacher temperature (Default: 30).')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=0.04, help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""")
    parser.add_argument('--weight_decay_end', type=float, default=0.4, help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""")
    parser.add_argument('--clip_grad', type=float, default=3.0, help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""")
    parser.add_argument('--batch_size_per_gpu', default=64, type=int,
        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=100, type=int, help='Number of epochs of training.')
    parser.add_argument('--freeze_last_layer', default=1, type=int, help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""")
    parser.add_argument("--lr", default=0.0005, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")
    parser.add_argument("--warmup_epochs", default=10, type=int,
        help="Number of epochs for the linear learning-rate warm up.")
    parser.add_argument('--min_lr', type=float, default=1e-6, help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""")
    parser.add_argument('--optimizer', default='adamw', type=str,
        choices=['adamw', 'sgd', 'lars'], help="""Type of optimizer. We recommend using adamw with ViTs.""")
    parser.add_argument('--drop_path_rate', type=float, default=0.1, help="stochastic depth rate")

    # Multi-crop parameters
    parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
    parser.add_argument('--local_crops_number', type=int, default=8, help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
    parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""")

    # Misc
    parser.add_argument('--data_path', default='/path/to/imagenet/train/', type=str,
        help='Please specify path to the ImageNet training data.')
    parser.add_argument('--output_dir', default=".", type=str, help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=20, type=int, help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=0, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument('--distilled_data_path', default=None, type=str,
    help='Path to distilled dataset .pt file (with "images" and "labels" keys). '
         'If set, overrides --data_path.')
    
    parser.add_argument('--num_dataset_repeats', default=10, type=int,
        help='Number of times to repeat the dataset.')

    # W&B parameters
    parser.add_argument('--use_wandb', type=utils.bool_flag, default=False,
        help='Enable Weights & Biases logging.')
    parser.add_argument('--wandb_project', default='dino-aqua20', type=str,
        help='W&B project name.')
    parser.add_argument('--wandb_entity', default=None, type=str,
        help='W&B entity (username or team). None uses the default entity.')
    parser.add_argument('--wandb_run_name', default=None, type=str,
        help='W&B run name. None lets W&B auto-generate.')

    # kNN evaluation parameters
    parser.add_argument('--knn_eval_freq', default=10, type=int,
        help='Run kNN evaluation every N epochs. 0 disables kNN eval.')
    parser.add_argument('--knn_nb_knn', default=[10, 20, 100, 200], nargs='+', type=int,
        help='Values of k for kNN classification. Each k is capped at num_train_samples-1.')
    parser.add_argument('--knn_temperature', default=0.07, type=float,
        help='Temperature for kNN voting coefficients.')
    parser.add_argument('--knn_test_data_path', default=None, type=str,
        help='Path to real test set (ImageFolder layout) for kNN queries. '
             'Memory bank comes from --distilled_data_path. Required for proper eval.')
    parser.add_argument('--num_classes', default=20, type=int,
        help='Number of classes in the dataset (20 for AQUA20).')

    # Linear probe evaluation parameters
    parser.add_argument('--lp_eval_freq', default=10, type=int,
        help='Run linear probe evaluation every N epochs. 0 disables LP eval.')
    parser.add_argument('--lp_train_frac', default=0.05, type=float,
        help='Fraction of real training data for LP (stratified, floor=min(5,n_class)).')
    parser.add_argument('--lp_split_seed', default=42, type=int,
        help='Random seed for stratified LP train subset.')
    parser.add_argument('--lp_real_data_path', default=None, type=str,
        help='ImageFolder root for real AQUA20 data used in LP eval. '
             'Falls back to --data_path when not set (Run A). '
             'Required when --distilled_data_path is set (Run B).')

    return parser


def _build_lp_subset(dataset, fraction, seed, min_per_class=5):
    """Stratified subset with floor guarantee: at least min(min_per_class, n_class) per class.
    Returns (Subset, per_class_counts dict)."""
    from collections import defaultdict
    import random as _rnd
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


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ optional wandb init ... ============
    if args.use_wandb and utils.is_main_process():
        if not _WANDB_AVAILABLE:
            print("WARNING: wandb not installed, disabling wandb logging.")
            args.use_wandb = False
        elif os.environ.get('WANDB_DISABLED', '').lower() in ('true', '1', 'yes'):
            print("WARNING: WANDB_DISABLED is set, disabling wandb logging.")
            args.use_wandb = False
        else:
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_run_name,
                config=vars(args),
            )
            # Define custom x-axes so iter/* and epoch/* can be logged independently
            # without step monotonicity conflicts.
            wandb.define_metric("global_step")
            wandb.define_metric("iter/*", step_metric="global_step")
            wandb.define_metric("epoch_num")
            wandb.define_metric("epoch/*", step_metric="epoch_num")
            wandb.define_metric("knn/*", step_metric="epoch_num")
            print(f"wandb run: {wandb.run.name} ({wandb.run.url})")

    # ============ preparing data ... ============
    transform = DataAugmentationDINO(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )
    if args.distilled_data_path is not None:
        distilled_data = torch.load(args.distilled_data_path)
        dataset = TransformedTensorDataset(
            distilled_data["images"],
            distilled_data["labels"].long(),
            transform=transform,
        )
        dataset = RepeatDataset(dataset, repeat=args.num_dataset_repeats)
    else:
        dataset = datasets.ImageFolder(args.data_path, transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load('facebookresearch/xcit:main', args.arch,
                                 pretrained=False, drop_path_rate=args.drop_path_rate)
        teacher = torch.hub.load('facebookresearch/xcit:main', args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    # multi-crop wrapper handles forward with inputs of different resolutions
    student = utils.MultiCropWrapper(student, DINOHead(
        embed_dim,
        args.out_dim,
        use_bn=args.use_bn_in_head,
        norm_last_layer=args.norm_last_layer,
    ))
    teacher = utils.MultiCropWrapper(
        teacher,
        DINOHead(embed_dim, args.out_dim, args.use_bn_in_head),
    )
    # move networks to gpu
    student, teacher = student.cuda(), teacher.cuda()
    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict())
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = DINOLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(student)
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.,  # linear scaling rule
        args.min_lr,
        args.epochs, len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs, len(data_loader),
    )
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1,
                                               args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting DINO training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of DINO ... ============
        train_stats = train_one_epoch(student, teacher, teacher_without_ddp, dino_loss,
            data_loader, optimizer, lr_schedule, wd_schedule, momentum_schedule,
            epoch, fp16_scaler, args)

        # ============ writing logs ... ============
        save_dict = {
            'student': student.state_dict(),
            'teacher': teacher.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'dino_loss': dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, 'checkpoint.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

        # ============ wandb per-epoch logging ... ============
        if args.use_wandb and utils.is_main_process():
            wandb.log(
                {**{f'epoch/{k}': v for k, v in train_stats.items()},
                 'epoch_num': epoch},
            )

        # ============ periodic kNN evaluation ... ============
        if (args.knn_eval_freq > 0
                and (epoch + 1) % args.knn_eval_freq == 0
                and utils.is_main_process()):
            knn_metrics = run_knn_eval(teacher_without_ddp, args, epoch)
            if knn_metrics and args.use_wandb:
                log_knn = {'epoch_num': epoch}
                for k, v in knn_metrics.items():
                    log_knn[f'knn/{k}'] = v
                wandb.log(log_knn)

        # ============ periodic linear probe evaluation ... ============
        if (args.lp_eval_freq > 0
                and (epoch + 1) % args.lp_eval_freq == 0
                and utils.is_main_process()):
            lp_metrics = run_lp_eval(teacher_without_ddp, args, epoch)
            if lp_metrics:
                lp_log_stats = {**{f'lp_{k}': v for k, v in lp_metrics.items()}, 'epoch': epoch}
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(lp_log_stats) + "\n")
                if args.use_wandb:
                    wandb.log({'epoch_num': epoch, **{f'lp/{k}': v for k, v in lp_metrics.items()}})

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    if args.use_wandb and utils.is_main_process():
        wandb.finish()


@torch.no_grad()
@torch.no_grad()
def _extract_features(backbone_model, data_loader):
    """Extract and L2-normalize backbone features for all images in data_loader."""
    all_feats, all_labels = [], []
    for imgs, lbls in data_loader:
        imgs = imgs.cuda(non_blocking=True)
        feats = backbone_model.backbone(imgs)
        if isinstance(feats, tuple):
            feats = feats[0]
        all_feats.append(F.normalize(feats, dim=1, p=2).cpu())
        all_labels.append(lbls)
    return torch.cat(all_feats), torch.cat(all_labels)


@torch.no_grad()
def run_knn_eval(backbone_model, args, epoch):
    """
    kNN evaluation.

    Memory bank:
      - distilled mode: --distilled_data_path (.pth, 1 image/class)
      - real-data mode: --data_path (ImageFolder train set)
    Queries: --knn_test_data_path (ImageFolder test set)
    """
    if not args.knn_test_data_path:
        print("WARNING: --knn_test_data_path not set; skipping kNN eval.")
        return {}
    if not args.distilled_data_path and not args.data_path:
        return {}

    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    backbone_model.eval()

    # ── Memory bank ───────────────────────────────────────────────────────────
    if args.distilled_data_path:
        distilled_data = torch.load(args.distilled_data_path, map_location='cpu')
        raw_images = distilled_data['images']   # (N_train, C, H, W)
        train_labels = distilled_data['labels'].long()
        N_train = len(train_labels)
        train_imgs = torch.stack([
            eval_transform(transforms.functional.to_pil_image(raw_images[i]))
            for i in range(N_train)
        ]).cuda()
        train_feats = backbone_model.backbone(train_imgs)
        if isinstance(train_feats, tuple):
            train_feats = train_feats[0]
        train_feats = F.normalize(train_feats, dim=1, p=2)  # (N_train, D)
    else:
        train_dataset = datasets.ImageFolder(args.data_path, transform=eval_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )
        train_feats, train_labels = _extract_features(backbone_model, train_loader)
        train_feats = train_feats.cuda()
        N_train = len(train_labels)

    # ── Query set: real test images ───────────────────────────────────────────
    test_dataset = datasets.ImageFolder(args.knn_test_data_path, transform=eval_transform)
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    test_feats_cpu, test_labels_cpu = _extract_features(backbone_model, test_loader)
    test_feats = test_feats_cpu.cuda()
    test_labels = test_labels_cpu.cuda()
    N_test = len(test_labels)

    # ── kNN classification ────────────────────────────────────────────────────
    # sim[i, j] = cosine similarity between test image i and train image j
    sim = torch.mm(test_feats, train_feats.t())  # (N_test, N_train)

    num_classes = args.num_classes
    metrics = {}
    best_preds = None  # store preds for smallest k (for plots)

    for k in sorted(args.knn_nb_knn):
        k_eff = min(k, N_train)  # cap at N_train=20
        distances, indices = sim.topk(k_eff, dim=1, largest=True, sorted=True)  # (N_test, k_eff)
        retrieved_neighbors = train_labels.cuda()[indices]  # (N_test, k_eff)

        ro = torch.zeros(N_test, k_eff, num_classes, device=train_feats.device)
        ro.scatter_(2, retrieved_neighbors.unsqueeze(2), 1)

        T = args.knn_temperature
        dt = distances.clone().div_(T).exp_()  # (N_test, k_eff)
        probs = (ro * dt.unsqueeze(2)).sum(dim=1)  # (N_test, num_classes)
        _, predictions = probs.sort(1, descending=True)

        correct = predictions.eq(test_labels.view(-1, 1))
        top1 = correct[:, 0].sum().item() * 100.0 / N_test
        top5 = correct[:, :5].any(dim=1).sum().item() * 100.0 / N_test

        preds_np = predictions[:, 0].cpu().numpy()
        labels_np = test_labels_cpu.numpy()
        f1_macro = float(f1_score(labels_np, preds_np, average='macro', zero_division=0) * 100)
        f1_weighted = float(f1_score(labels_np, preds_np, average='weighted', zero_division=0) * 100)

        metrics[f'{k}nn_top1'] = top1
        metrics[f'{k}nn_top5'] = top5
        metrics[f'{k}nn_f1_macro'] = f1_macro
        metrics[f'{k}nn_f1_weighted'] = f1_weighted

        print(f"Epoch {epoch} | {k_eff}-NN (k={k}, bank={N_train}, queries={N_test}): "
              f"top1={top1:.1f}%  top5={top5:.1f}%  "
              f"F1_macro={f1_macro:.1f}%  F1_weighted={f1_weighted:.1f}%")

        if best_preds is None:
            best_preds = preds_np  # smallest k

    if _SKLEARN_AVAILABLE and best_preds is not None:
        labels_np = test_labels_cpu.numpy()

        # Confusion matrix (query labels vs predicted)
        cm = sk_confusion_matrix(labels_np, best_preds, labels=list(range(num_classes)))
        class_names = test_dataset.classes  # alphabetical, matches integer labels
        fig_cm, ax_cm = plt.subplots(figsize=(10, 9))
        im = ax_cm.imshow(cm, interpolation='nearest', cmap='Blues')
        fig_cm.colorbar(im, ax=ax_cm)
        ax_cm.set_xticks(range(num_classes))
        ax_cm.set_yticks(range(num_classes))
        ax_cm.set_xticklabels(class_names, rotation=45, ha='right', fontsize=7)
        ax_cm.set_yticklabels(class_names, fontsize=7)
        ax_cm.set_title(f'kNN Confusion Matrix — epoch {epoch} (k={sorted(args.knn_nb_knn)[0]})')
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        fig_cm.tight_layout()
        metrics['confusion_matrix'] = wandb.Image(fig_cm) if args.use_wandb else None
        plt.close(fig_cm)

        # t-SNE of train (distilled) + test features combined, colored by class
        all_feats_np = torch.cat([train_feats.cpu(), test_feats_cpu], dim=0).numpy()
        all_labels_np = np.concatenate([train_labels.numpy(), labels_np])
        all_split = np.array(['train'] * N_train + ['test'] * N_test)
        perplexity = min(30, len(all_feats_np) - 1)
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
        coords = tsne.fit_transform(all_feats_np)
        fig_tsne, ax_tsne = plt.subplots(figsize=(9, 8))
        # test points (small), train points (large star markers)
        test_mask = all_split == 'test'
        train_mask = all_split == 'train'
        # Train d'abord, en gris léger pour donner le contexte
        ax_tsne.scatter(coords[train_mask, 0], coords[train_mask, 1],
                        c='lightgray', s=8, alpha=0.3, linewidths=0,
                        zorder=1, label='train')
        # Test par-dessus, bien visible
        sc = ax_tsne.scatter(coords[test_mask, 0], coords[test_mask, 1],
                            c=all_labels_np[test_mask], cmap='tab20',
                            s=40, alpha=0.9, edgecolors='k', linewidths=0.3,
                            zorder=2, label='test')
        fig_tsne.colorbar(sc, ax=ax_tsne, label='class')
        ax_tsne.legend()
        ax_tsne.set_title(f't-SNE: train (★) + test (·) features — epoch {epoch}')
        fig_tsne.tight_layout()
        metrics['tsne'] = wandb.Image(fig_tsne) if args.use_wandb else None
        plt.close(fig_tsne)

    backbone_model.train()
    return metrics


@torch.no_grad()
def run_lp_eval(backbone_model, args, epoch):
    """Periodic linear probe: stratified 5% train → sklearn LR → top-1/top-5 on real test set."""
    if not _SKLEARN_AVAILABLE:
        print("WARNING: sklearn not available, skipping LP eval.")
        return {}

    real_data_path = args.lp_real_data_path or args.data_path
    if args.distilled_data_path and not args.lp_real_data_path:
        print("WARNING: --distilled_data_path set but --lp_real_data_path not set; skipping LP eval.")
        return {}
    if not real_data_path or not os.path.isdir(real_data_path):
        print(f"WARNING: LP real data path '{real_data_path}' not found; skipping LP eval.")
        return {}

    eval_transform = transforms.Compose([
        transforms.Resize(256, interpolation=Image.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    backbone_model.eval()

    # ── Training subset (stratified, floor=min(5, n_class)) ───────────────────
    train_full = datasets.ImageFolder(os.path.join(real_data_path, "train"), transform=eval_transform)
    train_subset, per_class_counts = _build_lp_subset(train_full, args.lp_train_frac, args.lp_split_seed)

    # Save indices + per-class counts once (for audit)
    indices_path = Path(args.output_dir) / "lp_train_indices.json"
    if not indices_path.exists():
        with open(indices_path, "w") as f:
            json.dump({
                "indices": list(train_subset.indices),
                "per_class_counts": per_class_counts,
                "total": len(train_subset),
                "fraction": args.lp_train_frac,
                "seed": args.lp_split_seed,
            }, f, indent=2)
        print(f"LP subset saved: {len(train_subset)} imgs | per-class: {per_class_counts}")

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ── Test set ──────────────────────────────────────────────────────────────
    test_dir = os.path.join(real_data_path, "test")
    if not os.path.isdir(test_dir):
        test_dir = os.path.join(real_data_path, "val")
    test_set = datasets.ImageFolder(test_dir, transform=eval_transform)
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
    )

    # ── Feature extraction (reuse existing helper) ────────────────────────────
    train_feats, train_labels = _extract_features(backbone_model, train_loader)
    test_feats, test_labels = _extract_features(backbone_model, test_loader)

    X_tr = train_feats.numpy()
    y_tr = train_labels.numpy()
    X_te = test_feats.numpy()
    y_te = test_labels.numpy()

    # ── sklearn LogisticRegression ────────────────────────────────────────────
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=args.lp_split_seed, n_jobs=-1)
    clf.fit(X_tr, y_tr)

    train_acc = clf.score(X_tr, y_tr) * 100.0

    proba = clf.predict_proba(X_te)           # (N_test, num_classes)
    top5_idx = np.argsort(proba, axis=1)[:, -5:]
    test_acc_top1 = (proba.argmax(axis=1) == y_te).mean() * 100.0
    test_acc_top5 = np.mean([y_te[i] in top5_idx[i] for i in range(len(y_te))]) * 100.0

    print(f"Epoch {epoch} | LP ({len(train_subset)} train, {len(test_set)} test): "
          f"train={train_acc:.1f}%  top1={test_acc_top1:.1f}%  top5={test_acc_top5:.1f}%")

    backbone_model.train()
    return {"train_acc": train_acc, "test_acc_top1": test_acc_top1, "test_acc_top5": test_acc_top5}


def train_one_epoch(student, teacher, teacher_without_ddp, dino_loss, data_loader,
                    optimizer, lr_schedule, wd_schedule, momentum_schedule,epoch,
                    fp16_scaler, args):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for local_it, (images, _) in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + local_it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu
        images = [im.cuda(non_blocking=True) for im in images]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(images[:2])  # only the 2 global views pass through the teacher
            student_output = student(images)
            loss = dino_loss(student_output, teacher_output, epoch)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student,
                                              args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter
            for param_q, param_k in zip(student.module.parameters(), teacher_without_ddp.parameters()):
                param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        # per-iteration wandb logging (uses "global_step" as custom x-axis)
        if args.use_wandb and utils.is_main_process():
            # param_norms is a list of per-parameter L2 norms returned by clip_gradients
            grad_norm = float(max(param_norms)) if param_norms else 0.0
            wandb.log({
                'global_step': it,
                'iter/loss': loss.item(),
                'iter/lr': optimizer.param_groups[0]['lr'],
                'iter/wd': optimizer.param_groups[0]['weight_decay'],
                'iter/grad_norm': grad_norm,
                'iter/teacher_temp': float(dino_loss.teacher_temp_schedule[epoch]),
                'iter/momentum': float(m),
            })

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class DINOLoss(nn.Module):
    def __init__(self, out_dim, ncrops, warmup_teacher_temp, teacher_temp,
                 warmup_teacher_temp_epochs, nepochs, student_temp=0.1,
                 center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate((
            np.linspace(warmup_teacher_temp,
                        teacher_temp, warmup_teacher_temp_epochs),
            np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp
        ))

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = F.softmax((teacher_output - self.center) / temp, dim=-1)
        teacher_out = teacher_out.detach().chunk(2)

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if v == iq:
                    # we skip cases where student and teacher operate on the same view
                    continue
                loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(0.1),
            utils.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            utils.GaussianBlur(p=0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        for _ in range(self.local_crops_number):
            crops.append(self.local_transfo(image))
        return crops


if __name__ == '__main__':
    parser = argparse.ArgumentParser('DINO', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
