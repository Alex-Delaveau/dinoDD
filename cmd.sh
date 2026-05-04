#!/bin/bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

uv run -m torch.distributed.launch --nproc_per_node=1 main_dino_aqua.py \
    --arch vit_small \
    --distilled_data_path /home/alex/internship/GradientDistillation/logged_files/distillation/aqua20/dinov2_vitb/dinov2_vitb_distill_196_ipc1_augs3/data.pth \
    --output_dir ./outputs/dino_aqua20_distilled \
    --epochs 100 \
    --batch_size_per_gpu 50 \
    --num_workers 4 \
    --dist_url file:///tmp/dino_dist_$(date +%s)