#!/usr/bin/env python3
"""Train the SAUVC object detector.

Runs anywhere with a GPU -- the cluster is just one place to run it. On the
NUS SoC cluster it must go through Slurm (see run_train.sbatch): the login
node caps virtual memory at 1GB, under which torch cannot even import.

    python train.py                                  # defaults
    python train.py --epochs 300 --imgsz 960         # override
    python train.py --stage finetune --weights runs/x/weights/best.pt
    python train.py --resume runs/x/weights/last.pt  # continue an interrupted run
"""
from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path.home() / "sauvc"


# Augmentation for underwater footage. These are deliberately wider than the
# Ultralytics defaults: the two source datasets differ enormously in colour
# cast (mean RGB 90/141/152 clear vs 53/184/225 turbid), so the model has to
# span that range. hsv_s/hsv_v do most of the work.
AUG_SCRATCH = dict(
    hsv_h=0.02,       # hue: small. Colour *is* informative underwater.
    hsv_s=0.8,        # saturation: wide, spans clear <-> turbid
    hsv_v=0.5,        # brightness: wide, depth and lighting vary
    degrees=10,       # the vehicle rolls
    translate=0.1,
    scale=0.5,        # objects appear at very different distances
    fliplr=0.5,
    flipud=0.1,       # small: the pool floor is not usually above us
    mosaic=1.0,
    close_mosaic=15,  # disable mosaic for the last 15 epochs to settle on real images
    erasing=0.2,
)

# Fine-tuning adapts an already-competent model to a new domain, so everything
# is gentler -- aggressive augmentation here would undo what stage 1 learned.
AUG_FINETUNE = dict(
    hsv_h=0.015, hsv_s=0.5, hsv_v=0.3,
    degrees=5, translate=0.1, scale=0.3,
    fliplr=0.5, flipud=0.05,
    mosaic=0.5, close_mosaic=10,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", choices=["scratch", "finetune"], default="scratch",
                   help="scratch: train from COCO weights. finetune: adapt an existing model.")
    p.add_argument("--model", default="yolo11n.pt",
                   help="starting weights. yolo11n/s/m/l/x, or a path to a .pt")
    p.add_argument("--weights", type=Path,
                   help="checkpoint to fine-tune from (required for --stage finetune)")
    p.add_argument("--resume", type=Path,
                   help="continue an interrupted run from its last.pt")
    p.add_argument("--data", type=Path,
                   default=PROJECT_ROOT / "data" / "sauvc_combined.yaml")
    p.add_argument("--imgsz", type=int, default=640,
                   help="640 is the baseline; 960 localises the thin 'flare' class better")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=-1, help="-1 lets ultralytics auto-size")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--name", default=None, help="run name (default: derived from stage)")
    p.add_argument("--project", type=Path, default=PROJECT_ROOT / "runs")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    from ultralytics import YOLO
    import torch

    if not torch.cuda.is_available():
        # Worth failing loudly. On this cluster it usually means the job landed
        # on a MIG-enabled a100-40 node, where torch sees the card but cannot
        # attach to it. Request h100-96 or nv instead.
        print("ERROR: no CUDA device. On SoC, avoid --gres=gpu:a100-40 (MIG).")
        return 1
    print(f"device: {torch.cuda.get_device_name(0)}")

    # Resuming reads every hyperparameter back from the run's args.yaml, so
    # passing them again here would be ignored at best and confusing at worst.
    if args.resume:
        print(f"resuming from {args.resume}")
        YOLO(str(args.resume)).train(resume=True)
        return 0

    if args.stage == "finetune":
        if not args.weights or not args.weights.exists():
            print("ERROR: --stage finetune needs --weights pointing at a .pt")
            return 1
        start, aug = str(args.weights), AUG_FINETUNE
        extra = dict(lr0=0.001, lrf=0.01, warmup_epochs=1, patience=20)
    else:
        start, aug = args.model, AUG_SCRATCH
        extra = dict(patience=50)

    name = args.name or f"sauvc-{args.stage}-{args.imgsz}"
    print(f"start={start}  data={args.data}  imgsz={args.imgsz}  epochs={args.epochs}")

    YOLO(start).train(
        data=str(args.data),
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        workers=args.workers,
        device=0,
        project=str(args.project),
        name=name,
        seed=0,
        cos_lr=True,
        **extra,
        **aug,
    )
    print(f"done -> {args.project / name / 'weights' / 'best.pt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
