# yolo11n_640_clean_20260921_v1

**Author:** lamlamlam0  **Date:** 2026-09-21

## What it is

| | |
|---|---|
| file | `model.pt` (5.5 MB) |
| architecture | YOLO11n, 2.6M params |
| input | 640 x 640 |
| classes | `0: flag`, `1: gate`, `2: flare`, `3: bucket` |
| trained | best at epoch 81 of 100, ~82 min on a Tesla T4 |

## Training data

Kaggle `kushagrajaveri/mantaclaus-sauvc-2026-yolo-dataset`, **`yolo_dataset_v2` only**.

| split | images | flag | gate | flare | bucket |
|---|---|---|---|---|---|
| train | 5547 | 1030 | 1375 | 1510 | 3599 |
| valid | 1253 | 162 | 544 | 379 | 774 |

The bundled `finetune_dataset` (898 turbid-water images) was **not** used.

## Metrics

**Measured on: `yolo_dataset_v2` valid — 1253 images, CLEAR water only.**

| P | R | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.9393 | 0.9556 | 0.9676 | 0.7382 |

Best on clear water of any model here — see the comparison in
[`../CONVENTIONS.md`](../CONVENTIONS.md) for why that is not the whole story.

## Known limitations

- **Never evaluated on turbid water, and never trained on it.** Its validation
  set is entirely clear-water imagery, so it cannot show a turbid weakness even
  if one exists. A comparable clean-only model scored **0.536** mAP50-95 on the
  turbid set versus **0.739** for one trained on both, so a large drop in
  competition conditions is likely but has not been measured for this model.
- Per-class breakdown not recorded.

## Reproduce

Trained on Kaggle, outside this repo, so the notebook travels with the model:
[`train.ipynb`](train.ipynb).

Key settings: `epochs=100 batch=16 imgsz=640 optimizer=AdamW lr0=0.001
patience=25 seed=42 cos_lr=false erasing=0.4 degrees=0 flipud=0`

Environment: `metrics/environment.json` (T4, torch 2.10.0+cu128, ultralytics 8.4.157).

**Note:** `batch=16` is likely why this outperforms the combined model on clear
water -- that run used ultralytics auto-batch, which selected 174 on an H100.
Large batches generalise worse. Worth carrying over.
