# yolo11n_640_combined_20260921_v1

**Author:** Kimi  **Date:** 2026-09-21

## What it is

| | |
|---|---|
| architecture | YOLO11n (nano), 2,590,620 params |
| task | object detection |
| input | 640 × 640 |
| classes | `0: flag`, `1: gate`, `2: flare`, `3: bucket` |
| size | 5.5 MB |
| trained | 150 epochs, AdamW, NVIDIA H100 |
| framework | Ultralytics 8.4.157, torch 2.14 |

## Data

Kaggle `kushagrajaveri/mantaclaus-sauvc-2026-yolo-dataset`, both bundled sets
**trained together** (6445 train images):

- `yolo_dataset_v2` — 5547 imgs, clearer water (mean RGB 90/141/152)
- `finetune_dataset` — 898 imgs, turbid water (mean RGB 53/184/225)

Training on the union matters. Training on the clean set alone lost 18 mAP50
points on turbid water; fine-tuning sequentially afterwards fixed that but
dropped flare recall on clear water from 0.82 to 0.46. The combined model is
the only one that handles both.

## Metrics

**Measured on: `finetune_dataset` valid — 249 images, TURBID water.**

Closest available proxy for competition conditions:

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| gate | 1.000 | 0.993 | 0.995 | 0.906 |
| flag | 0.930 | 0.983 | 0.981 | 0.737 |
| bucket | 0.966 | 0.873 | 0.911 | 0.679 |
| flare | 0.916 | 0.887 | 0.946 | 0.636 |
| **all** | 0.953 | 0.934 | 0.958 | 0.739 |

On `yolo_dataset_v2` valid (1253 imgs, clear water): mAP50 0.9631, mAP50-95 0.7269.

Both validation sets are small — the turbid one is only 249 images — so treat
the third decimal as noise. The clear-vs-turbid gap is not noise.

## Known limitations

- **`bucket` recall is 0.873** — misses ~1 in 8. Buckets sit on the pool floor
  and blend into it. This is the one that costs a task; consider lowering
  `conf` since a false bucket is far cheaper than a missed one.
- **`flare` boxes are loose** (mAP50 0.946 but mAP50-95 0.636). It finds flares
  reliably but cannot pin their edges — they are thin and vertical, so small
  horizontal errors wreck IoU. Fine for bearing, unreliable for estimating
  distance from apparent width.
- **Evaluated only on the Kaggle data.** Real performance depends on how well
  that matches our camera and housing. Validate on our own footage.

## Deploying to the Jetson Orin Nano


Build the TensorRT engine **on the board** — engines are tied to the exact GPU
and TensorRT version that created them.

```bash
pip install ultralytics
yolo export model=model.pt format=engine half=True imgsz=640
yolo predict model=model.engine source=0 conf=0.25
```

Do not commit the `.engine` — it is device-specific and rebuilt in a minute.

## Improving it

Roughly in order of value:

1. **Our own footage.** Label a few hundred frames through the real camera and
   housing, then fine-tune. Beats everything else here.
2. **Higher resolution** (imgsz 960) — the direct fix for flare. Orin can afford it.
3. **Bigger model** (`yolo11s`, 9.4M params) — Orin has headroom.
4. **Threshold tuning** — free, no retraining.

## Reproduce

```bash
sbatch training/run_train.sbatch --data datasets/sauvc_combined.yaml
```

Trained in this repo — see git history for `training/` at the time.
`args.yaml` holds every hyperparameter, `metrics/environment.json` the versions.

**Note:** this run used ultralytics auto-batch, which chose 174 on a 95GB H100.
`yolo11n_640_clean_20260921_v1` used `batch=16` and scored better on clear water;
the smaller batch is likely the reason and should be tried here.
