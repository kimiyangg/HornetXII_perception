# SAUVC 2026 object detector — v1

`sauvc_yolo11n_640_combined_v1.pt`

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

## Performance

Turbid water (closest to competition conditions):

| class | P | R | mAP50 | mAP50-95 |
|---|---|---|---|---|
| gate | 1.000 | 0.993 | 0.995 | 0.906 |
| flag | 0.930 | 0.983 | 0.981 | 0.737 |
| bucket | 0.966 | 0.873 | 0.911 | 0.679 |
| flare | 0.916 | 0.887 | 0.946 | 0.636 |
| **all** | 0.953 | 0.934 | 0.958 | 0.739 |

Clear water: mAP50 0.963, mAP50-95 0.727.

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
yolo export model=sauvc_yolo11n_640_combined_v1.pt format=engine half=True imgsz=640
yolo predict model=sauvc_yolo11n_640_combined_v1.engine source=0 conf=0.25
```

Do not commit the `.engine` — it is device-specific and rebuilt in a minute.

## Improving it

Roughly in order of value:

1. **Our own footage.** Label a few hundred frames through the real camera and
   housing, then fine-tune. Beats everything else here.
2. **Higher resolution** (imgsz 960) — the direct fix for flare. Orin can afford it.
3. **Bigger model** (`yolo11s`, 9.4M params) — Orin has headroom.
4. **Threshold tuning** — free, no retraining.

## Reproducing

`args.yaml` holds every hyperparameter; `results.csv` the full metric history.
Training scripts are in `cluster/`.
