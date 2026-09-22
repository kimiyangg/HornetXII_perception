# HornetXII_perception

Vision models for the Hornet XII AUV — object detection for SAUVC 2026.
Four classes: `flag`, `gate`, `flare`, `bucket`. Deployed on a Jetson Orin Nano.

---

## Repo layout

```
models/       trained models, one folder each (see Conventions below)
training/     training code — runs on Colab, Kaggle, or the SoC cluster
datasets/     dataset configs (the .yaml files, not the images)
inference/    code that runs on the Orin
docs/         deployment and cluster notes
```

---

## Quick start

Two ways to train. They produce the same thing — pick whichever you have access to.

### A. Notebook — Colab or Kaggle

Easiest to start, no cluster account needed. Free GPUs are modest (T4 class),
so a full run takes 1–2 hours.

```python
!pip install -q ultralytics kagglehub

import kagglehub
path = kagglehub.dataset_download("kushagrajaveri/mantaclaus-sauvc-2026-yolo-dataset")
print(path)   # note this, the data.yaml needs an absolute path
```

The dataset ships a `data.yaml` with `path: yolo_dataset` — a **relative path that
resolves nowhere**. Rewrite it before training or nothing will load:

```python
import yaml
cfg = {
    "path":  f"{path}/yolo_dataset_v2/yolo_dataset_v2",
    "train": "images/train", "val": "images/valid", "test": "images/test",
    "nc": 4, "names": ["flag", "gate", "flare", "bucket"],
}
yaml.safe_dump(cfg, open("sauvc.yaml", "w"))
```

```python
from ultralytics import YOLO
YOLO("yolo11n.pt").train(data="sauvc.yaml", epochs=100, imgsz=640, batch=16)
```

**Use both datasets, not just `yolo_dataset_v2`.** See *Datasets* below — this
is the single easiest mistake to make here.

### B. NUS SoC compute cluster

Faster (H100s) and the runs are reproducible. Needs an SoC account with cluster
access enabled at `mysoc.nus.edu.sg`.

```bash
# once: build the environment
sbatch -A allusers -p gpu --gres=gpu:h100-96:1 -c 8 --mem=32G --time=01:00:00 \
       --wrap "bash training/setup_env.sh"

# once: fetch the dataset
srun -A allusers -c 4 --mem=16G --time=01:00:00 bash training/get_data.sh

# train — any train.py flag passes through
sbatch training/run_train.sbatch
sbatch training/run_train.sbatch --imgsz 960 --epochs 300

# watch
squeue -u $USER
tail -f ~/sauvc/sauvc-<jobid>.out
```

**Everything must run under `srun`/`sbatch`.** The login node caps virtual
memory at 1 GB, processes at 64, and CPU time at 300 s — `import torch` fails
there. `/tmp` also has a 10 MB quota. See `docs/CLUSTER.md`.

Request `--gres=gpu:h100-96:1` or `gpu:nv:1`. **Not `a100-40`** — those nodes
have MIG enabled, and torch reports `device_count=1` with `is_available()=False`,
which looks like a broken install rather than a bad request.

---

## Datasets

The Kaggle dataset bundles **two** sets — train on both:

| config | images | water |
|---|---|---|
| `yolo_dataset_v2` | 5547 / 1253 / 723 | clearer |
| `finetune_dataset` | 898 / 249 / 110 | **turbid** (~2.5x the green/blue cast) |

A model trained on the clear set alone scores **0.536** mAP50-95 on turbid water
versus **0.739** for one trained on both — and competition conditions are closer
to turbid. It will validate beautifully and underperform in the pool.

Don't fine-tune sequentially either: it fixes turbid but drops flare recall on
clear water from 0.82 to 0.46. Pool the two sets instead.

Configs in `datasets/`.

---

## Model conventions

```
models/<arch>_<imgsz>_<dataset>_<YYYYMMDD>_v<N>/
```

`yolo11n_640_combined_20260921_v1` — one folder per model, containing
`MODEL_CARD.md`, `model.pt`, `args.yaml`, `metrics/` and `plots/`.

**A metric without its validation set is not a result.** Never compare two
models unless they were measured on the same validation set — the gap between
two validation sets can exceed the gap between a good model and a bad one.

Full rules, including how each model pins the code that produced it:
[`models/CONVENTIONS.md`](models/CONVENTIONS.md).
`training/package_model.py` produces the layout from any Ultralytics run.

---

## Deployment — Jetson Orin Nano

Build the TensorRT engine **on the board**; engines are not portable.

```bash
yolo export model=model.pt format=engine half=True imgsz=640
```

Favour recall over precision — a missed gate fails a task, a false positive the
control logic rejects costs nothing. Full notes: [`docs/DEPLOY_ORIN.md`](docs/DEPLOY_ORIN.md).
