# Model artifact conventions

Referenced from the [repo README](../README.md).

### Naming

```
models/<arch>_<imgsz>_<dataset>_<YYYYMMDD>_v<N>/
```

`yolo11n_640_combined_20260921_v1`

| part | values | why it is in the name |
|---|---|---|
| `arch` | `yolo11n`, `yolo11s` | sets the deployment budget |
| `imgsz` | `640`, `960` | must match at export or accuracy drops |
| `dataset` | `clean`, `combined`, `field` | most likely thing to differ between runs |
| date | `YYYYMMDD` | self-assigning, sorts chronologically |
| `v<N>` | `v1`, `v2` | distinguishes runs from the same day |

The date does the versioning — it assigns itself, needs no coordination between
teammates, and answers "is this stale?" at a glance. `v<N>` only separates runs
that share a date, which happens whenever two people train in parallel.

**Always write `_v1`, even when there is no collision.** Adding the suffix later
means renaming a folder already referenced by model cards, deploy scripts and
whatever is on the robot.

Author is **not** in the name — git and the model card record it.

## Layout

```
models/yolo11n_640_combined_20260921_v1/
├── MODEL_CARD.md      # REQUIRED
├── model.pt           # always this name
├── args.yaml          # every hyperparameter
├── metrics/
│   ├── results.csv
│   ├── dataset_summary.csv
│   └── environment.json
├── plots/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── BoxPR_curve.png
│   └── val_batch0_pred.jpg
└── train.ipynb        # ONLY if trained outside this repo (see below)
```

`training/package_model.py` produces this from any Ultralytics run directory.

## Pinning the code that produced it

`args.yaml` records every hyperparameter and `environment.json` the library
versions, but neither captures custom code — a callback, an augmentation
change, a different split. So each model must point at the code that made it:

| trained from | how it is pinned |
|---|---|
| **this repo** | the git commit SHA, in `MODEL_CARD.md` under *Reproduce* |
| **an external notebook** (Kaggle, Colab) | commit the `.ipynb` **into the model folder** |

The repo case is a pointer, not a copy — the history already holds every
version of `training/`, and copying it per model guarantees the copies drift.

The notebook case has no history to point at, so the notebook itself is the
only record of how that model was made and travels with it.

## Every MODEL_CARD.md states

1. Architecture, input size, classes, file size
2. Training data — which dataset, images per split
3. **Metrics, with the validation set they were measured on**
4. Known limitations — weak classes, conditions not covered
5. Author and date
6. **How to reproduce** — the git commit SHA and command, or the bundled notebook


---

