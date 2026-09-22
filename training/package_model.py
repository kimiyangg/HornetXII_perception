#!/usr/bin/env python3
"""Package an Ultralytics run directory into the standard models/ layout.

See models/CONVENTIONS.md. Pulls the metrics out of results.csv, records the
environment, copies only the artifacts worth keeping, and writes a MODEL_CARD
skeleton with the numbers already filled in -- the parts a human must supply
are marked TODO.

    python training/package_model.py <run_dir> --name yolo11n_640_combined_v1 \
        --dataset combined --author "Kimi" --val-set "clean+turbid (1502 imgs)"
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from datetime import date
from pathlib import Path

# Only these are worth committing. Training batch previews show augmentation
# rather than results, and there are usually dozens of them.
PLOTS = ["results.png", "confusion_matrix.png", "confusion_matrix_normalized.png",
         "BoxPR_curve.png", "BoxF1_curve.png", "val_batch0_pred.jpg", "labels.jpg"]
METRICS = ["results.csv", "dataset_summary.csv", "environment.json"]


def best_row(results_csv: Path) -> dict | None:
    """The epoch with the highest mAP50-95 -- which is what best.pt holds."""
    if not results_csv.exists():
        return None
    rows = [{k.strip(): v for k, v in r.items()}
            for r in csv.DictReader(results_csv.open())]
    if not rows:
        return None
    return max(rows, key=lambda r: float(r.get("metrics/mAP50-95(B)", 0) or 0))


def environment() -> dict:
    env = {"python": sys.version.split()[0]}
    try:
        import torch
        env["torch"] = torch.__version__
        env["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            env["gpu"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import ultralytics
        env["ultralytics"] = ultralytics.__version__
    except Exception:
        pass
    return env


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path, help="an ultralytics run directory")
    ap.add_argument("--name", required=True, help="<arch>_<imgsz>_<dataset>_v<N>")
    ap.add_argument("--dataset", required=True, help="which dataset config was used")
    ap.add_argument("--author", required=True)
    ap.add_argument("--val-set", required=True,
                    help="the validation set the metrics were measured on -- "
                         "a number without this is not a result")
    ap.add_argument("--models-dir", type=Path, default=Path("models"))
    args = ap.parse_args()

    run = args.run_dir
    weights = run / "weights" / "best.pt"
    if not weights.exists():
        weights = run / "best.pt"
    if not weights.exists():
        print(f"no best.pt under {run}", file=sys.stderr)
        return 1

    out = args.models_dir / args.name
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "plots").mkdir(parents=True, exist_ok=True)

    shutil.copy2(weights, out / "model.pt")
    if (run / "args.yaml").exists():
        shutil.copy2(run / "args.yaml", out / "args.yaml")
    for f in METRICS:
        if (run / f).exists():
            shutil.copy2(run / f, out / "metrics" / f)
    for f in PLOTS:
        if (run / f).exists():
            shutil.copy2(run / f, out / "plots" / f)

    if not (out / "metrics" / "environment.json").exists():
        (out / "metrics" / "environment.json").write_text(
            json.dumps(environment(), indent=2) + "\n")

    b = best_row(run / "results.csv")
    size_mb = (out / "model.pt").stat().st_size / 1e6

    def g(k, d="?"):
        try:
            return f"{float(b[k]):.4f}"
        except Exception:
            return d

    card = out / "MODEL_CARD.md"
    if card.exists():
        print(f"MODEL_CARD.md already exists, leaving it alone")
    else:
        card.write_text(f"""# {args.name}

**Author:** {args.author}  **Date:** {date.today().isoformat()}

## What it is

| | |
|---|---|
| file | `model.pt` ({size_mb:.1f} MB) |
| architecture | TODO |
| input | TODO |
| classes | `0: flag`, `1: gate`, `2: flare`, `3: bucket` |
| trained | epoch {int(float(b["epoch"])) if b else "?"}, see `args.yaml` |

## Training data

`{args.dataset}` -- TODO: name the source and images per split.

## Metrics

**Measured on: {args.val_set}**

| P | R | mAP50 | mAP50-95 |
|---|---|---|---|
| {g("metrics/precision(B)")} | {g("metrics/recall(B)")} | {g("metrics/mAP50(B)")} | {g("metrics/mAP50-95(B)")} |

TODO: per-class breakdown. TODO: if this was validated on only one domain,
say so here -- the number does not generalise on its own.

## Known limitations

TODO

## Reproduce

```bash
TODO: the exact command
```

Environment: see `metrics/environment.json`.
""")

    print(f"packaged -> {out}")
    for p in sorted(out.rglob("*")):
        if p.is_file():
            print(f"  {p.relative_to(out)}")
    if b:
        print(f"\nbest epoch {int(float(b['epoch']))}: mAP50={g('metrics/mAP50(B)')} "
              f"mAP50-95={g('metrics/mAP50-95(B)')}")
    print(f"\nNow fill in the TODOs in {card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
