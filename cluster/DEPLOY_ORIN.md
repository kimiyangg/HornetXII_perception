# Deploying to Jetson Orin Nano

Target confirmed: **Jetson Orin Nano** (Ampere, JetPack 5/6, Python 3.8-3.10).
Ultralytics runs natively on-device, so this is the short path.

## Export on the board

TensorRT engines are tied to the exact GPU + TensorRT version that built them,
so building on the board is the reliable route.

```bash
# on the Orin, once
pip install ultralytics

# copy the trained weights over
scp <you>@xlogin.comp.nus.edu.sg:~/sauvc/runs/<run>/weights/best.pt .

# build the engine (takes a few minutes; FP16 roughly doubles throughput)
yolo export model=best.pt format=engine half=True imgsz=640

# run it
yolo predict model=best.engine source=0 imgsz=640 conf=0.25
```

## Inference in your own code

```python
from ultralytics import YOLO

model = YOLO("best.engine")          # TensorRT engine
results = model(frame, conf=0.25, iou=0.45, verbose=False)

for r in results:
    for box in r.boxes:
        cls  = int(box.cls)          # 0=flag 1=gate 2=flare 3=bucket
        conf = float(box.conf)
        x1, y1, x2, y2 = box.xyxy[0].tolist()
```

## Resolution

Orin has headroom, so input size is a real choice rather than a constraint:

| imgsz | expected FP16 fps | notes |
|---|---|---|
| 416 | ~150+ | unnecessary here |
| 640 | ~80-130 | baseline |
| 960 | ~40-60 | better on the weak `flare` class |

`flare` is the weakest class (mAP50-95 ~0.50 vs ~0.92 for `gate`) because it is
thin and vertical, so small horizontal errors wreck IoU. Higher resolution is
the lever that actually moves it. A 960 run is training now for comparison.
