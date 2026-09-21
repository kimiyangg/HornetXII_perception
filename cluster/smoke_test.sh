#!/bin/bash
# End-to-end pipeline check: train 2 epochs on the built-in coco8 dataset,
# then export to ONNX. Proves venv + GPU + ultralytics + export all work.
set -euo pipefail
# Keep BLAS/OpenMP inside the cgroup CPU allocation.
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

PROJ=${PROJ:-$HOME/sauvc}
source "$PROJ/venv/bin/activate"
cd "$PROJ"

echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA NOT AVAILABLE'; print('cuda OK:', torch.cuda.get_device_name(0))"

echo "==> train (coco8, 2 epochs)"
yolo detect train model=yolo11n.pt data=coco8.yaml epochs=2 imgsz=640 \
     device=0 project="$PROJ/runs" name=smoke exist_ok=True plots=False

W="$PROJ/runs/smoke/weights/best.pt"
echo "==> export $W"
# Straight to ONNX via ultralytics. For the Orin the engine is built on the
# board itself, so nothing more is needed here.
yolo export model="$W" format=onnx imgsz=640 simplify=True

echo "==> SMOKE TEST PASSED"
