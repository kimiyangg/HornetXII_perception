#!/bin/bash
# Build the YOLO training environment.
#
# MUST run on a compute node, never on xlogin: the login node caps virtual
# memory at 1GB and CPU time at 300s, under which pip and torch cannot run.
#   srun -A allusers -p gpu --gres=gpu:a100-40:1 -c 8 --mem=32G --time=01:00:00 \
#        bash ~/hornet/cluster/setup_env.sh
set -euo pipefail

case "$(hostname)" in
  xlogin*) echo "ERROR: run this under srun on a compute node, not $(hostname)." >&2; exit 1 ;;
esac

PROJ=${PROJ:-$HOME/sauvc}
VENV="$PROJ/venv"
mkdir -p "$PROJ"

echo "==> creating venv at $VENV (python $(python3 --version 2>&1 | cut -d' ' -f2))"
python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"

pip install --quiet --upgrade pip wheel setuptools

echo "==> installing torch (bundled CUDA; node driver is 580.x so cu12x is fine)"
pip install --quiet torch torchvision

echo "==> installing ultralytics + export toolchain"
pip install --quiet ultralytics onnx onnxslim onnxruntime-gpu kaggle

echo "==> verifying"
python - <<'PY'
import torch, ultralytics
print("torch       ", torch.__version__)
print("ultralytics ", ultralytics.__version__)
print("cuda avail  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device      ", torch.cuda.get_device_name(0))
    print("capability  ", torch.cuda.get_device_capability(0))
PY
echo "==> done. activate with: source $VENV/bin/activate"
