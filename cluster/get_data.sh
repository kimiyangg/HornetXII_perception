#!/bin/bash
# Fetch the SAUVC YOLO dataset via kagglehub.
#
# kagglehub resolves public datasets anonymously, so no API token is needed
# unless Kaggle starts requiring one for this slug.
#
# Run on a compute node -- xlogin's 300s CPU cap and 1GB VA limit will kill
# a large download/unzip.
#   srun -A allusers -c 4 --mem=16G --time=01:00:00 bash ~/hornet/cluster/get_data.sh
set -euo pipefail

PROJ=${PROJ:-$HOME/sauvc}
SLUG="kushagrajaveri/mantaclaus-sauvc-2026-yolo-dataset"

# Keep the cache inside the project, not ~/.cache, so it's easy to find and purge.
export KAGGLEHUB_CACHE="$PROJ/data"
mkdir -p "$KAGGLEHUB_CACHE"

# shellcheck disable=SC1091
source "$PROJ/venv/bin/activate"
python -c "import kagglehub" 2>/dev/null || pip install --quiet kagglehub

python - <<'PY'
import kagglehub, os, pathlib
path = kagglehub.dataset_download("kushagrajaveri/mantaclaus-sauvc-2026-yolo-dataset")
print("DATASET_PATH:", path)
pathlib.Path(os.environ["PROJ"] + "/dataset_path.txt").write_text(path + "\n")
PY
