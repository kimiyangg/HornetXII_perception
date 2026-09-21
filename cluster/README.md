# SAUVC 2026 — YOLO training on the NUS SoC compute cluster

Train on the cluster's A100/H100 nodes, deploy to a Jetson.

## Cluster constraints that shape all of this

`xlogin` is a **login node only**. It enforces hard, unraisable per-user limits:

| limit | value |
|---|---|
| virtual memory (`ulimit -v`) | 1 GB |
| processes/threads (`ulimit -u`) | 64 |
| CPU time (`ulimit -t`) | 300 s |
| `/tmp` quota | 10 MB |

Node.js and the JVM cannot even start under the 1 GB cap, which is why VS Code
Remote-SSH and JetBrains Gateway do not work here (see `../.vscode/sftp.json`
for the sync-based workflow used instead). It also means **pip, unzip and
training must all run under `srun`/`sbatch`**, never on the login node.

Compute nodes have no such limits.

## Layout

    ~/hornet/cluster/     <- these scripts (synced from your Mac via SFTP)
    ~/sauvc/venv/         <- python env   (big, deliberately outside the synced tree)
    ~/sauvc/data/         <- dataset
    ~/sauvc/runs/         <- training outputs

## Files

| file | purpose |
|---|---|
| `train.py` | **all training logic** -- scratch, finetune, resume |
| `run_train.sbatch` | Slurm wrapper; forwards all args to `train.py` |
| `setup_env.sh` | builds the venv (once) |
| `get_data.sh` | downloads the dataset (once) |
| `smoke_test.sh` | end-to-end environment check |
| `DEPLOY_ORIN.md` | Jetson Orin Nano deployment |

## Steps

```bash
# 1. build the environment (once, ~10 min)
sbatch -A allusers -p gpu --gres=gpu:h100-96:1 -c 8 --mem=32G --time=01:00:00 \
       -J yolo-setup -o ~/sauvc/setup-%j.out --wrap "bash ~/hornet/cluster/setup_env.sh"

# 2. get the dataset (once)
srun -A allusers -c 4 --mem=16G --time=01:00:00 bash ~/hornet/cluster/get_data.sh

# 3. train -- any train.py flag passes straight through
sbatch run_train.sbatch
sbatch run_train.sbatch --imgsz 960 --epochs 300
sbatch run_train.sbatch --stage finetune --weights ~/sauvc/runs/<run>/weights/best.pt
sbatch run_train.sbatch --resume ~/sauvc/runs/<run>/weights/last.pt

# 4. watch
squeue -u $USER
tail -f ~/sauvc/sauvc-<jobid>.out

```

Everything must run under `srun`/`sbatch`. On the login node even `import torch`
fails -- see the limits table above.

## Partitions

| partition | time limit | use |
|---|---|---|
| `gpu` | 3 h | quick runs, debugging |
| `gpu-long` | 3 days | real training runs |
| `test` | short | smoke tests |

GPUs available: `a100-40`, `a100-80`, `h100-47`, `h100-96`, `h200-141`, `nv`.
Request with `--gres=gpu:a100-40:1`. Your account is `allusers`, max 16 jobs.
