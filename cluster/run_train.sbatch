#!/bin/bash
#SBATCH --job-name=sauvc
#SBATCH --account=allusers
#SBATCH --partition=gpu-long
#SBATCH --gres=gpu:h100-96:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=%x-%j.out
#SBATCH --error=%x-%j.out
#
# Thin wrapper: all the logic lives in train.py. Arguments pass straight through.
#
#   sbatch run_train.sbatch
#   sbatch run_train.sbatch --imgsz 960 --epochs 300
#   sbatch run_train.sbatch --stage finetune --weights ~/sauvc/runs/x/weights/best.pt
set -euo pipefail

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-8}
export MKL_NUM_THREADS=$OMP_NUM_THREADS

source ~/sauvc/venv/bin/activate
python ~/hornet/cluster/train.py --workers "${SLURM_CPUS_PER_TASK:-8}" "$@"
