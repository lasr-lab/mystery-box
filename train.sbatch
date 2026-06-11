#!/bin/bash
#SBATCH --job-name=tactile-train
#SBATCH --output=slurm/%x_%j.out
#SBATCH --error=slurm/%x_%j.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=187000M
#SBATCH --time=2:00:00
#SBATCH --mail-user=max.haufe@tu-dresden.de
#SBATCH --mail-type=FAIL

set -euo pipefail

# Optional: keep logs from different runs separated
mkdir -p slurm

# Direct Python used in your environment.
PYTHON_BIN="/data/cat/ws/maha688h-secai_demo/conda_envs/secai_demo_server/bin/python"

# Ensure we can run from wherever the job is launched.
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1

echo "Using Python: $PYTHON_BIN"
srun "$PYTHON_BIN" -m src.ML.train "$@"
