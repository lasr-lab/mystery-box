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

mkdir -p slurm

PYTHON_BIN="/data/cat/ws/maha688h-secai_demo/conda_envs/secai_demo_server/bin/python"
REPO_ROOT="${SLURM_SUBMIT_DIR:-$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")}"

if [ ! -d "${REPO_ROOT}/src" ]; then
  echo "Could not locate project root (expected src/ at ${REPO_ROOT})" >&2
  exit 1
fi

cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Python not found or not executable: $PYTHON_BIN" >&2
  exit 1
fi

export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1

echo "Using Python: $PYTHON_BIN"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PYTHON_BIN" -m src.ML.train "$@"
