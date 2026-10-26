#!/usr/bin/env bash
# Submit from the repo root, or export KAE_ROOT. Activate your environment first.
# Resource example: sbatch --gpus=4 --cpus-per-task=72 --time=00:30:00 ...
set -euo pipefail
ROOT=${KAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
exec "${PYTHON:-python}" "$ROOT/test_scripts/evaluate_math.py" "$@" --execute
