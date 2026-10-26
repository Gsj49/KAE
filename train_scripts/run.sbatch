#!/usr/bin/env bash
# Supply your site's account, partition, GPUs, CPUs and time to sbatch.
# Activate the intended environment before submitting. No jobs are submitted here.
set -euo pipefail
ROOT=${KAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
exec "${PYTHON:-python}" "$ROOT/train_scripts/train.py" "$@" --execute
