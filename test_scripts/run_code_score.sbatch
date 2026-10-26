#!/usr/bin/env bash
#SBATCH --cpus-per-task=72
# No GPU requested. The same allocation scores both benchmarks with 64 workers.
set -euo pipefail
ROOT=${KAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
exec "${PYTHON:-python}" "$ROOT/test_scripts/code/score.py" --cpus 72 --workers 64 "$@" --execute
