#!/usr/bin/env bash
# Submit with Slurm on Isambard using one NVIDIA GH200 GPU.
# Set CPU, memory and time requests for the experiment at submission.
set -euo pipefail
ROOT=${KAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
exec "${PYTHON:-python}" "$ROOT/experiments/run.py" "$@" --execute
