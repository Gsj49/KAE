#!/usr/bin/env bash
# Request 4 GPUs at submission. One model, both benchmarks, one allocation.
set -euo pipefail
ROOT=${KAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
exec "${PYTHON:-python}" "$ROOT/test_scripts/code/generate.py" "$@" --execute
