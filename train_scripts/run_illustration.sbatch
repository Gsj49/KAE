#!/usr/bin/env bash
# Supply the site's resources to sbatch: one_shot=4 GPUs; single GSM=1, MATH=4.
set -euo pipefail
ROOT=${KAE_ROOT:-${SLURM_SUBMIT_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}}
experiment=${1:?Specify one_shot or single_stream}
shift
case "$experiment" in one_shot|single_stream) ;; *) exit 2 ;; esac
exec "${PYTHON:-python}" "$ROOT/train_scripts/$experiment.py" "$@" --execute
