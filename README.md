# KAE: paper code

Kernelized Advantage Estimation, policy training, evaluation and estimation
experiments built on VERL.

## Layout

```text
verl/                         upstream VERL, with a small KAE integration
data/                         shared data paths and prepared dataset layout
train_scripts/                policy training, one-shot and single-stream settings
test_scripts/                 mathematics evaluation and isolated code evaluation
experiments/
  value_mse/                  CPU baseline estimation comparison
  kernel_sensitivity/         CPU kernel/bandwidth grid from saved rewards
  gradient_mse/               saved-pool gradient estimation comparison
  assumption6/                score norms, correlations and cross terms
  mc_error/                   CPU Monte Carlo reference-error table
  computing_cost/             CPU timing, throughput and memory table
reproducibility/               environment and reproduction instructions
tests/kae/                    small CPU tests for the added components
```

License: Apache-2.0. See `LICENSE` and `NOTICE`.

## Start here

1. Set up [the environment](reproducibility/README.md), then run
   `pip install -e . --no-deps`. The reference system is Isambard with Slurm,
   NVIDIA GH200 GPUs and Linux/aarch64.
2. Place prepared data under [data/](data/README.md); shared defaults are in
   [data/paths.json](data/paths.json). Compact reward samples for value MSE
   are included. Training/test datasets, full response pools and model weights
   are distributed separately.
3. Use [training](train_scripts/README.md),
   [mathematics evaluation](test_scripts/README.md),
   [code evaluation](test_scripts/code/README.md), or
   [estimation experiments](experiments/README.md).
   The Python launchers print their commands unless `--execute` is supplied.
   Value MSE, MC-error and computing-cost scripts run directly on CPU with
   the included numeric inputs.

The method names are `grpo`, `drgrpo`, `gpg`, `js`, `kalman`, `ema`,
`kae`, and the additional `kae_norm` comparison. The primary repeated policy
experiments use training seeds 1, 42 and 420.
Inside VERL, `kae` uses the `grpo_kae_tri` estimator and `ema` uses
`grpo_kae_exp`; `tri` and `exp` denote triangular and exponential kernels.

## Run value MSE on CPU

From the repository root, install the analysis dependencies and run:

```bash
python -m pip install numpy pandas pyarrow matplotlib
python experiments/value_mse/run.py --data-name math --target-step 10 \
  --data-root data/value_mse --oracle-sample-tag '' --n-bootstrap 0 \
  --output-dir results/value_mse/math_step10
```

No GPU, model checkpoint or VERL installation is needed. The included reward
samples cover GSM8K and MATH at steps 10, 90 and 170. See
[the six-run commands](experiments/value_mse/REWARD_ASSETS.md).

## KAE implementation

- `core_algos.py`: adds the paper's triangular KAE, EMA with leave-one-out
  correction, James–Stein and Kalman estimators, and the one-shot pooled baseline.
- `sampler.py`: adds fixed repetition of each prompt batch.
- `ray_trainer.py`: connects repetition, stable prompt IDs, step numbers,
  prompt traces and historical-baseline checkpoint state.
- `kae_state.py`: validates prompt identities and stores/restores history.
- `kae_math_verify.py`: configures mathematical answer extraction and scoring.

## Tests

```bash
python -m unittest discover -s tests/kae -v
python -m unittest discover -s experiments/value_mse -v
```

Tests cover the estimators, batch scheduling, history state and evaluation
checks. A two-step DAPO KAE training run passed on four GH200 GPUs,
including generation, scoring, updates, validation and
checkpoint saving.
See [experiment entry points](experiments/COVERAGE.md) and
[saved-model reproduction](reproducibility/MSE_REPRODUCTION.md).

The optional `paper` dependency extra adds Math-Verify and CPU analysis libraries.
