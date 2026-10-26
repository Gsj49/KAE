# Estimation experiments

The experiments use saved responses in `data/rollouts/<dataset>`.
Use `--rollouts` to specify another location. `run.py` prints the command;
add `--execute` to run it.

The saved GSM8K and MATH pools contain 256 prompts with 128 completions each
at steps 6–10, 86–90 and 166–170. File names follow
`<dataset>_generation_k10_step_<step>_batchTRACE.parquet`.
The `k10` in these names describes prompt reuse, not the response group size.
Hashes are listed in [../data/rollout_assets.json](../data/rollout_assets.json).
The data and checkpoints are distributed separately from source code.

Value MSE and kernel sensitivity need only prompt IDs and rewards; see
[reward export](value_mse/REWARD_ASSETS.md). Gradient MSE and score norms
also require response text and checkpoints.

## MC error and computing-cost tables

These two commands run immediately on CPU using the small numeric inputs
included in `data/`; they need only the Python standard library:

```bash
python experiments/mc_error/run.py
python experiments/computing_cost/run.py
```

Each writes a CSV summary and a LaTeX table under `results/`. See
[MC error](mc_error/README.md) and [computing cost](computing_cost/README.md)
for the calculations and source measurements.

## Value MSE

Runs on CPU, without loading a model:

```bash
python experiments/run.py --experiment value_mse --dataset math --step 10 \
  --rollouts data/rollouts/math --output results/value_mse/math_step10 --execute
```

The comparison includes GRPO, RF++, KAE, JS and Kalman. It uses a 100-response
reference subset, groups of eight, four preceding steps, triangular alpha 0.2
and 100 repetitions. Evaluation groups are sampled separately from the
reference subset.

## Kernel and bandwidth sensitivity

The [kernel_sensitivity entry](kernel_sensitivity/README.md) evaluates ten
triangular and ten exponential settings on CPU using the same five reward
pools and paired draws. A plotting command produces PDF/PNG figures.

```bash
python experiments/run.py --experiment kernel_sensitivity --dataset math --step 10 \
  --rollouts /path/to/existing/pools --output results/kernel_sensitivity/math_step10 --execute
```

Both entry points default to 100 repetitions and apply completion-level LOO
to RF++'s full-batch reward mean, matching the value-estimation protocol.
EMA initializes from the oldest historical reward mean.

## Gradient MSE

```bash
python experiments/run.py --experiment gradient_mse --dataset math --step 10 \
  --rollouts data/rollouts/math --checkpoint models/math_step10 \
  --output results/gradient_mse/math/step_10 --execute

# After both datasets at steps 10, 90 and 170 have completed:
python experiments/gradient_mse/report_global_loo.py \
  --results-root results/gradient_mse
```

The saved-pool experiment uses G=8 and 50 repetitions, with replacement,
history steps t-4 through t-1, and the full 128-response current pool for its
oracle-centered gradient reference. GSM8K uses all parameters. MATH uses the
last five decoder blocks, final model norm and language-model output head,
excluding embeddings (1,710,289,920 parameters).

The aggregator reports RF++, GRPO and KAE (`overt_no_norm` in the saved output).
RF++ uses all 256 x 8 sampled rewards at a repeat and excludes only
the current completion: `(sum_all_rewards - reward)/(2048 - 1)`. Sampling and
batch statistics use the six persisted draw plans in
`experiments/gradient_mse/draws`. The three methods share each repeat's forward
pass, and squared distances are reduced to scalars on the GPU. Reference
microbatches are 16 for GSM8K and 8 for MATH.

The launcher first verifies the downloaded checkpoint against the fixed HF
file manifest, then runs `global_loo.py`. Model and sample locations and
settings are listed in
[MSE_REPRODUCTION.md](../reproducibility/MSE_REPRODUCTION.md).

## Assumption 6

```bash
python experiments/run.py --experiment assumption6 --dataset gsm8k --step 10 \
  --rollouts data/rollouts/gsm8k --checkpoint models/gsm8k_step10 \
  --output results/assumption6/gsm8k_step10 --execute
```

For each saved completion, this computes the squared norm of the gradient of
the **sum** of response-token log probabilities. It uses all parameters on
both models. Within each prompt it computes reward–score-norm covariance and
correlation, the value estimate, and the score-norm-weighted optimal baseline.

`paper_summary.json` reports mean absolute Pearson correlation and its SD,
the fraction of negative correlations, the mean absolute difference between
the optimal baseline and value, and the signed KAE cross-term/quadratic-term
ratio. Correlations and the optimal-baseline difference use prompts with
defined correlations; the cross and quadratic terms average over all prompts.
The quadratic term uses the mean squared baseline error, including its
variance, rather than the squared mean error. Baseline estimates use alpha 0.2, a 100-response
reference, G=8, five repetitions, and four preceding steps. Bootstrap is
optional and disabled by default. See [calculation details and CPU reuse of
saved scores](assumption6/README.md).

Both GPU analyses can run through `experiments/run_gpu.sbatch` with the
same arguments, using one GPU per task. Repeat for steps 10, 90 and 170.

## Recreating the saved pools

Train with `train_scripts/train.py --purpose estimation`, retain the trace and checkpoints,
and merge each needed actor checkpoint using the upstream model merger.
Then `assumption6/generate_pool.py` accepts `--dataset`, `--step`, `--checkpoint`,
`--trace`, `--train-data`, and `--output` and previews the generation command.
It reconstructs the exact training batch and uses 128 samples per prompt,
temperature 1, top-p 1, and top-k -1.
GSM8K generation uses one GPU/Ray 20 CPUs, MATH four GPUs/Ray 41 CPUs.

See [COVERAGE.md](COVERAGE.md) for the paper's experiment entry points.
