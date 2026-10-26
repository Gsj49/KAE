# Monte Carlo reference error

Recreate the supplement's `tab:MCSE` table on CPU:

```bash
python experiments/mc_error/run.py
```

The small input `data/mc_error/reference_counts.csv` contains one row per
prompt: dataset, training step, prompt ID, reference size `n`, and number of
correct reference responses. It retains the original 100-response reference
for each of 256 prompts at steps 10, 90 and 170 on GSM8K and MATH.

For each prompt, compute `p = successes / n` and `SE = sqrt(p*(1-p)/n)`.
Report the mean, median and linearly interpolated 90th percentile across
prompts, using the binomial plug-in standard error without a finite-pool
correction.

Outputs are `results/mc_error/reference_mc_summary.csv` and
`reference_mc_table.tex` (a tabular fragment requiring `booktabs`). An existing
nonempty output directory is not overwritten; select another with `--output`.

To recompute the same counts from the separately supplied full or reward-only
sample pools, install NumPy, pandas and PyArrow and run:

```bash
python experiments/mc_error/run.py --rollouts data/rollouts --output results/mc_error_from_pools
```

This follows the value-MSE reference selection: 100 of 128 rewards without
replacement, seed 1234, prompts ordered by their string IDs. The default
counts-only command uses the Python standard library.
