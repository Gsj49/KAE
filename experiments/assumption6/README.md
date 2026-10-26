# Assumption 6 calculations

Use `experiments/run.py --experiment assumption6` for the complete analysis.
The score calculation uses all model parameters and the sum of response-token
log probabilities, with 128 saved completions for each of 256 prompts.

## Correlation and optimal baseline

For each prompt, estimate the covariance and Pearson correlation between the
reward and the squared policy-score norm from the 128 completions. Estimate
the value by the reward mean and the optimal baseline by the score-norm-weighted
reward mean. The paper summary reports mean absolute correlation, its sample
SD, the negative-correlation fraction, and the mean absolute optimal-baseline
displacement over prompts with defined correlations. Constant-reward prompts
do not enter these four statistics.

## Cross term and quadratic term

Write $q_i(x)$ for the mean squared score norm and $\kappa_i(x)$ for its
covariance with reward, both estimated using all 128 completions. For KAE,
the calculation uses four preceding steps, triangular alpha 0.2, a held-out
100-response value reference, group size $G=8$, and $R=5$ repetitions with
seed 1234. Each repetition supplies eight completion-level leave-one-out
baseline estimates, giving 40 estimates per prompt.

With the reference estimate $\widehat V_{i,\mathrm{ref}}(x)$ fixed across
repetitions, compute

\[
\widehat{\mathrm{Bias}}_i(x)
=\frac{1}{RG}\sum_{r=1}^R\sum_{g=1}^G
\left[\widehat V_i^{(g,r)}(x)-\widehat V_{i,\mathrm{ref}}(x)\right],
\qquad
\widehat{\mathrm{MSE}}_i(x)
=\frac{1}{RG}\sum_{r=1}^R\sum_{g=1}^G
\left[\widehat V_i^{(g,r)}(x)-\widehat V_{i,\mathrm{ref}}(x)\right]^2.
\]

The reported signed ratio is $L/Q$, where

\[
L=-\frac{2}{B}\sum_{b=1}^B
\widehat\kappa_i(x_b)\widehat{\mathrm{Bias}}_i(x_b),
\qquad
Q=\frac{1}{B}\sum_{b=1}^B
\widehat q_i(x_b)\widehat{\mathrm{MSE}}_i(x_b).
\]

Both averages include all $B=256$ prompts. The common factor $1/(BG)$ in
the theoretical cross and quadratic terms cancels from their ratio.
The quadratic term includes baseline variance: it uses the mean of the
squared errors, not the square of the mean error. The stored variance uses
divisor $RG$, so MSE equals squared mean error plus variance exactly.

The 100-response reference used here is distinct from the full-pool value
estimate used for the optimal-baseline displacement above.

## Reuse saved score norms

The following CPU commands recompute baseline moments and the paper summary
without loading a model or regenerating responses:

```bash
python experiments/assumption6/build_baseline_table.py \
  --rollout-dir data/rollouts/math --dataset math --target-step 10 \
  --output results/assumption6/math_step10/kae_baselines.parquet

python experiments/assumption6/report.py \
  --scores results/assumption6/math_step10/scores \
  --baseline-table results/assumption6/math_step10/kae_baselines.parquet \
  --output results/assumption6/math_step10/paper_summary.json
```

Use a new output filename when retaining an existing summary. The score
directory must contain `per_completion.parquet`. Repeat for GSM8K and MATH
at steps 10, 90 and 170. Baseline tables contain `baseline_value`,
`oracle_value`, `baseline_mse`, `baseline_variance` and `baseline_samples`;
a table containing only averaged baselines is insufficient for the quadratic
term.
