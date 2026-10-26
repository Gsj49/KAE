# Kernel and bandwidth sensitivity

This CPU experiment compares kernels and bandwidths using saved GSM8K or
MATH rewards.

## Run

~~~bash
for step in 10 90 170; do
  python experiments/run.py --experiment kernel_sensitivity --dataset math \
    --step "$step" --output "results/kernel_sensitivity/math_step$step" --execute
done

python experiments/kernel_sensitivity/plot.py \
  --summaries results/kernel_sensitivity/math_step10/sensitivity/summary.json \
              results/kernel_sensitivity/math_step90/sensitivity/summary.json \
              results/kernel_sensitivity/math_step170/sensitivity/summary.json \
  --output results/kernel_sensitivity/math_figure
~~~

Omit `--execute` to preview. Set `--rollouts` for external pools; the default
is `data/rollouts/math`. Summaries contain mean, SD and SE of prompt MSE.

The common launcher and direct entry both default to 100 repetitions,
matching the paper's value-estimation section:

~~~bash
python experiments/kernel_sensitivity/run.py --dataset math --step 10 \
  --rollouts /path/to/existing/pools --repeats 100 \
  --output results/kernel_sensitivity/math_step10
~~~

The direct entry runs immediately. Use an empty output directory.

## Inputs and sampling

For target step i, provide five files at i-4,...,i, named
math_generation_k10_step_<step>_batchTRACE.parquet (or the GSM8K equivalent).
Each must contain the same 256 explicit prompt IDs and 128 finite rewards in
[0,1] per prompt.

With the current saved pools, 100 target rewards are held out for the reference
value; each repeat draws eight rewards from the remaining 28, and eight from
each historical pool. Draws are without replacement within each repeat, and
can overlap across repeats. All kernels and bandwidths use identical draws.
The RNG seed is 1234. Use `--reference-pool` to supply an independent
100-response target pool instead.

RF++ uses completion-level LOO over the full batch: for each reward, subtract
that reward from the sum of all B*G current rewards and divide by B*G-1.
GRPO and both KAE families also use LOO.

## Weights and bandwidth

Let d be the number of steps before the target step and H=i*h_i.

- Triangular: weights proportional to max(1-alpha*d, 0), with
  H=1/alpha and h_i=1/(i*alpha).
  Alpha grid: 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.
- Exponential:
  b_t=(1-beta)*b_(t-1)+beta*mean_reward_t.
  Away from initialization, weights are proportional to (1-beta)^d.
  For the convention K(u)=exp(-u), H=-1/log(1-beta) and h_i=H/i.
  Beta grid: 1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1.
  Beta=1 is the current-only, zero-bandwidth limit.

The recursion is initialized with the oldest of the four history means.
Consequently its oldest weight is (1-beta)^4, whereas the other four weights
are beta*(1-beta)^d.

For normalized current-group weight w, current group size G=8, and current
reward Z_g, the LOO estimate is (b - w*Z_g/G)/(1-w/G).

## Outputs

- prompt_metrics.parquet: 256 prompts × 22 settings (20 KAE, GRPO, RF++).
- summary.csv and summary.json: prompt-averaged bias/variance/MSE, MSE SD/SE,
  parameter values, H, h_i, configuration and input SHA-256.
- draw_indices.npz: reference, current and history indices, shared across settings.
- complete.json: output and code hashes, written only after success.
- The separate plotting entry produces PDF/PNG and source hashes.
