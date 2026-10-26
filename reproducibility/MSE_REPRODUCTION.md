# Reproducing estimation experiments from saved checkpoints

The experiments use saved GSM8K and MATH responses and checkpoints.
Value MSE needs only prompt IDs and rewards. Gradient MSE also needs
prompt/response text, the matching checkpoint and a GPU.

## Inputs

| Result | Required samples for target step t | Checkpoint | Entry point | Computation |
|---|---|---|---|---|
| Value MSE | IDs and 128 rewards per prompt at t-4 through t | None | `experiments/value_mse/run.py` | CPU |
| Kernel sensitivity | Same five reward pools | None | `experiments/kernel_sensitivity/run.py` | CPU |
| Gradient MSE | Full target prompt/responses/rewards and four preceding reward pools | Matching dataset and step t | `experiments/run.py --experiment gradient_mse` | One GPU |
| Assumption 6 correlations and optimal baseline | Full target prompt/responses/rewards | Matching dataset and step t | `experiments/run.py --experiment assumption6` | One GPU |
| KAE cross term accompanying Assumption 6 | Target scores plus the four preceding reward pools | Same checkpoint, no additional model required | Included in the Assumption 6 entry | GPU scores followed by CPU analysis |

Each dataset has 256 prompts per pool, 128 completions per prompt, and 15 pools:
steps 6–10, 86–90 and 166–170. Prompt IDs, not row positions, match observations
across steps. `data/rollout_assets.json` lists every filename, size
and SHA-256. Both datasets together occupy 316,117,078 bytes (about 301.5 MiB).
The optional lossless reward-only projection occupies about 116 KiB and is
sufficient for value MSE and kernel sensitivity, but not for gradient analysis.

Sample files are separate inputs. Place them under `data/rollouts/<dataset>`
or specify their location with `--rollouts`. See
[reward files and CPU commands](../experiments/value_mse/REWARD_ASSETS.md).

## Checkpoints

Download the six checkpoints from
[Gsj49/verl_reasoning_checkpoints](https://huggingface.co/Gsj49/verl_reasoning_checkpoints)
at revision `0bd75af00032d23191a869d6a7dcf64ac923651e`.
The [checkpoint manifest](../experiments/gradient_mse/checkpoint_assets.json) records all 72 filenames,
sizes and SHA-256 values. The six checkpoints total 56,451,826,300 bytes.

Paths below are relative to the repository's `JASA-REVISION-ASSUMPTION6/` folder.
Each folder contains Hugging Face model weights, tokenizer and configuration.

| Dataset/model | Step | Hugging Face subfolder | Suggested local folder |
|---|---:|---|---|
| GSM8K / Qwen2.5-1.5B | 10 | `assump6-qwen2.5-1.5b-gsm8k-kae-k10-seed1/global_step_10` | `models/gsm8k_step10` |
| GSM8K / Qwen2.5-1.5B | 90 | `assump6-qwen2.5-1.5b-gsm8k-kae-k10-seed1/global_step_90` | `models/gsm8k_step90` |
| GSM8K / Qwen2.5-1.5B | 170 | `assump6-qwen2.5-1.5b-gsm8k-kae-k10-seed1/global_step_170` | `models/gsm8k_step170` |
| MATH / Qwen2.5-Math-7B | 10 | `assump6-qwen2.5-math-7b-math-kae-k10-seed1/global_step_10` | `models/math_step10` |
| MATH / Qwen2.5-Math-7B | 90 | `assump6-qwen2.5-math-7b-math-kae-k10-seed1/global_step_90` | `models/math_step90` |
| MATH / Qwen2.5-Math-7B | 170 | `assump6-qwen2.5-math-7b-math-kae-k10-seed1/global_step_170` | `models/math_step170` |

The gradient launcher verifies local model files against the manifest.

## Gradient configuration

| Setting | GSM8K | MATH |
|---|---|---|
| Target steps | 10, 90, 170 | 10, 90, 170 |
| Selected parameters | All, 1,543,714,304 parameters | Last 5 decoder blocks (indices 23–27), final model norm and lm_head; no embeddings; 1,710,289,920 parameters |
| Statistical batch | 256 prompts × G=8 completions | Same |
| Monte Carlo repeats | 50 | 50 |
| Sampling | With replacement, seed 0, persisted draw indices | Same |
| Reference | Entire 128-completion target pool | Same |
| KAE history | Four previous steps, triangular alpha=0.2, variance floor=0.01 | Same |
| Methods | REINFORCE++, GRPO and KAE, with completion-level LOO | Same |
| RF++ baseline | Other 2047 completions in the full statistical batch | Same |
| Precision and score | bfloat16, sum of response-token log probabilities | Same |
| Reference microbatch | 16 | 8 |
| Repeated estimates | Shared forward for the three methods, `repeat_microbatch=0` | Same |
| Gradient distance | Scalar reduction on GPU | Same |
| Reference system | Isambard, Slurm, one NVIDIA GH200 GPU | Same |

Request resources through Slurm and choose the time limit after a short smoke
test on the allocated node. Stored response sequences are used without new
truncation. Their original generation limits were 512/512 prompt/response
tokens for GSM8K and 1024/2048 for MATH. The Isambard GH200 software versions are recorded in
[environment.txt](environment.txt).

Sampled completion indices are stored in `experiments/gradient_mse/draws/`.
Each run processes all 256 prompts in one task.

```bash
python experiments/run.py --experiment gradient_mse --dataset math --step 10 \
  --rollouts data/rollouts/math --checkpoint models/math_step10 \
  --output results/gradient_mse/math/step_10 --execute
```

Repeat for each dataset and step. Outputs include per-prompt results,
`prompt_details.parquet`, `summary.json`, configuration and verification records.
Use an empty output folder with the common launcher. To resume an interrupted
run, use `global_loo.py` directly with the same arguments.

After all six runs pass:

```bash
python experiments/gradient_mse/report_global_loo.py \
  --results-root results/gradient_mse
```

## Value configuration

Use the same saved datasets and steps, without a model. The value experiment
uses a held-out 100-reward reference, G=8, 100 repeats and seed 1234. Target
groups are drawn from the remaining 28 rewards, without replacement within
each group. KAE uses four preceding steps and alpha=0.2. GRPO, RF++ and KAE
use completion-level LOO; RF++ uses the full 256-prompt statistical batch.

Run `experiments/value_mse/run.py` with reward-only files, or
`experiments/run.py --experiment value_mse` with full response pools.

Assumption 6 uses full-parameter score norms for both models.
Its quadratic term weights the mean squared baseline error by the mean
squared score norm. [The calculation guide](../experiments/assumption6/README.md)
also provides CPU commands for reusing saved scores.
