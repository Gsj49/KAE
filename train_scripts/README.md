# Policy optimization

Run from the repository root with your environment activated. `train.py`
prints the command; add `--execute` to run it, or use the Slurm wrapper.
Default data paths are in [../data/paths.json](../data/paths.json);
command-line paths override them.

```bash
python train_scripts/train.py --dataset dapo --method kae --seed 1 --kernel-ih 10 \
  --output results/dapo_kae_seed1
```

For example, on a four-GPU node:

```bash
sbatch --gpus=4 --cpus-per-task=72 --time=16:00:00 train_scripts/run.sbatch \
  --dataset dapo --method kae --seed 1 \
  --output results/dapo_kae_seed1
```

The three training data seeds are **1, 42, 420**. Change `--method` for the
seven-method comparison.

| Method | Estimator | Reward/advantage standardization | Loss aggregation |
|---|---|---|---|
| GRPO | upstream GRPO | current-group standard deviation | token mean |
| DrGRPO | upstream GRPO | none | sequence mean, token sum, fixed-length normalization |
| GPG | upstream GPG | upstream GPG rule | upstream GPG policy loss |
| JS (`js`) | James–Stein shrinkage | none | sequence mean of token means |
| Kalman | Kalman posterior baseline | posterior standard deviation | token mean |
| EMA | temporal exponential averaging, leave-one-out | none | token mean |
| KAE | temporal triangular weighting, leave-one-out | none | token mean |
| KAE + normalization (`kae_norm`) | same KAE baseline | current-group std, fallback 0.5 below 0.1 | token mean |

`reinforcepp` is also available for the upstream comparison. It is not one of
the seven methods in the repeated DAPO/MATH policy tables.

`--method kae` uses `grpo_kae_tri` (triangular kernel), and `--method ema`
uses `grpo_kae_exp` (exponential kernel).
`--method kae_norm` also uses `grpo_kae_tri`, with `kae_std_mode=group` and
`kae_std_fallback=0.5`.

## Configurations

The complete settings are in [configs/policy_optimization.json](configs/policy_optimization.json)
and `train.py`. Batch size is 256 prompts, PPO minibatch size is 128, learning
rate is 1e-6, and no KL or entropy penalty is used. `J` below repeats a prompt
batch for consecutive updates; each update generates **new** responses.
It is distinct from the number of responses per prompt, `G`.

| Setting | Model | Steps | G | J | Prompt / response limit | GPUs |
|---|---|---:|---:|---:|---:|---:|
| DAPO | Qwen2.5-Math-7B | 500 | 4 | 8 | 1024 / 3072 | 4 |
| MATH | Qwen2.5-Math-7B | 200 | 8 | 10 | 1024 / 2048 | 4 |
| GSM8K estimation checkpoints | Qwen2.5-1.5B-Instruct | 200 | 8 | 10 | 512 / 512 | 1 |

Use the paper's preprocessed training files: DAPO training data, MATH levels
3–5 (`train_level_ge3.parquet`), or GSM8K. `--model` overrides the model location,
not the architecture. Preprocessing must preserve each training row's unique
`extra_info.index`, chat-format `prompt`, `data_source`, and `reward_model.ground_truth`.
The launcher validates these fields and records the data hash.

For the triangular kernel, set the product $ih$ with `--kernel-ih`
(`algorithm.kae_ih` in VERL). The default is 10 throughout training:
\[
  w_d=\max\{1-d/(ih),0\},\qquad \alpha=1/(ih),\qquad h_i=(ih)/i.
\]
Here $d=0$ denotes the current group. Thus `--kernel-ih 10` gives weights
1, 0.9, 0.8, and so on; `--kernel-ih 5` gives 1, 0.8, 0.6, and so on.
The prompt-reuse schedule supplies the historical groups, while $ih$ controls
their weights. The triangular history resets after a gap in prompt visits.
Switching is disabled by default (`kae_ih_switch_step=0`). An optional
`--kernel-ih-after` together with `--kernel-switch-step` changes $ih$ after
the specified step.
EMA uses beta 0.2 and removes the current response's weight for its
leave-one-out baseline.

Ray uses 20 CPUs; the Slurm allocation is specified separately.

## Checkpoints for value/gradient MSE and Assumption 6

Use `--purpose estimation --method kae`, with `--dataset gsm8k` or `math`.
This saves every step and validates every 20 steps. For this MATH experiment,
use `data/math/test.parquet` for validation, not the six-benchmark `valid.all`.
The training prompt IDs are recorded in `prompt_trace.jsonl`.

Outputs include the command/configuration, checkpoints and W&B logs
(offline by default). Set `WANDB_MODE=online` after logging in to W&B to sync logs.

Use an empty output directory. Checkpoints save model weights and baseline
history, without optimizer state. Resume is disabled in the launcher.

## One-shot illustration

Prepare one chosen question as 128 separate copies. The input must already
have VERL's chat and reward schema. `--row` selects the question explicitly.

```bash
python train_scripts/prepare_oneshot.py --row 0

python train_scripts/one_shot.py --method kae \
  --output results/oneshot_kae
```

Choose `grpo`, `pool_oracle` or `kae` using the **same** prepared file:

| Method | Baseline inputs | Identifier |
|---|---|---|
| GRPO | the current G=8 responses of one copy | fresh per-copy rollout UID |
| Pool oracle | all B×G=128×8=1024 responses to the same question at the current step | shared `extra_info.pool_id` |
| KAE | the current group and historical groups of that copy | distinct, persistent `extra_info.index` |

The pooled baseline is a finite-sample reference. All 128 copies form one
batch, reused at every step.

The preset uses Qwen2.5-Math-1.5B, 100 updates, four GPUs, prompt/response limits
1024/2048, learning rate 1e-6, and fixed triangular $ih=5$ (alpha 0.2). All three
methods disable reward standardization and use `seq-mean-token-mean` loss
aggregation. Here `grpo` is the unnormalized per-copy reward-mean baseline.

Add `--execute`, or submit `train_scripts/run_illustration.sbatch one_shot`
with the same options and four GPUs. The launcher checks that copies share
the same question and answer while retaining different history IDs.

## Single-stream policy optimization

```bash
python train_scripts/single_stream.py --dataset gsm8k --method kae --seed 1 \
  --output results/single_gsm8k_kae_run1
```

Methods are `reinforce`, `reinforce_schedule` and `kae`. They all generate
one response per prompt (G=1), use 1024 prompts per batch, PPO minibatches
of 512 prompts, and train for 400 steps. The schedule variants reuse each
prompt batch for 10 updates. KAE uses fixed $ih=5$ (alpha 0.2).
Both illustration launchers accept `--kernel-ih` to override this value.

| Dataset | Model | Prompt / response limit | GPUs |
|---|---|---:|---:|
| GSM8K | Qwen2.5-1.5B-Instruct | 512 / 512 | 1 |
| MATH | Qwen2.5-Math-1.5B | 1024 / 2048 | 4 |

REINFORCE uses baseline zero: the unscheduled variant divides the reward by
`1 + 1e-6`, and the scheduled variant uses the raw reward. KAE has zero
advantage on the first visit, then subtracts the weighted mean of current
and historical rewards. Its G=1 case does not apply leave-one-out rescaling.

Use separate output directories for the five GSM8K replications. `--seed`
controls the data sampler.
Submit through `run_illustration.sbatch single_stream`, with the matching GPU
count. Validation accuracy is logged every five updates for these curves.
