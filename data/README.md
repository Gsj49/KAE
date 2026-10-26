# Data layout

Default input paths are set in [paths.json](paths.json), relative to this
directory. Place data here or set `KAE_DATA_ROOT` to another directory with
the same layout. Command-line data paths override these defaults.

```text
data/
  paths.json
  check.py
  dapo/train.parquet
  math/train_level_ge3.parquet
  math/test.parquet
  gsm8k/train.parquet
  gsm8k/test.parquet
  openr1/valid.all.parquet
  oneshot/question_128copies.parquet
  code/humaneval_plus/test.parquet
  code/mbpp_plus/test.parquet
  rollouts/gsm8k/<saved rollout parquets>
  rollouts/math/<saved rollout parquets>
  value_mse/gsm8k/<reward-only parquets>
  value_mse/math/<reward-only parquets>
  traces/gsm8k_prompt_trace.jsonl
  traces/math_prompt_trace.jsonl
  mc_error/reference_counts.csv
  computing_cost/runs.csv
  computing_cost/steps.csv
  computing_cost/gpu_memory.csv.gz
```

| Entry point | Default inputs |
|---|---|
| `train_scripts/train.py --dataset dapo` | DAPO train; combined mathematical validation set |
| `train_scripts/train.py --dataset math` | MATH levels 3–5 train; combined mathematical validation set |
| `train_scripts/train.py --purpose estimation` | GSM8K/MATH train and its own test set |
| `train_scripts/single_stream.py` | GSM8K/MATH train and its own test set |
| `train_scripts/one_shot.py` | Prepared 128-copy question; MATH test set |
| `test_scripts/evaluate_math.py` | Combined six-benchmark `valid.all.parquet` |
| `test_scripts/code/generate.py` | HumanEval+ and MBPP+ under `code/` |
| `experiments/run.py` | The dataset's saved pools under `rollouts/` |
| `experiments/value_mse/run.py --data-root data/value_mse` | Included IDs and rewards under `value_mse/` |
| `experiments/assumption6/generate_pool.py` | Training data and saved trace; writes into the dataset's rollout directory |

Place the separately supplied training and test datasets in the layout above.
File sizes, row counts and SHA-256 values are recorded in the
[training/math manifest](training_assets.json) and
[code manifest](code_assets.json).
DAPO's source file `dapo-math/dapo-math-17k-rename.parquet` is stored here as
`dapo/train.parquet` without changing its contents.

Reward samples for CPU value MSE are included under `value_mse/`: 30 files,
about 116 KiB, containing only prompt IDs and rewards. See
[CPU calculation commands](../experiments/value_mse/REWARD_ASSETS.md).
Full response pools are supplied separately with `--rollouts /path/to/pools`.
Prepare the one-shot data with
`prepare_oneshot.py --row <question_index>`.

The small numeric inputs for the [MC-error](../experiments/mc_error/README.md)
and [computing-cost](../experiments/computing_cost/README.md) tables are included
and tracked in Git (about 0.57 MiB total): reference success counts, timings,
token counts and memory measurements.

Large data files are ignored by Git and distributed separately. Place model
weights under `models/`.

## Sources

- Training: [GSM8K](https://github.com/openai/grade-school-math),
  [MATH](https://arxiv.org/abs/2103.03874) and [DAPO](https://github.com/BytedTsinghua-SIA/DAPO).
- Mathematics evaluation: [AIME/AMC](https://maa.org/student-programs/amc/),
  [MATH](https://arxiv.org/abs/2103.03874), [Minerva](https://arxiv.org/abs/2206.14858)
  and [OlympiadBench](https://github.com/OpenBMB/OlympiadBench).
- HumanEval+ and MBPP+: [EvalPlus](https://github.com/evalplus/evalplus).

Saved response pools were generated for our experiments. Dataset terms and
citations are provided by the linked sources.

## Check available files

```bash
python data/check.py
python data/check.py --group training --strict
python data/check.py --group math-eval --strict
python data/check.py --group code --strict
```

This optional command checks only that files exist and are nonempty; it does
not read the asset manifests. `--strict` makes missing or empty files an error.

## Examples with default paths

```bash
python train_scripts/train.py --dataset dapo --method kae --seed 1 \
  --output results/dapo_kae_seed1

python train_scripts/prepare_oneshot.py --row 0
python train_scripts/one_shot.py --method pool_oracle --output results/oneshot_pool

python train_scripts/single_stream.py --dataset gsm8k --method kae \
  --output results/single_gsm8k_kae

python test_scripts/evaluate_math.py --trained-on dapo \
  --model models/dapo_kae_seed1_step500 --run-name dapo_kae_seed1 \
  --output results/math_eval/dapo_kae_seed1

python test_scripts/code/generate.py --model models/dapo_kae_seed1_step500 \
  --method kae --train-seed 1 --mode k16 --output results/code/kae_seed1

python experiments/value_mse/run.py --data-name math --target-step 10 \
  --data-root data/value_mse --oracle-sample-tag '' --n-bootstrap 0 \
  --output-dir results/value_mse/math_step10
```

The training/evaluation launchers and `experiments/run.py` preview by default;
add `--execute` to run. The value-MSE and two table commands run immediately.
`prepare_oneshot.py` is a data-preparation command and writes the
selected question's copies immediately, refusing an existing output file.
