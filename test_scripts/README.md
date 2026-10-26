# Mathematical reasoning evaluation

First merge an FSDP actor checkpoint with the upstream merger:

```bash
python -m verl.model_merger merge --backend fsdp \
  --local_dir results/dapo_kae_seed1/checkpoints/global_step_500/actor \
  --target_dir models/dapo_kae_seed1_step500
```

Preview the evaluation command:

```bash
python test_scripts/evaluate_math.py --trained-on dapo \
  --model models/dapo_kae_seed1_step500 \
  --run-name dapo_kae_seed1 --output results/eval/dapo_kae_seed1
```

Add `--execute` on an allocated node, or pass the same arguments to
`test_scripts/run_math.sbatch` with four GPUs and enough CPUs for Ray's 48-CPU
configuration. The GH200 allocation used 30 minutes. Submit from the
repository root, or export `KAE_ROOT`.

The input defaults to `data/openr1/valid.all.parquet` through
`data/paths.json`. Use `--data` for an explicit override.

| Training experiment | Checkpoint | Response limit |
|---|---:|---:|
| DAPO | 500 | 3072 |
| MATH | 200 | 2048 |

Both use prompt limit 1024, temperature 0.6, top-p 1, top-k -1, one response
per input row, and generation batch size 1024. The launcher verifies the
dataset SHA-256 and model shards before starting.

The 6,023 input rows comprise AIME24 (960), AIME25 (960), AMC (2,656), MATH500
(500), Minerva (272), and OlympiadBench (675). AIME/AMC questions already appear
32 times; the launcher does not repeat them again. `Avg` is the unweighted mean
of the six benchmark accuracies. Additional pass@32 columns use the existing
32 responses per AIME/AMC question.

Outputs are the scored parquet, summary CSV/JSON and completion receipt.
`_pre_reward.parquet` is the intermediate, unscored output.

To aggregate, repeat `--input` for all final result parquets:

```bash
python test_scripts/summarize_math.py \
  --input results/eval/dapo_kae_seed1/generation_*[0-9].parquet \
  --input results/eval/dapo_kae_seed42/generation_*[0-9].parquet \
  --input results/eval/dapo_kae_seed420/generation_*[0-9].parquet \
  --output-csv results/eval/per_run.csv --aggregate-csv results/eval/mean_sd.csv
```

Aggregate three training seeds with the same evaluation settings.

HumanEval+/MBPP+ use the separate [code entry points](code/README.md): one
generation job per model and one isolated, parallel CPU scoring job for both
datasets. Generated code is never executed directly on the login host.
