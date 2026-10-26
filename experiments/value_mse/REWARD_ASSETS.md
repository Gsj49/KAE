# Saved rewards for CPU-only value-MSE calculation

The repository includes prompt IDs and rewards from the saved GSM8K and MATH
response pools under `data/value_mse/<dataset>/`. The 30 files total about
116 KiB and contain no prompt or response text.

## Reward files

- Two datasets, each with 15 pools: steps 6–10, 86–90 and 166–170.
- Each file contains 256 unique prompts and 128 rewards per prompt.
- `extra_info.index` identifies prompts across steps within each dataset.
- `rewards` contains the original 128 float64 correctness rewards in their
  original order. List position is the completion index. Values are finite
  and in [0, 1].
- Prompt IDs match within each five-step window.

Reward files support value MSE and kernel sensitivity. Gradient MSE and
score-norm calculations also require response text and checkpoints.

## CPU-only calculation

Run from the repository root. No PyTorch, VERL runtime, GPU or model checkpoint
is needed.

```bash
python -m pip install numpy pandas pyarrow matplotlib

for dataset in gsm8k math; do
  for step in 10 90 170; do
    python experiments/value_mse/run.py \
      --data-name "$dataset" --target-step "$step" \
      --steps-glob "data/value_mse/$dataset/${dataset}_generation_k10_step_*_batchTRACE.parquet" \
      --oracle-sample-tag '' --true-n 100 --eval-n 8 --history-n 8 \
      --history-window 4 --repeats 100 --kae-alpha 0.2 \
      --n-bootstrap 0 --seed 1234 \
      --output-dir "results/value_mse/${dataset}_step${step}"
  done
done
```

Use the direct value-MSE entry above for reward-only inputs. The common
`experiments/run.py` launcher validates full response pools and therefore
expects response text as well as rewards.

The current analysis holds out 100 of the 128 target rewards as the reference,
then draws groups of eight from the remaining 28 without replacement within
each draw. Historical groups come from the four preceding pools. GRPO,
REINFORCE++ and KAE use completion-level LOO; RF++ uses the full sampled batch.
