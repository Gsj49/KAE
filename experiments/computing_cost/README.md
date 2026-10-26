# DAPO computing cost

Recreate the supplement's `tab:response-resources` table on CPU:

```bash
python experiments/computing_cost/run.py
```

The supplied files under `data/computing_cost/` contain the measurements used
for the table: one 500-step DAPO run per method on four NVIDIA GH200 GPUs.

- `runs.csv`: method, GPU count, number of steps, and elapsed training-loop
  seconds from the final progress marker.
- `steps.csv`: method, step, training-step seconds, advantage-stage seconds,
  and total prompt/response tokens, extracted from all 3,000 step records.
- `gpu_memory.csv.gz`: method, GPU index, consecutive sample index, and
  device memory in bytes, extracted from the original W&B system records.

The table reports training-loop hours, the mean training-step time, the mean
advantage time in milliseconds, `sum(tokens)/(4*sum(step_seconds))`, and the
largest sampled device memory divided by `1024**3`. Training-loop time includes
validation and checkpointing but excludes initial setup. The memory statistic
is the maximum of periodic device measurements, not a PyTorch allocator peak.

Outputs are `results/computing_cost/computing_cost_summary.csv` and
`computing_cost_table.tex` (a tabular fragment requiring `booktabs`). Use
`--output` for a different empty directory. Only the Python standard library
is needed.
