# Code generation and scoring

One GPU job generates responses for HumanEval+ (164 problems) and MBPP+
(378 problems); one CPU job scores both. Inputs are
`data/code/<benchmark>/test.parquet`, with test harnesses in
`reward_model.ground_truth`. See [code_assets.json](../../data/code_assets.json).

## Generate

```bash
python test_scripts/code/generate.py --model models/dapo_kae_seed1_step500 \
  --method kae --train-seed 1 --mode k16 \
  --output results/code/kae_seed1_k16
```

Add `--execute`, or submit the same options with
`test_scripts/run_code_generate.sbatch` and four GPUs. A model is loaded only
once for both datasets. `k16` uses temperature 0.6, top-p 1, top-k -1,
and 16 responses per problem: **8672 completions per model**. `greedy` uses
temperature zero and one response: 542 completions. Both use prompt/response
limits 1024/3072. Use a merged checkpoint as input.

The default data root is `data/code`, configured in `data/paths.json`.
`--data-root` can override it.

## Score in an isolated environment

Run generated Python only in the provided isolated scorer. It extracts
the last Python code block, uses HumanEval+'s signature-prefix fallback and
`check(candidate)`, and executes MBPP+'s module-level assertions. Each response
gets a fresh child process and a 15-second deadline. Failed tests or timeouts
score zero; missing test harnesses or failed scoring workers fail the job.

For Docker, build the small CPU image, then use its immutable image ID:

```bash
docker build -t kae-code-score test_scripts/code
docker image inspect --format '{{.Id}}' kae-code-score
python test_scripts/code/score.py --backend docker --image sha256:YOUR_IMAGE_ID \
  --smoke --cpus 2 --workers 2 --output results/code/sandbox_smoke --execute
```

Replace `sha256:YOUR_IMAGE_ID` with the ID printed above. The container runs
without network access, as an unprivileged user, with a read-only filesystem
and temporary scoring inputs/outputs.

On a Linux cluster with Bubblewrap and a Python environment containing NumPy,
pandas and PyArrow, use `--backend bwrap` instead. It requires user namespaces
and uses Slurm's CPU/memory limits. Run the same smoke test first and keep
isolation enabled.

Then score the real generations, using 72 allocated CPUs and 64 workers:

```bash
sbatch --cpus-per-task=72 --mem=64G --time=01:00:00 test_scripts/run_code_score.sbatch \
  --backend bwrap --input results/code/kae_seed1_k16 \
  --output results/code/kae_seed1_k16_scored
```

Both datasets share the worker pool. Adjust the time limit to the workload.
Failures are recorded in `scoring.log`.

## Metrics and three-run summaries

For exactly 16 responses per problem, `pass@16` is the fraction of problems
with at least one passing response; `avg@16` is the average fraction passing
within each problem. Greedy reports `pass@1` (equal to `avg@1`). `Avg` is the
equally weighted mean across the two benchmarks. Values are stored in [0,1].

```bash
python test_scripts/code/aggregate.py \
  results/code/kae_seed1_k16_scored/summary.json \
  results/code/kae_seed42_k16_scored/summary.json \
  results/code/kae_seed420_k16_scored/summary.json \
  --output results/code/kae_k16_aggregate
```

Include all methods' summaries to build one table. Outputs contain per-run
results, means, SDs and SEs. Greedy and K16 are aggregated separately. The
aggregator checks data/protocol/scorer hashes and all three training seeds.
The prepared datasets are not redistributed in this code repository; their
source versions and licenses must accompany the separate data release.
