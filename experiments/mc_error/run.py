#!/usr/bin/env python3
"""Recreate the appendix Monte Carlo standard-error table on CPU."""

import argparse
import csv
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ("gsm8k", "math")
STEPS = (10, 90, 170)
REFERENCE_N = 100


def read_counts(path):
    """Read per-prompt success counts, not precomputed standard errors."""
    with Path(path).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key in ("step", "n", "successes"):
            row[key] = int(row[key])
    return rows


def counts_from_pools(root):
    """Use the value-MSE experiment's original 100-of-128 reference split."""
    import numpy as np
    import pandas as pd

    rows = []
    for dataset in DATASETS:
        for step in STEPS:
            path = Path(root) / dataset / f"{dataset}_generation_k10_step_{step}_batchTRACE.parquet"
            frame = pd.read_parquet(path, columns=["extra_info", "rewards"])
            pools = {}
            for extra, rewards in zip(frame.extra_info, frame.rewards):
                if not isinstance(extra, dict) or extra.get("index") is None:
                    raise ValueError(f"Missing prompt identifier in {path}")
                uid = str(extra["index"])
                values = np.asarray(rewards, dtype=float)
                if uid in pools or values.shape != (128,) or not np.isin(values, [0, 1]).all():
                    raise ValueError(f"Invalid or duplicate reward pool: {path}, uid={uid}")
                pools[uid] = values
            if len(pools) != 256:
                raise ValueError(f"Expected 256 prompts in {path}")
            rng = np.random.default_rng(1234)
            for uid in sorted(pools):
                indices = rng.choice(128, size=REFERENCE_N, replace=False)
                rows.append(dict(dataset=dataset, step=step, uid=uid, n=REFERENCE_N,
                                 successes=int(pools[uid][indices].sum())))
    return rows


def quantile(values, probability):
    """Linear interpolation, matching the original numpy.quantile calculation."""
    values = sorted(values)
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    return values[lower] + (position - lower) * (values[upper] - values[lower])


def summarize(rows, prompts_per_step=256):
    groups = {(dataset, step): [] for dataset in DATASETS for step in STEPS}
    seen = set()
    for row in rows:
        group = (row["dataset"], row["step"])
        key = (*group, row["uid"])
        if group not in groups or key in seen or not str(row["uid"]).strip():
            raise ValueError(f"Unexpected or duplicate prompt: {key}")
        n, successes = row["n"], row["successes"]
        if n != REFERENCE_N or not isinstance(successes, int) or not 0 <= successes <= n:
            raise ValueError(f"Invalid binomial counts: {key}")
        seen.add(key)
        value = successes / n
        groups[group].append(math.sqrt(value * (1 - value) / n))
    output = []
    for (dataset, step), errors in groups.items():
        if len(errors) != prompts_per_step:
            raise ValueError(f"Expected {prompts_per_step} prompts for {dataset}, step {step}")
        output.append(dict(dataset=dataset, step=step, prompts=len(errors), reference_n=REFERENCE_N,
                           mean_mc_se=statistics.fmean(errors), median_mc_se=statistics.median(errors),
                           p90_mc_se=quantile(errors, 0.9)))
    return output


def write_table(rows, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "reference_mc_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Dataset & Step & Prompts & Mean SE & Median SE & 90\% quantile \\", r"\midrule"]
    for row in rows:
        lines.append(f"{row['dataset'].upper()} & {row['step']} & {row['prompts']} & "
                     f"{row['mean_mc_se']:.5f} & {row['median_mc_se']:.5f} & {row['p90_mc_se']:.5f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "reference_mc_table.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--input", type=Path, help="Per-prompt reference counts CSV")
    source.add_argument("--rollouts", type=Path, help="Saved pools root containing gsm8k/ and math/")
    parser.add_argument("--output", type=Path, default=ROOT / "results/mc_error")
    args = parser.parse_args()
    counts = counts_from_pools(args.rollouts) if args.rollouts else read_counts(
        args.input or ROOT / "data/mc_error/reference_counts.csv")
    rows = summarize(counts)
    write_table(rows, args.output)
    print(f"Wrote MC standard-error table: {args.output}")


if __name__ == "__main__":
    main()
