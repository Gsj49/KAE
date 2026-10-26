#!/usr/bin/env python3
"""Recreate the appendix DAPO computing-cost table from saved measurements."""

import argparse
import csv
import gzip
import math
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("grpo", "gpg", "drgrpo", "js", "kalman", "kae")
LABELS = {"grpo": "GRPO", "gpg": "GPG", "drgrpo": r"Dr.~GRPO", "js": "JS", "kalman": "Kalman", "kae": "KAE"}


def read_csv(path):
    opener = gzip.open if Path(path).suffix == ".gz" else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value, positive=False):
    result = float(value)
    if not math.isfinite(result) or result < 0 or (positive and result == 0):
        raise ValueError(f"Invalid resource measurement: {value}")
    return result


def summarize(directory):
    directory = Path(directory)
    runs = read_csv(directory / "runs.csv")
    if len(runs) != len(METHODS) or {r["method"] for r in runs} != set(METHODS):
        raise ValueError("Expected one run for each of the six reported methods")
    runs = {r["method"]: r for r in runs}
    steps = {method: {} for method in METHODS}
    for row in read_csv(directory / "steps.csv"):
        method, step = row["method"], int(row["step"])
        if method not in steps or step in steps[method] or not 1 <= step <= 500:
            raise ValueError(f"Unexpected or duplicate training step: {method}, {step}")
        steps[method][step] = dict(
            seconds=number(row["step_seconds"], positive=True),
            advantage=number(row["advantage_seconds"]),
            tokens=number(row["total_tokens"], positive=True))

    memory = {(method, gpu): [] for method in METHODS for gpu in range(4)}
    for row in read_csv(directory / "gpu_memory.csv.gz"):
        key = (row["method"], int(row["gpu"]))
        if key not in memory or int(row["sample"]) != len(memory[key]) + 1:
            raise ValueError(f"Unexpected GPU or out-of-order memory sample: {key}")
        memory[key].append(number(row["memory_bytes"]))

    output = []
    for method in METHODS:
        run = runs[method]
        if int(run["gpus"]) != 4 or int(run["steps"]) != 500 or set(steps[method]) != set(range(1, 501)):
            raise ValueError(f"Expected four GPUs and all 500 steps for {method}")
        if any(not memory[(method, gpu)] for gpu in range(4)):
            raise ValueError(f"Missing GPU memory samples for {method}")
        records = [steps[method][step] for step in range(1, 501)]
        duration = sum(record["seconds"] for record in records)
        elapsed = number(run["training_loop_seconds"], positive=True)
        if elapsed < duration:
            raise ValueError(f"Training-loop duration shorter than summed training steps: {method}")
        output.append(dict(
            method=method, training_loop_hours=elapsed / 3600,
            training_step_seconds=statistics.fmean(r["seconds"] for r in records),
            advantage_ms=1000 * statistics.fmean(r["advantage"] for r in records),
            tokens_per_second_per_gpu=sum(r["tokens"] for r in records) / (4 * duration),
            sampled_gpu_memory_gib=max(max(memory[(method, gpu)]) for gpu in range(4)) / 1024**3))
    return output


def write_table(rows, output):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    with (output / "computing_cost_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [r"\begin{tabular}{lrrrrr}", r"\toprule",
             r"Method & Time (h) & Step (s) & Advantage (ms) & Tokens/s/GPU & GPU memory (GiB) \\", r"\midrule"]
    for row in rows:
        lines.append(f"{LABELS[row['method']]} & {row['training_loop_hours']:.2f} & "
                     f"{row['training_step_seconds']:.2f} & {row['advantage_ms']:.2f} & "
                     f"{row['tokens_per_second_per_gpu']:.0f} & {row['sampled_gpu_memory_gib']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "computing_cost_table.tex").write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/computing_cost")
    parser.add_argument("--output", type=Path, default=ROOT / "results/computing_cost")
    args = parser.parse_args()
    write_table(summarize(args.input), args.output)
    print(f"Wrote computing-cost table: {args.output}")


if __name__ == "__main__":
    main()
