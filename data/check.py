#!/usr/bin/env python3
"""Check whether configured data files exist and are nonempty."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_scripts"))
from data_paths import data_path, data_root


def assets():
    rows = []
    for dataset in ("dapo", "math", "gsm8k"):
        rows.append(dict(group="training", name=f"{dataset}/train", path=data_path("training", dataset)))
    for dataset in ("math", "gsm8k"):
        rows.append(dict(group="training", name=f"{dataset}/test", path=data_path("validation", "single_stream", dataset)))
    rows.append(dict(group="math-eval", name="six-benchmark mathematics evaluation", path=data_path("math_evaluation")))
    for benchmark in ("humaneval_plus", "mbpp_plus"):
        rows.append(dict(group="code", name=benchmark,
                         path=data_path("code") / benchmark / "test.parquet"))
    for dataset in ("gsm8k", "math"):
        for target in (10, 90, 170):
            for step in range(target - 4, target + 1):
                rows.append(dict(group="rollouts", name=f"{dataset}/step{step}",
                                 path=data_path("rollouts", dataset) / f"{dataset}_generation_k10_step_{step}_batchTRACE.parquet"))
    for dataset in ("math", "gsm8k"):
        rows.append(dict(group="traces", name=f"{dataset}/training trace", path=data_path("traces", dataset)))
    rows.append(dict(group="one-shot", name="one-shot copies", path=data_path("one_shot", "prepared")))
    return rows


def inspect(record):
    path = record["path"]
    result = dict(record, path=str(path))
    if not path.is_file():
        result["status"] = "missing"
    elif path.stat().st_size == 0:
        result["status"] = "empty"
    else:
        result.update(status="present", bytes=path.stat().st_size)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--group", choices=("all", "training", "math-eval", "code", "rollouts", "traces", "one-shot"), default="all")
    parser.add_argument("--strict", action="store_true", help="Exit nonzero if any selected file is absent or empty")
    args = parser.parse_args()
    results = [inspect(row) for row in assets() if args.group in ("all", row["group"])]
    print(json.dumps(dict(data_root=str(data_root()), assets=results), indent=2))
    if args.strict and any(row["status"] != "present" for row in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
