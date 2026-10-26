#!/usr/bin/env python3
"""Check reward-only inputs, optionally comparing six calculations to full pools."""

import argparse
import json
from pathlib import Path
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd

from prepare_reward_assets import ROOT, STEPS, reward_table, sha256
from run import METHODS, run


def calculate(root, dataset, step):
    return run(SimpleNamespace(
        data_name=dataset, target_step=step, data_root=str(root), k=10,
        steps_glob=str(root / dataset / f"{dataset}_generation_k10_step_*_batchTRACE.parquet"),
        sample_tag="batchTRACE", oracle_steps_glob=None, oracle_sample_tag="",
        history_window=4, repeats=100, true_n=100, eval_n=8, history_n=8,
        kae_alpha=0.2, kalman_q=1e-5, kalman_r=1e-2,
        kalman_initial_mean=0.0, kalman_initial_variance=1.0,
        n_bootstrap=0, seed=1234))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact-root", type=Path, default=ROOT / "data/value_mse")
    parser.add_argument("--full-root", type=Path,
                        help="If supplied, also verify source hashes and exact MSE equivalence")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest_path = args.compact_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    expected = {(dataset, step) for dataset in ("gsm8k", "math") for step in STEPS}
    records = manifest["assets"]
    if manifest["status"] != "complete" or len(records) != 30 or {
        (record["dataset"], record["step"]) for record in records
    } != expected:
        raise ValueError("Incomplete reward asset manifest")
    uid_sets = {}
    for record in records:
        relative = Path(record["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid manifest path")
        path = args.compact_root / relative
        if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
            raise ValueError(f"Compact size/hash mismatch: {relative}")
        table, digest, uids = reward_table(path)
        if digest != record["ordered_rewards_sha256"]:
            raise ValueError(f"Compact ordered rewards mismatch: {relative}")
        uid_sets[record["dataset"], record["step"]] = uids
        if args.full_root is not None:
            source = args.full_root / relative
            if source.stat().st_size != record["source_bytes"] or sha256(source) != record["source_sha256"]:
                raise ValueError(f"Full source size/hash mismatch: {relative}")
            original, original_digest, _ = reward_table(source)
            if not original.equals(table) or digest != original_digest:
                raise ValueError(f"Source projection mismatch: {relative}")
    for dataset in ("gsm8k", "math"):
        for target in (10, 90, 170):
            if any(uid_sets[dataset, target] != uid_sets[dataset, step]
                   for step in range(target - 4, target + 1)):
                raise ValueError(f"History UID coverage differs: {dataset}/{target}")
    comparisons = []
    if args.full_root is not None:
        for dataset in ("gsm8k", "math"):
            for step in (10, 90, 170):
                started = time.monotonic()
                original, original_summary = calculate(args.full_root, dataset, step)
                compact, compact_summary = calculate(args.compact_root, dataset, step)
                pd.testing.assert_frame_equal(original, compact, check_exact=True)
                if len(compact) != 256 * len(METHODS) or compact.uid.nunique() != 256:
                    raise ValueError("Incomplete value-MSE coverage")
                if compact_summary["num_excluded_prompts"] or original_summary["num_excluded_prompts"]:
                    raise ValueError("Unexpected excluded prompts")
                numeric = compact[["bias", "variance", "mse"]].to_numpy()
                if not np.isfinite(numeric).all() or (compact[["variance", "mse"]] < 0).any().any():
                    raise ValueError("Invalid value-MSE output")
                record = dict(dataset=dataset, step=step, prompts=256, methods=len(METHODS),
                              repeats=100, rows=len(compact), exact_equal=True,
                              max_absolute_metric_difference=float(np.abs(
                                  original[["bias", "variance", "mse"]].to_numpy() - numeric).max()),
                              wall_seconds=time.monotonic() - started)
                comparisons.append(record)
                print(json.dumps(record), flush=True)
    receipt = dict(status="complete", files=30, prompt_step_rows=30 * 256,
                   total_individual_rewards=30 * 256 * 128,
                   all_ordered_rewards_unchanged=args.full_root is not None,
                   manifest_sha256=sha256(manifest_path), comparisons=comparisons,
                   code_sha256={name: sha256(Path(__file__).parent / name)
                                for name in ("run.py", "baseline_estimators.py",
                                             "prepare_reward_assets.py", "validate_reward_assets.py")})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print("REWARD_ASSETS_VALIDATED", str(args.output), flush=True)


if __name__ == "__main__":
    main()
