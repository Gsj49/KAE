#!/usr/bin/env python3
"""CPU-only bandwidth comparison using rewards in existing rollout parquets."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "assumption6"))
sys.path.insert(0, str(HERE.parents[1] / "train_scripts"))
from data_paths import data_path
from validate_rollout_pool import get_uid, sha256
from estimators import (
    ALPHAS, BETAS, bandwidth, ema_weighted_mean_with_current_weight,
    linear_weighted_mean_with_current_weight, loo_baselines_from_clean_mean,
)


def load_pool(path, prompts, samples):
    # Response strings are not needed for value estimation.
    frame = pd.read_parquet(path, columns=["extra_info", "rewards"])
    uids = frame.extra_info.map(get_uid)
    if len(frame) != prompts or uids.isna().any() or uids.nunique() != prompts:
        raise ValueError(f"{path}: expected {prompts} unique, explicit prompt IDs")
    result = {}
    for uid, rewards in zip(uids, frame.rewards):
        values = np.asarray(rewards, dtype=np.float64)
        if values.shape != (samples,) or not np.isfinite(values).all():
            raise ValueError(f"{path}: invalid reward count/values for {uid}")
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"{path}: rewards outside [0,1] for {uid}")
        result[uid] = values.astype(np.float32)
    return result


def metric_row(uid, method, family, parameter, estimates, reference, step):
    estimates = np.asarray(estimates, dtype=np.float32)
    if not np.isfinite(estimates).all():
        raise ValueError(f"Nonfinite estimates for {uid}/{method}")
    center = float(estimates.mean())
    width, h = bandwidth(family, parameter, step) if parameter is not None else (None, None)
    return dict(uid=uid, method=method, family=family, parameter=parameter,
                bandwidth_H=width, bandwidth_h_i=h, reference_value=reference,
                mean_estimate=center, bias=center - reference,
                variance=float(((estimates - center) ** 2).mean()),
                mse=float(((estimates - reference) ** 2).mean()))


def analyze(pools, step, repeats=100, seed=1234, oracle=None):
    """Compare kernels using paired draws in prompt-ID and repeat order."""
    if repeats < 1:
        raise ValueError("repeats must be positive")
    current = pools[-1]
    uids = sorted(current)
    if not uids or len(pools) != 5 or any(set(pool) != set(uids) for pool in pools):
        raise ValueError("Target and all four historical pools must have identical UID sets")
    if oracle is not None and set(oracle) != set(uids):
        raise ValueError("Reference pool must have the same UID set")
    rng = np.random.default_rng(seed)
    references = np.empty(len(uids), dtype=np.float32)
    reference_idx = np.empty((len(uids), 100), dtype=np.int64)
    current_idx = np.empty((len(uids), repeats, 8), dtype=np.int64)
    history_idx = np.empty((len(uids), repeats, 4, 8), dtype=np.int64)
    current_draws = np.empty((len(uids), repeats, 8), dtype=np.float32)
    rows = []
    for i, uid in enumerate(uids):
        values = current[uid]
        if oracle is None:
            ref_idx = rng.choice(values.size, size=100, replace=False)
            reference = float(values[ref_idx].mean())
            available = np.flatnonzero(~np.isin(np.arange(values.size), ref_idx))
        else:
            ref_idx = np.arange(100)
            reference = float(oracle[uid].mean())
            available = np.arange(values.size)
        if available.size < 8:
            raise ValueError(f"{uid}: fewer than eight non-reference rewards")
        references[i] = reference
        reference_idx[i] = ref_idx
        estimates = {"GRPO": []}
        specs = [("triangular", a, linear_weighted_mean_with_current_weight) for a in ALPHAS]
        specs += [("exponential", b, ema_weighted_mean_with_current_weight) for b in BETAS]
        for family, parameter, _ in specs:
            estimates[f"{family}:{parameter:g}"] = []
        for r in range(repeats):
            indices = available[rng.choice(available.size, size=8, replace=False)]
            rewards = values[indices]
            current_idx[i, r] = indices
            current_draws[i, r] = rewards
            historical_means = []
            for d, pool in enumerate(pools[:-1]):
                indices = rng.choice(pool[uid].size, size=8, replace=False)
                history_idx[i, r, d] = indices
                historical_means.append(float(pool[uid][indices].mean()))
            current_mean = float(rewards.mean())
            estimates["GRPO"].extend(loo_baselines_from_clean_mean(rewards, current_mean, 1.0))
            for family, parameter, smoother in specs:
                mean, current_weight = smoother(historical_means + [current_mean], parameter)
                estimates[f"{family}:{parameter:g}"].extend(
                    loo_baselines_from_clean_mean(rewards, mean, current_weight))
        rows.append(metric_row(uid, "GRPO", "reference", None, estimates["GRPO"], reference, step))
        for family, parameter, _ in specs:
            name = f"{family}:{parameter:g}"
            rows.append(metric_row(uid, name, family, parameter, estimates[name], reference, step))

    # Remove only this completion from the full B*G statistical batch.
    batch_sums = current_draws.sum(axis=(0, 2), dtype=np.float64)
    batch_count = len(uids) * 8
    for i, uid in enumerate(uids):
        estimates = ((batch_sums[:, None] - current_draws[i]) / (batch_count - 1)).reshape(-1)
        rows.append(metric_row(uid, "REINFORCE++", "reference", None, estimates, float(references[i]), step))
    frame = pd.DataFrame(rows)
    if frame.duplicated(["uid", "method"]).any() or len(frame) != len(uids) * 22:
        raise ValueError("Incomplete or duplicate prompt/method coverage")
    draws = dict(uids=np.asarray(uids, dtype=str), reference_indices=reference_idx,
                 current_indices=current_idx, history_indices=history_idx)
    return frame, draws


def summarize(frame):
    rows = []
    for method, group in frame.groupby("method", sort=False):
        meta = group.iloc[0]
        values = group.mse.to_numpy(dtype=float)
        sd = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        row = {key: (None if pd.isna(meta[key]) else meta[key]) for key in
               ("family", "parameter", "bandwidth_H", "bandwidth_h_i")}
        row.update(method=method, prompts=len(values), mean_mse=float(values.mean()),
                   sd_mse=sd, se_mse=sd / np.sqrt(len(values)),
                   mean_bias=float(group.bias.mean()), mean_variance=float(group.variance.mean()))
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("math", "gsm8k"), required=True)
    parser.add_argument("--step", type=int, choices=(10, 90, 170), required=True)
    parser.add_argument("--rollouts", type=Path)
    parser.add_argument("--reference-pool", type=Path,
                        help="Optional independent 100-response target-step pool; no silent fallback")
    parser.add_argument("--repeats", type=int, default=100, help="Paper default: 100 repetitions")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    root = args.rollouts or data_path("rollouts", args.dataset)
    steps = list(range(args.step - 4, args.step + 1))
    paths = [root / f"{args.dataset}_generation_k10_step_{step}_batchTRACE.parquet" for step in steps]
    pools = [load_pool(path, 256, 128) for path in paths]
    oracle = None if args.reference_pool is None else load_pool(args.reference_pool, 256, 100)
    inputs = [dict(path=str(path.resolve()), step=step, sha256=sha256(path))
              for path, step in zip(paths, steps)]
    if args.reference_pool is not None:
        inputs.append(dict(path=str(args.reference_pool.resolve()), step=args.step,
                           role="independent_reference", sha256=sha256(args.reference_pool)))
    frame, draws = analyze(pools, args.step, args.repeats, args.seed, oracle)
    args.output.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(args.output / "prompt_metrics.parquet", index=False)
    np.savez_compressed(args.output / "draw_indices.npz", **draws)
    summary = dict(dataset=args.dataset, step=args.step, num_prompts=256,
                   config=dict(repeats=args.repeats, seed=args.seed, group_size=8,
                               reference_n=100, history_steps=steps[:-1],
                               sampling="without_replacement_within_each_draw",
                               reference_source="independent" if oracle is not None else "held_out_target_subset",
                               paired_kernel_draws=True, linear_alphas=ALPHAS, exponential_betas=BETAS,
                               exponential_initialization="oldest_history_group_mean",
                               rfpp_leave_one_out=True, bootstrap_replicates=0),
                   inputs=inputs, method_summary=summarize(frame))
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    pd.DataFrame(summary["method_summary"]).to_csv(args.output / "summary.csv", index=False)
    outputs = {name: sha256(args.output / name) for name in
               ("prompt_metrics.parquet", "draw_indices.npz", "summary.json", "summary.csv")}
    receipt = dict(status="complete", dataset=args.dataset, step=args.step, prompts=256,
                   rows=len(frame), methods=22, outputs=outputs,
                   code_sha256={name: sha256(HERE / name) for name in ("run.py", "estimators.py")})
    (args.output / "complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("KERNEL_SENSITIVITY_DONE", json.dumps(receipt))


if __name__ == "__main__":
    main()
