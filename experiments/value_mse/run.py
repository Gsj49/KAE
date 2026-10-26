#!/usr/bin/env python3
"""Compare value-baseline MSE, including Kalman and shrinkage baselines.

This preserves the previous experiment's core protocol: an optional independent
oracle pool estimates the prompt value; repeated groups are drawn without
replacement from the target and history-step pools. REINFORCE++, GRPO and KAE
use completion-level leave-one-out corrections. REINFORCE++ uses all prompts
in the batch, excluding only the completion whose advantage is computed.
"""

from __future__ import annotations

import argparse
import ast
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from baseline_estimators import (
    global_batch_leave_one_out,
    kalman_posterior_baselines,
    leave_one_out_mean,
    shrinkage_baselines,
    triangular_kae_baselines,
)


METHODS = ["Naive", "REINFORCE++", "GRPO", "KAE", "Kalman", "Shrinkage"]


def parse_extra_info_index(value: Any) -> str | None:
    if isinstance(value, dict):
        index = value.get("index")
        return None if index is None else str(index)
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(value)
            except (ValueError, SyntaxError):
                return None
        if isinstance(parsed, dict) and parsed.get("index") is not None:
            return str(parsed["index"])
    return None


def to_rewards(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        raw = value.tolist()
    elif isinstance(value, list):
        raw = value
    else:
        return np.array([], dtype=np.float64)
    cleaned: list[float] = []
    for item in raw:
        if item is None:
            continue
        try:
            cleaned.append(float(item))
        except (TypeError, ValueError):
            continue
    return np.asarray(cleaned, dtype=np.float64)


def load_generation(path: str) -> dict[str, np.ndarray]:
    frame = pd.read_parquet(path)
    if "rewards" not in frame.columns:
        raise ValueError(f"{path} has no rewards column")
    uid: pd.Series | None = None
    if "extra_info" in frame.columns:
        uid = frame["extra_info"].map(parse_extra_info_index)
    if uid is None or uid.isna().all():
        for candidate in ("uid", "id", "qid"):
            if candidate in frame.columns:
                uid = frame[candidate].astype(str)
                break
    if uid is None:
        uid = pd.Series(np.arange(len(frame)), index=frame.index).astype(str)
    result: dict[str, np.ndarray] = {}
    for prompt_id, rewards in zip(uid.astype(str), frame["rewards"].map(to_rewards)):
        result[prompt_id] = rewards
    return result


def step_from_path(path: str) -> int | None:
    match = re.search(r"step[_-]?(\d+)", Path(path).name)
    return int(match.group(1)) if match else None


def load_steps(pattern: str) -> dict[int, dict[str, np.ndarray]]:
    result: dict[int, dict[str, np.ndarray]] = {}
    for path in sorted(glob.glob(pattern)):
        step = step_from_path(path)
        if step is not None:
            result[step] = load_generation(path)
    if not result:
        raise FileNotFoundError(f"No step files matched: {pattern}")
    return result


def sample_exact(rng: np.random.Generator, values: np.ndarray, n: int) -> np.ndarray:
    if values.size < n:
        raise ValueError(f"Need {n} samples but only {values.size} are available")
    indices = rng.choice(values.size, size=n, replace=False)
    return values[indices]


def prompt_stats(estimates: list[float], truth: float) -> tuple[float, float, float]:
    values = np.asarray(estimates, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan, math.nan
    bias = float(values.mean() - truth)
    variance = float(np.mean((values - values.mean()) ** 2))
    return bias, variance, float(np.mean((values - truth) ** 2))


def bootstrap_ci(values: np.ndarray, rng: np.random.Generator, n_bootstrap: int) -> tuple[float, float]:
    if n_bootstrap <= 0:
        return float("nan"), float("nan")
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    draws = rng.choice(values, size=(n_bootstrap, values.size), replace=True).mean(axis=1)
    return float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def default_pattern(args: argparse.Namespace, sample_tag: str) -> str:
    return str(
        Path(args.data_root)
        / args.data_name
        / f"{args.data_name}_generation_k{args.k}_step_*_{sample_tag}.parquet"
    )


def run(args: argparse.Namespace) -> tuple[pd.DataFrame, dict[str, Any]]:
    rng = np.random.default_rng(args.seed)
    step_pattern = args.steps_glob or default_pattern(args, args.sample_tag)
    steps = load_steps(step_pattern)
    history_steps = list(range(args.target_step - args.history_window, args.target_step))
    missing_steps = [step for step in history_steps + [args.target_step] if step not in steps]
    if missing_steps:
        raise ValueError(f"Missing required steps {missing_steps}; available steps are {sorted(steps)}")

    oracle: dict[str, np.ndarray] | None = None
    oracle_pattern = args.oracle_steps_glob
    if oracle_pattern is None and args.oracle_sample_tag:
        oracle_pattern = default_pattern(args, args.oracle_sample_tag)
    if oracle_pattern:
        if glob.glob(oracle_pattern):
            oracle_steps = load_steps(oracle_pattern)
            if args.target_step not in oracle_steps:
                raise ValueError(f"Oracle pattern has no target step {args.target_step}")
            oracle = oracle_steps[args.target_step]
        elif args.oracle_steps_glob:
            raise FileNotFoundError(f"No oracle step files matched explicit pattern: {oracle_pattern}")
        else:
            print(f"[info] no optional oracle files matched {oracle_pattern}; splitting the target pool")

    target = steps[args.target_step]
    prompt_ids = set(target)
    for step in history_steps:
        prompt_ids &= set(steps[step])
    if oracle is not None:
        prompt_ids &= set(oracle)

    truth: dict[str, float] = {}
    target_pool: dict[str, np.ndarray] = {}
    excluded: dict[str, str] = {}
    for uid in sorted(prompt_ids):
        target_values = target[uid]
        if oracle is not None:
            if oracle[uid].size == 0:
                excluded[uid] = "empty oracle pool"
                continue
            truth[uid] = float(oracle[uid].mean())
            target_pool[uid] = target_values
        else:
            if target_values.size < args.true_n + args.eval_n:
                excluded[uid] = f"need true_n+eval_n={args.true_n + args.eval_n} target rewards"
                continue
            oracle_indices = rng.choice(target_values.size, size=args.true_n, replace=False)
            truth[uid] = float(target_values[oracle_indices].mean())
            mask = np.ones(target_values.size, dtype=bool)
            mask[oracle_indices] = False
            target_pool[uid] = target_values[mask]

        if target_pool[uid].size < args.eval_n:
            excluded[uid] = f"need eval_n={args.eval_n} target rewards"
            truth.pop(uid, None)
            target_pool.pop(uid, None)
            continue
        if any(steps[step][uid].size < args.history_n for step in history_steps):
            excluded[uid] = f"need history_n={args.history_n} rewards at every history step"
            truth.pop(uid, None)
            target_pool.pop(uid, None)

    valid_uids = sorted(truth)
    if len(valid_uids) < 2:
        raise ValueError("Fewer than two prompts have sufficient target/history/oracle samples")

    estimates: dict[str, dict[str, list[float]]] = {
        method: {uid: [] for uid in valid_uids} for method in METHODS
    }
    shrinkage_lambdas: list[float] = []
    for _ in range(args.repeats):
        current_groups = {uid: sample_exact(rng, target_pool[uid], args.eval_n) for uid in valid_uids}
        history_means: dict[str, list[float]] = {}
        for uid in valid_uids:
            history_means[uid] = [
                float(sample_exact(rng, steps[step][uid], args.history_n).mean()) for step in history_steps
            ]

        batch_loo = global_batch_leave_one_out(current_groups)
        shrinkage, lambdas = shrinkage_baselines(current_groups)
        shrinkage_lambdas.extend(lambdas.values())
        for uid, rewards in current_groups.items():
            estimates["Naive"][uid].extend(np.zeros(rewards.size).tolist())
            estimates["REINFORCE++"][uid].extend(batch_loo[uid].tolist())
            estimates["GRPO"][uid].extend(leave_one_out_mean(rewards).tolist())
            estimates["KAE"][uid].extend(
                triangular_kae_baselines(rewards, history_means[uid], args.kae_alpha).tolist()
            )
            kalman, _ = kalman_posterior_baselines(
                rewards,
                process_variance=args.kalman_q,
                measurement_variance=args.kalman_r,
                initial_mean=args.kalman_initial_mean,
                initial_variance=args.kalman_initial_variance,
            )
            estimates["Kalman"][uid].extend(kalman.tolist())
            estimates["Shrinkage"][uid].extend(shrinkage[uid].tolist())

    rows: list[dict[str, Any]] = []
    for uid in valid_uids:
        for method in METHODS:
            bias, variance, mse = prompt_stats(estimates[method][uid], truth[uid])
            rows.append(
                {
                    "uid": uid,
                    "method": method,
                    "oracle_value": truth[uid],
                    "mean_baseline": float(np.nanmean(estimates[method][uid])),
                    "bias": bias,
                    "variance": variance,
                    "mse": mse,
                }
            )
    metrics = pd.DataFrame(rows)
    bootstrap_rng = np.random.default_rng(args.seed + 10_000)
    method_summary: dict[str, dict[str, float | int]] = {}
    for method in METHODS:
        subset = metrics.loc[metrics["method"] == method, "mse"].to_numpy(dtype=float)
        low, high = bootstrap_ci(subset, bootstrap_rng, args.n_bootstrap)
        method_summary[method] = {
            "num_prompts": int(np.isfinite(subset).sum()),
            "mean_mse": float(np.nanmean(subset)),
            "median_mse": float(np.nanmedian(subset)),
            "ci95_mean_mse_low": low,
            "ci95_mean_mse_high": high,
        }

    summary: dict[str, Any] = {
        "data_name": args.data_name,
        "target_step": args.target_step,
        "history_steps": history_steps,
        "sample_tag": args.sample_tag,
        "oracle_sample_tag": args.oracle_sample_tag,
        "num_prompts": len(valid_uids),
        "num_excluded_prompts": len(excluded),
        "repeats": args.repeats,
        "eval_n": args.eval_n,
        "history_n": args.history_n,
        "true_n": args.true_n,
        "seed": args.seed,
        "kae_alpha": args.kae_alpha,
        "reinforcepp_baseline": "global_batch_completion_leave_one_out",
        "reinforcepp_batch_prompts": len(valid_uids),
        "reinforcepp_batch_completions": len(valid_uids) * args.eval_n,
        "kalman": {
            "process_variance_q": args.kalman_q,
            "measurement_variance_r": args.kalman_r,
            "initial_mean": args.kalman_initial_mean,
            "initial_variance": args.kalman_initial_variance,
        },
        "shrinkage_lambda_mean": float(np.mean(shrinkage_lambdas)),
        "method_summary": method_summary,
    }
    return metrics, summary


def plot(metrics: pd.DataFrame, output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    values = [metrics.loc[metrics["method"] == method, "mse"].dropna().to_numpy() for method in METHODS]
    figure, axis = plt.subplots(figsize=(8.2, 4.6))
    axis.boxplot(values, tick_labels=METHODS, showfliers=False)
    axis.set_ylabel("Prompt-level baseline MSE")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=240, bbox_inches="tight")
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-name", choices=["math", "gsm8k", "dapo"], required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--data-root", default="data/rollouts")
    parser.add_argument("--k", type=int, default=10, help="Generation-pool K used in the step filenames")
    parser.add_argument("--steps-glob", help="Override the regular step-file glob")
    parser.add_argument("--sample-tag", default="batchTRACE")
    parser.add_argument("--oracle-steps-glob")
    parser.add_argument("--oracle-sample-tag", default="batchTRACE_MSE100")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--true-n", type=int, default=100)
    parser.add_argument("--eval-n", type=int, default=8)
    parser.add_argument("--history-n", type=int, default=8)
    parser.add_argument("--kae-alpha", type=float, default=0.2)
    parser.add_argument("--kalman-q", type=float, default=1e-5)
    parser.add_argument("--kalman-r", type=float, default=1e-2)
    parser.add_argument("--kalman-initial-mean", type=float, default=0.0)
    parser.add_argument("--kalman-initial-variance", type=float, default=1.0)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, summary = run(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.data_name}_step{args.target_step}_revision_mse"
    metrics.to_parquet(output_dir / f"{stem}_prompt_metrics.parquet", index=False)
    metrics.to_csv(output_dir / f"{stem}_prompt_metrics.csv", index=False)
    with (output_dir / f"{stem}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
    plot(metrics, output_dir / f"{stem}.png")
    print(json.dumps(summary["method_summary"], indent=2))
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
