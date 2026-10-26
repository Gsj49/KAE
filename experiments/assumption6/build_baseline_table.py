#!/usr/bin/env python3
"""Build per-prompt baseline means and mean squared errors from saved rewards."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rollout-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=["gsm8k", "math"], required=True)
    parser.add_argument("--target-step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--history-window", type=int, default=4)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--sample-tag", default="batchTRACE")
    parser.add_argument("--oracle-n", type=int, default=100)
    parser.add_argument("--eval-n", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=1234)
    return parser.parse_args()


def parse_mapping(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(value)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                return parsed
    return None


def get_uid(value: Any) -> str | None:
    mapping = parse_mapping(value)
    if mapping is None or mapping.get("index") is None:
        return None
    return str(mapping["index"])


def to_rewards(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        values = value.tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
    else:
        values = []
    rewards = np.asarray([float(item) for item in values], dtype=np.float64)
    if rewards.size == 0 or not bool(np.isfinite(rewards).all()):
        raise ValueError("reward list is empty or non-finite")
    return rewards


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rollout_path(args: argparse.Namespace, step: int) -> Path:
    return args.rollout_dir / (
        f"{args.dataset}_generation_k{args.k}_step_{step}_{args.sample_tag}.parquet"
    )


def load_reward_map(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(path)
    frame = pd.read_parquet(path)
    if "extra_info" not in frame.columns or "rewards" not in frame.columns:
        raise ValueError(f"{path}: expected extra_info and rewards columns")
    uids = frame["extra_info"].map(get_uid)
    if bool(uids.isna().any()) or int(uids.nunique()) != len(frame):
        raise ValueError(f"{path}: missing or duplicate prompt UIDs")
    return {uid: to_rewards(reward) for uid, reward in zip(uids, frame["rewards"])}


def sample_without_replacement(
    values: np.ndarray, count: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if values.size < count:
        raise ValueError(f"Need {count} samples, found {values.size}")
    indices = rng.choice(values.size, size=count, replace=False)
    return values[indices], indices


def linear_weighted_mean(values: list[float], alpha: float) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    distance = (array.size - 1) - np.arange(array.size, dtype=np.float64)
    weights = np.maximum(1.0 - alpha * distance, 0.0)
    valid = np.isfinite(array) & (weights > 0.0)
    if not bool(valid.any()):
        return np.nan, np.nan
    normalized = weights[valid] / weights[valid].sum()
    return float(np.dot(array[valid], normalized)), float(normalized[-1])


def loo_baselines(
    rewards: np.ndarray, clean_mean: float, current_step_weight: float
) -> np.ndarray:
    denominator = 1.0 - current_step_weight / rewards.size
    if rewards.size < 2 or not np.isfinite(clean_mean) or denominator <= 1e-12:
        raise ValueError("Cannot construct leave-one-out baseline")
    return (
        (clean_mean - (current_step_weight / rewards.size) * rewards) / denominator
    )


def baseline_moments(values: Any, oracle_value: float) -> dict[str, float | int]:
    """Average over all repetitions and completion-level LOO estimates."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("Baseline samples must be a nonempty finite vector")
    if not np.isfinite(oracle_value):
        raise ValueError("Reference value must be finite")
    mean = float(values.mean())
    return {
        "baseline_value": mean,
        "oracle_value": float(oracle_value),
        "baseline_mse": float(np.mean((values - oracle_value) ** 2)),
        "baseline_variance": float(np.mean((values - mean) ** 2)),
        "baseline_samples": int(values.size),
    }


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("alpha must lie in [0, 1]")
    if args.target_step <= args.history_window:
        raise ValueError("target step must exceed the history window")
    if args.eval_n < 2 or args.repeats < 1:
        raise ValueError("eval-n must be >=2 and repeats must be >=1")

    history_steps = list(range(args.target_step - args.history_window, args.target_step))
    paths = {step: rollout_path(args, step) for step in history_steps + [args.target_step]}
    reward_maps = {step: load_reward_map(path) for step, path in paths.items()}
    common_uids = set(reward_maps[args.target_step])
    for reward_map in reward_maps.values():
        common_uids.intersection_update(reward_map)
    uids = sorted(common_uids)
    expected_prompts = len(reward_maps[args.target_step])
    if len(uids) != expected_prompts:
        raise ValueError(
            f"Only {len(uids)}/{expected_prompts} target prompts align across history steps"
        )

    rng = np.random.default_rng(args.seed)
    oracle_values: dict[str, float] = {}
    grpo_values: dict[str, list[float]] = {uid: [] for uid in uids}
    kae_values: dict[str, list[float]] = {uid: [] for uid in uids}
    current_means: dict[str, list[float]] = {uid: [] for uid in uids}
    current_samples: dict[str, list[np.ndarray]] = {uid: [] for uid in uids}

    for uid in uids:
        target = reward_maps[args.target_step][uid]
        if target.size < args.oracle_n + args.eval_n:
            raise ValueError(
                f"uid={uid}: target pool has {target.size}, need at least "
                f"oracle_n + eval_n = {args.oracle_n + args.eval_n}"
            )
        oracle_sample, oracle_indices = sample_without_replacement(target, args.oracle_n, rng)
        oracle_values[uid] = float(oracle_sample.mean())
        keep = np.ones(target.size, dtype=bool)
        keep[oracle_indices] = False
        estimation_pool = target[keep]

        for _ in range(args.repeats):
            current_rewards, _ = sample_without_replacement(estimation_pool, args.eval_n, rng)
            current_mean = float(current_rewards.mean())
            current_means[uid].append(current_mean)
            current_samples[uid].append(current_rewards)

            history_means: list[float] = []
            for step in history_steps:
                sampled, _ = sample_without_replacement(
                    reward_maps[step][uid], args.eval_n, rng
                )
                history_means.append(float(sampled.mean()))

            grpo_values[uid].extend(
                loo_baselines(current_rewards, current_mean, current_step_weight=1.0).tolist()
            )
            kae_mean, current_weight = linear_weighted_mean(
                history_means + [current_mean], args.alpha
            )
            kae_values[uid].extend(
                loo_baselines(current_rewards, kae_mean, current_weight).tolist()
            )

    rfpp_by_repeat = [
        float(np.mean([current_means[uid][repeat] for uid in uids]))
        for repeat in range(args.repeats)
    ]
    rows: list[dict[str, Any]] = []
    batch_completions = len(uids) * args.eval_n
    for uid in uids:
        method_values = {
            f"KAE(alpha={args.alpha:g})": kae_values[uid],
            "GRPO": grpo_values[uid],
            "REINFORCE++": np.concatenate([
                (batch_completions * mean - current_samples[uid][repeat]) / (batch_completions - 1)
                for repeat, mean in enumerate(rfpp_by_repeat)
            ]),
        }
        for method, values in method_values.items():
            rows.append(
                {
                    "uid": uid,
                    "method": method,
                    **baseline_moments(values, oracle_values[uid]),
                    "target_step": args.target_step,
                }
            )

    output = pd.DataFrame(rows)
    if output.duplicated(["uid", "method"]).any():
        raise RuntimeError("Duplicate uid/method output rows")
    numeric = output[["baseline_value", "oracle_value"]].to_numpy(dtype=np.float64)
    if not bool(np.isfinite(numeric).all()):
        raise RuntimeError("Output contains non-finite baseline values")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    output.to_parquet(tmp, index=False)
    os.replace(tmp, args.output)
    summary_path = args.output.with_suffix(".summary.json")
    summary = {
        "dataset": args.dataset,
        "target_step": args.target_step,
        "history_steps": history_steps,
        "num_prompts": len(uids),
        "methods": sorted(output["method"].unique().tolist()),
        "oracle_n": args.oracle_n,
        "eval_n": args.eval_n,
        "repeats": args.repeats,
        "alpha": args.alpha,
        "seed": args.seed,
        "baseline_samples_per_prompt": args.repeats * args.eval_n,
        "baseline_mse_definition": "mean((baseline_sample - oracle_value)**2)",
        "rollout_files": {str(step): {"path": str(path), "sha256": sha256(path)} for step, path in paths.items()},
        "output": str(args.output),
        "output_sha256": sha256(args.output),
    }
    tmp_summary = summary_path.with_name(f".{summary_path.name}.tmp-{os.getpid()}")
    tmp_summary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp_summary, summary_path)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
