#!/usr/bin/env python3
"""Strict validation and a compact manifest for saved rollout pools."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=256)
    parser.add_argument("--expected-samples", type=int, default=128)
    parser.add_argument("--expected-step", type=int, required=True)
    parser.add_argument("--summary", type=Path)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_list(value: Any) -> list[Any]:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return []


def get_uid(value: Any) -> str | None:
    mapping: Any = value
    if isinstance(value, str):
        mapping = None
        for loader in (json.loads, ast.literal_eval):
            try:
                parsed = loader(value)
            except (ValueError, SyntaxError, json.JSONDecodeError):
                continue
            if isinstance(parsed, dict):
                mapping = parsed
                break
    if not isinstance(mapping, dict) or mapping.get("index") is None:
        return None
    return str(mapping["index"])


def main() -> None:
    args = parse_args()
    if not args.input.is_file() or args.input.stat().st_size == 0:
        raise FileNotFoundError(f"Missing or empty rollout pool: {args.input}")
    frame = pd.read_parquet(args.input)
    required = {"responses", "rewards", "extra_info"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"{args.input}: missing columns {missing}")
    if len(frame) != args.expected_rows:
        raise RuntimeError(
            f"{args.input}: rows={len(frame)}, expected {args.expected_rows}"
        )

    response_counts = frame["responses"].map(lambda value: len(as_list(value)))
    reward_counts = frame["rewards"].map(lambda value: len(as_list(value)))
    if not bool((response_counts == args.expected_samples).all()):
        raise RuntimeError(
            f"{args.input}: response counts range "
            f"[{response_counts.min()}, {response_counts.max()}], expected {args.expected_samples}"
        )
    if not bool((reward_counts == args.expected_samples).all()):
        raise RuntimeError(
            f"{args.input}: reward counts range "
            f"[{reward_counts.min()}, {reward_counts.max()}], expected {args.expected_samples}"
        )

    rewards = np.asarray(
        [float(item) for values in frame["rewards"] for item in as_list(values)],
        dtype=np.float64,
    )
    if rewards.size != args.expected_rows * args.expected_samples:
        raise RuntimeError(f"{args.input}: unexpected flattened reward count {rewards.size}")
    if not bool(np.isfinite(rewards).all()):
        raise RuntimeError(f"{args.input}: rewards contain NaN or Inf")
    if bool(((rewards < 0.0) | (rewards > 1.0)).any()):
        raise RuntimeError(f"{args.input}: rewards fall outside [0, 1]")

    uids = frame["extra_info"].map(get_uid)
    if bool(uids.isna().any()) or int(uids.nunique()) != args.expected_rows:
        raise RuntimeError(f"{args.input}: prompt UIDs are missing or duplicated")

    result = {
        "input": str(args.input),
        "step": args.expected_step,
        "rows": int(len(frame)),
        "samples_per_prompt": args.expected_samples,
        "total_completions": int(rewards.size),
        "reward_mean": float(rewards.mean()),
        "reward_std": float(rewards.std(ddof=1)),
        "reward_min": float(rewards.min()),
        "reward_max": float(rewards.max()),
        "sha256": sha256(args.input),
    }
    if not all(math.isfinite(value) for key, value in result.items() if key.startswith("reward_")):
        raise RuntimeError(f"{args.input}: non-finite summary")
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.summary.with_name(f".{args.summary.name}.tmp-{os.getpid()}")
        tmp.write_text(rendered, encoding="utf-8")
        os.replace(tmp, args.summary)
    print(rendered, end="")


if __name__ == "__main__":
    main()
