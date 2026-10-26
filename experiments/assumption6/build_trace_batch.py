#!/usr/bin/env python3
"""Materialize the exact prompt batch recorded for one training step."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--train-parquet", type=Path, required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-rows", type=int, default=256)
    parser.add_argument("--expected-sticky-k", type=int, default=10)
    return parser.parse_args()


def normalize_mapping(value: Any) -> dict[str, Any] | None:
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
    mapping = normalize_mapping(value)
    if mapping is None or mapping.get("index") is None:
        return None
    return str(mapping["index"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = parse_args()
    if not args.trace.is_file():
        raise FileNotFoundError(args.trace)
    if not args.train_parquet.is_file():
        raise FileNotFoundError(args.train_parquet)

    matches: list[dict[str, Any]] = []
    with args.trace.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if int(record.get("global_step", -1)) == args.step:
                matches.append(record)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one trace record for step={args.step}, found {len(matches)}"
        )

    record = matches[0]
    sticky_k = int(record.get("sticky_k", -1))
    if sticky_k != args.expected_sticky_k:
        raise RuntimeError(
            f"step={args.step}: sticky_k={sticky_k}, expected {args.expected_sticky_k}"
        )
    uid_list = [str(value) for value in record.get("uid_list", [])]
    if len(uid_list) != args.expected_rows:
        raise RuntimeError(
            f"step={args.step}: trace has {len(uid_list)} uids, expected {args.expected_rows}"
        )
    if len(set(uid_list)) != len(uid_list):
        raise RuntimeError(f"step={args.step}: trace uid_list contains duplicates")

    frame = pd.read_parquet(args.train_parquet)
    if "extra_info" not in frame.columns:
        raise RuntimeError("train parquet is missing the extra_info column")
    frame = frame.copy()
    frame["__trace_uid"] = frame["extra_info"].map(get_uid)
    frame = frame[frame["__trace_uid"].notna()]
    frame = frame.drop_duplicates(subset="__trace_uid", keep="first")
    frame = frame.set_index("__trace_uid", drop=False)
    missing = [uid for uid in uid_list if uid not in frame.index]
    if missing:
        raise RuntimeError(
            f"step={args.step}: {len(missing)} traced prompts are absent; examples={missing[:5]}"
        )

    ordered = frame.loc[uid_list].reset_index(drop=True).drop(columns=["__trace_uid"])
    if len(ordered) != args.expected_rows:
        raise RuntimeError(
            f"step={args.step}: materialized {len(ordered)} rows, expected {args.expected_rows}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    ordered.to_parquet(tmp, index=False)
    os.replace(tmp, args.output)
    print(
        json.dumps(
            {
                "step": args.step,
                "rows": len(ordered),
                "sticky_k": sticky_k,
                "repeat_idx_in_k": record.get("repeat_idx_in_k"),
                "output": str(args.output),
                "sha256": sha256(args.output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
