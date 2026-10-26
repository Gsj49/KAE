#!/usr/bin/env python3
"""Summarize the paper-faithful valid.all evaluation parquet files."""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd


EXPECTED_COUNTS = {
    "aime": 960,
    "aime25": 960,
    "amc": 2656,
    "math": 500,
    "minerva": 272,
    "olympiad_bench": 675,
}
EXPECTED_MULTIPLICITY = {
    "aime": 32,
    "aime25": 32,
    "amc": 32,
    "math": 1,
    "minerva": 1,
    "olympiad_bench": 1,
}
MAIN_SOURCES = ("aime", "aime25", "amc", "math", "minerva", "olympiad_bench")
PASS_AT_32_SOURCES = ("aime", "aime25", "amc")
METRIC_COLUMNS = (
    "aime24",
    "aime25",
    "amc",
    "math",
    "minerva",
    "olympiad_bench",
    "avg",
    "aime24_pass@32",
    "aime25_pass@32",
    "amc_pass@32",
)
FILENAME_RE = re.compile(
    r"^generation_(?P<model>.+)_s(?P<step>\d+)_temperature"
    r"(?P<temperature>[\d.\-eE]+)(?:_evalseed(?P<eval_seed>\d+))?\.parquet$"
)


def reward_scalar(value: object) -> float:
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value, dtype=float)
        if array.size != 1:
            raise ValueError(f"Expected one generated response per valid.all row, found {array.size}")
        result = float(array[0])
    else:
        result = float(value)
    if not math.isfinite(result) or not 0 <= result <= 1:
        raise ValueError(f"Non-finite reward: {result}")
    return result


def sample_index(extra_info: object) -> object:
    if isinstance(extra_info, dict):
        return extra_info.get("index")
    raise ValueError(f"extra_info is not a dict: {type(extra_info).__name__}")


def validate_validall(df: pd.DataFrame, path: Path, strict: bool) -> None:
    required = {"data_source", "responses", "rewards", "extra_info"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")

    response_lengths = df["responses"].map(len)
    reward_lengths = df["rewards"].map(len)
    if not (response_lengths == 1).all() or not (reward_lengths == 1).all():
        raise ValueError(f"{path}: every valid.all row must contain exactly one response and reward")
    if not df["responses"].map(lambda x: isinstance(x[0], str) and bool(x[0].strip())).all():
        raise ValueError(f"{path}: empty or invalid response")

    if not strict:
        return

    counts = df["data_source"].value_counts().to_dict()
    if counts != EXPECTED_COUNTS:
        raise ValueError(f"{path}: task counts differ from valid.all protocol: {counts}")

    for source, expected_multiplicity in EXPECTED_MULTIPLICITY.items():
        subset = df[df["data_source"] == source]
        indices = subset["extra_info"].map(sample_index)
        if indices.isna().any():
            raise ValueError(f"{path}: missing question identity in {source}")
        index_counts = indices.value_counts()
        observed = set(index_counts.astype(int).tolist())
        if observed != {expected_multiplicity}:
            raise ValueError(
                f"{path}: {source} index multiplicities {sorted(observed)}; "
                f"expected {expected_multiplicity}"
            )


def pass_at_32(df: pd.DataFrame, source: str) -> float:
    subset = df[df["data_source"] == source].copy()
    subset["sample_index"] = subset["extra_info"].map(sample_index)
    success = subset.groupby("sample_index")["reward_score"].apply(lambda values: (values > 0).any())
    return float(success.mean())


def summarize_file(path: Path, strict: bool) -> dict[str, object]:
    match = FILENAME_RE.match(path.name)
    if not match:
        raise ValueError(f"Unexpected generation filename: {path.name}")
    df = pd.read_parquet(path)
    validate_validall(df, path, strict=strict)
    df["reward_score"] = df["rewards"].map(reward_scalar)

    source_accuracy = df.groupby("data_source")["reward_score"].mean().to_dict()
    missing_sources = set(MAIN_SOURCES).difference(source_accuracy)
    if strict and missing_sources:
        raise ValueError(f"{path}: missing benchmark sources {sorted(missing_sources)}")

    model = match.group("model")
    train_seed_match = re.search(r"_seed(?P<seed>\d+)$", model)
    train_seed = int(train_seed_match.group("seed")) if train_seed_match else np.nan
    method = model[: train_seed_match.start()] if train_seed_match else model
    method = method.removeprefix("dapo_")
    row: dict[str, object] = {
        "model": model,
        "method": method,
        "train_seed": train_seed,
        "step": int(match.group("step")),
        "temperature": float(match.group("temperature")),
        "eval_seed": int(match.group("eval_seed") or 0),
        "aime24": source_accuracy.get("aime", np.nan),
        "aime25": source_accuracy.get("aime25", np.nan),
        "amc": source_accuracy.get("amc", np.nan),
        "math": source_accuracy.get("math", np.nan),
        "minerva": source_accuracy.get("minerva", np.nan),
        "olympiad_bench": source_accuracy.get("olympiad_bench", np.nan),
    }
    available = [float(source_accuracy[source]) for source in MAIN_SOURCES if source in source_accuracy]
    row["avg"] = float(np.mean(available)) if available else np.nan
    for source in PASS_AT_32_SOURCES:
        label = "aime24" if source == "aime" else source
        row[f"{label}_pass@32"] = pass_at_32(df, source) if source in source_accuracy else np.nan
    row["result_path"] = str(path.resolve())
    return row


def atomic_write_csv(table: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    table.to_csv(temporary, index=False)
    os.replace(temporary, output_path)


def atomic_write_json(rows: list[dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(rows, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[], help="Generation parquet; repeat as needed")
    parser.add_argument("--input-dir", type=Path, help="Also read generation_*.parquet from this directory")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--aggregate-csv", type=Path, help="Optional method-level mean/std across train seeds")
    parser.add_argument("--allow-subset", action="store_true", help="Disable exact valid.all count checks")
    args = parser.parse_args()

    paths = [Path(value) for value in args.input]
    if args.input_dir:
        paths.extend(args.input_dir.glob("generation_*.parquet"))
    paths = sorted({path.resolve() for path in paths if not path.name.endswith("_pre_reward.parquet")})
    if not paths:
        raise SystemExit("No generation parquet files found")

    rows = [summarize_file(path, strict=not args.allow_subset) for path in paths]
    table = pd.DataFrame(rows).sort_values(["method", "train_seed", "step", "eval_seed"])
    atomic_write_csv(table, args.output_csv)
    if args.output_json:
        json_rows = json.loads(table.to_json(orient="records"))
        atomic_write_json(json_rows, args.output_json)
    if args.aggregate_csv:
        keys = ["method", "train_seed", "step", "temperature", "eval_seed"]
        if table["train_seed"].isna().any() or table.duplicated(keys).any():
            raise ValueError("Aggregation requires a unique method/training-seed identity for each run")
        receipts = []
        for path in paths:
            receipt_path = path.parent / "evaluation_complete.json"
            if not receipt_path.is_file():
                raise ValueError(f"Aggregation requires completion receipt {receipt_path}")
            receipt = json.loads(receipt_path.read_text())
            if receipt.get("status") != "complete" or Path(receipt.get("result", "")).resolve() != path:
                raise ValueError(f"Invalid completion receipt for {path}")
            from evaluate_math import sha256
            if receipt.get("result_sha256") != sha256(path):
                raise ValueError(f"Output SHA mismatch: {path}")
            protocol = receipt["protocol"]
            receipts.append(tuple(protocol.get(key) for key in (
                "rollout.prompt_length", "rollout.response_length", "rollout.temperature", "rollout.top_p",
                "rollout.top_k", "data.n_samples")) + (receipt["evaluation_seed"], receipt["data_sha256"]))
        if len(set(receipts)) != 1 or table["step"].nunique() != 1:
            raise ValueError("Do not aggregate different checkpoints or evaluation protocols together")
        grouped = table.groupby(["method", "step", "temperature", "eval_seed"], dropna=False)
        for key, group in grouped:
            if set(group["train_seed"]) != {1, 42, 420} or len(group) != 3:
                raise ValueError(f"Expected exactly the three paper training seeds for {key}")
        aggregate = grouped[list(METRIC_COLUMNS)].agg(["count", "mean", "std"]).reset_index()
        aggregate.columns = [
            "_".join(str(part) for part in column if part).rstrip("_")
            if isinstance(column, tuple)
            else column
            for column in aggregate.columns
        ]
        atomic_write_csv(aggregate, args.aggregate_csv)

    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(table.drop(columns=["result_path"]).to_string(index=False, float_format=lambda x: f"{x:.6f}"))
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
