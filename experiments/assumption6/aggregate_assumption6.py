#!/usr/bin/env python3
"""Validate, merge, and summarize sharded Assumption 6 score outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from verify_assumption6 import summarize  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--expected-prompts", type=int, required=True)
    parser.add_argument("--expected-completions-per-prompt", type=int, required=True)
    parser.add_argument("--baseline-table")
    parser.add_argument("--bootstrap-replicates", type=int, default=0)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dirs = [Path(value).resolve() for value in args.input_dir]
    output_dir = Path(args.output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing aggregate: {output_dir}")

    required_columns = {
        "uid",
        "completion_id",
        "reward",
        "score_norm_sq",
        "sequence_log_prob",
        "response_tokens",
    }
    frames: list[pd.DataFrame] = []
    shard_records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for directory in input_dirs:
        parquet = directory / "per_completion.parquet"
        prompt_csv = directory / "per_prompt.csv"
        summary_json = directory / "summary.json"
        for path in (parquet, prompt_csv, summary_json):
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty shard artifact: {path}")
        frame = pd.read_parquet(parquet)
        missing = required_columns.difference(frame.columns)
        if missing:
            raise ValueError(f"{parquet} is missing columns: {sorted(missing)}")
        summary = json.loads(summary_json.read_text())
        summaries.append(summary)
        frames.append(frame)
        shard_records.append(
            {
                "input_dir": str(directory),
                "prompt_shard_index": int(summary["prompt_shard_index"]),
                "num_prompt_shards": int(summary["num_prompt_shards"]),
                "rows": int(len(frame)),
                "prompts": int(frame["uid"].nunique()),
                "per_completion_sha256": sha256(parquet),
                "per_prompt_sha256": sha256(prompt_csv),
                "summary_sha256": sha256(summary_json),
            }
        )

    reference_keys = (
        "checkpoint",
        "data",
        "parameter_scope",
        "selected_parameter_count",
        "score_reduction",
    )
    for key in reference_keys:
        values = {json.dumps(summary.get(key), sort_keys=True) for summary in summaries}
        if len(values) != 1:
            raise ValueError(f"Shard summaries disagree on {key}: {values}")
    expected_shards = int(summaries[0]["num_prompt_shards"])
    shard_indices = sorted(int(summary["prompt_shard_index"]) for summary in summaries)
    if len(input_dirs) != expected_shards or shard_indices != list(range(expected_shards)):
        raise ValueError(
            f"Incomplete shard coverage: have {shard_indices}, expected {list(range(expected_shards))}"
        )

    frame = pd.concat(frames, ignore_index=True)
    expected_rows = args.expected_prompts * args.expected_completions_per_prompt
    if len(frame) != expected_rows:
        raise ValueError(f"Expected {expected_rows} rows, found {len(frame)}")
    if frame["uid"].nunique() != args.expected_prompts:
        raise ValueError(
            f"Expected {args.expected_prompts} prompts, found {frame['uid'].nunique()}"
        )
    counts = frame.groupby("uid", sort=False).size()
    if not counts.eq(args.expected_completions_per_prompt).all():
        raise ValueError(f"Unexpected completions per prompt: {counts.value_counts().to_dict()}")
    if frame.duplicated(["uid", "completion_id"]).any():
        raise ValueError("Duplicate (uid, completion_id) pairs found")
    for column in ("reward", "score_norm_sq", "sequence_log_prob", "response_tokens"):
        if not np.isfinite(frame[column].to_numpy(dtype=float)).all():
            raise ValueError(f"Non-finite values in {column}")
    if not frame["reward"].between(0.0, 1.0).all():
        raise ValueError("Rewards fall outside [0, 1]")
    if (frame["score_norm_sq"] < 0).any() or (frame["response_tokens"] <= 0).any():
        raise ValueError("Invalid score norm or response token count")

    per_prompt, cross_terms, summary = summarize(
        frame,
        args.baseline_table,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    valid_spearman = per_prompt["spearman"].dropna().to_numpy(dtype=float)
    displacement = per_prompt["optimal_minus_value"].to_numpy(dtype=float)
    summary.update(
        dataset=args.dataset,
        step=args.step,
        aggregation="validated_shard_union",
        expected_prompts=args.expected_prompts,
        expected_completions_per_prompt=args.expected_completions_per_prompt,
        mean_conditional_spearman=(
            float(valid_spearman.mean()) if valid_spearman.size else float("nan")
        ),
        median_abs_conditional_spearman=(
            float(np.median(np.abs(valid_spearman))) if valid_spearman.size else float("nan")
        ),
        mean_optimal_minus_value=float(displacement.mean()),
        median_optimal_minus_value=float(np.median(displacement)),
        rmse_optimal_minus_value=float(np.sqrt(np.mean(displacement**2))),
        input_shards=shard_records,
        source_config={key: summaries[0].get(key) for key in reference_keys},
    )

    temp_dir = output_dir.with_name(f"{output_dir.name}.tmp-{os.getpid()}")
    if temp_dir.exists():
        shutil.rmtree(temp_dir)
    temp_dir.mkdir(parents=True)
    frame.to_parquet(temp_dir / "per_completion.parquet", index=False)
    per_prompt.to_csv(temp_dir / "per_prompt.csv", index=False)
    if not cross_terms.empty:
        cross_terms.to_csv(temp_dir / "baseline_cross_terms.csv", index=False)
    (temp_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=True) + "\n"
    )
    manifest = {
        "dataset": args.dataset,
        "step": args.step,
        "num_prompts": int(per_prompt.shape[0]),
        "num_completions": int(frame.shape[0]),
        "shards": shard_records,
    }
    (temp_dir / "aggregate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    for name in ("per_completion.parquet", "per_prompt.csv", "summary.json"):
        manifest[f"{name}_sha256"] = sha256(temp_dir / name)
    (temp_dir / "aggregate_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
    )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir.replace(output_dir)
    print(
        f"AGGREGATE_VALID dataset={args.dataset} step={args.step} "
        f"prompts={len(per_prompt)} completions={len(frame)} output={output_dir}"
    )


if __name__ == "__main__":
    main()
