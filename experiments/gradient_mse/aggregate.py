#!/usr/bin/env python3
"""Validate and aggregate deterministic retrained gradient-MSE shards."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ERROR_COLUMNS = ["trace_ref_error_grpo_no_norm", "trace_ref_error_overt_no_norm", "trace_ref_error_reinforcepp"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def one_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one {pattern} in {directory}, found {len(matches)}")
    if matches[0].stat().st_size == 0:
        raise RuntimeError(f"Empty output: {matches[0]}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-dir", type=Path, required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--step", type=int, choices=(10, 90, 170), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Merged checkpoint used for this step")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--prompts-per-shard", type=int, default=256)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.num_shards < 1 or args.prompts_per_shard < 1 or args.num_shards * args.prompts_per_shard != 256:
        parser.error("The formal aggregate must cover exactly 256 prompts")

    output_dir = args.output_dir or args.formal_dir / "aggregate"
    frames: list[pd.DataFrame] = []
    inputs: list[dict[str, object]] = []
    seen: set[str] = set()
    global_plan_identity = None

    for shard_index in range(args.num_shards):
        shard_dir = args.formal_dir / f"shard_{shard_index}_of_{args.num_shards}"
        summary_path = one_file(shard_dir, "*_summary.json")
        detail_path = one_file(shard_dir, "*_detail.parquet")
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        config = summary["config"]
        selection = summary["selection_stats"]
        expected = {
            "target_step_abs": args.step,
            "k_eval": 8,
            "m_repeats": 50,
            "num_prompt_shards": args.num_shards,
            "prompt_shard_index": shard_index,
            "sampling_with_replacement": True,
            "linear_alpha": 0.2,
            "ema_min_var": 0.01,
            "dtype": "bf16",
            "seed": 0,
            "reinforcepp_baseline": "global_batch_completion_leave_one_out",
            "reinforcepp_batch_prompts": 256,
            "reinforcepp_batch_completions": 2048,
        }
        for key, value in expected.items():
            if config.get(key) != value:
                raise RuntimeError(f"{summary_path}: config[{key}]={config.get(key)!r}, expected {value!r}")
        identity = tuple(config.get(key) for key in ("reinforcepp_batch_uid_sha256", "sampling_plan_sha256"))
        if any(not isinstance(digest, str) or len(digest) != 64 for digest in identity):
            raise RuntimeError("Missing full-batch sampling identity")
        if global_plan_identity is not None and identity != global_plan_identity:
            raise RuntimeError("Shards did not use the same full statistical batch and draws")
        global_plan_identity = identity
        expected_history = [args.step - 4, args.step - 3, args.step - 2, args.step - 1]
        if config.get("history_steps") != expected_history:
            raise RuntimeError(f"{summary_path}: history_steps != {expected_history}")
        if Path(config.get("model_path", "")).resolve() != args.checkpoint.resolve():
            raise RuntimeError(f"{summary_path}: checkpoint does not match step {args.step}")
        if args.dataset == "gsm8k":
            if config.get("reference_logprob_micro_batch") != 16:
                raise RuntimeError(f"{summary_path}: GSM8K reference microbatch != 16")
        else:
            if config.get("logprob_micro_batch") != 8:
                raise RuntimeError(f"{summary_path}: MATH logprob microbatch != 8")
            scope = config.get("grad_scope_info", {})
            scope_expected = {
                "grad_scope": "last_n",
                "grad_last_n_layers": 5,
                "include_lm_head": True,
                "include_model_norm": True,
                "include_embed_tokens": False,
                "selected_layer_ids": [23, 24, 25, 26, 27],
            }
            for key, value in scope_expected.items():
                if scope.get(key) != value:
                    raise RuntimeError(f"{summary_path}: grad_scope_info[{key}]={scope.get(key)!r}, expected {value!r}")
        if selection.get("prompts_before_sharding") != 256:
            raise RuntimeError(f"{summary_path}: prompts_before_sharding != 256")
        if selection.get("selected_prompts") != args.prompts_per_shard:
            raise RuntimeError(f"{summary_path}: selected_prompts != {args.prompts_per_shard}")
        if summary.get("num_prompts_succeeded") != args.prompts_per_shard:
            raise RuntimeError(f"{summary_path}: unexpected success count")
        if summary.get("num_prompts_failed") != 0:
            raise RuntimeError(f"{summary_path}: nonzero failed prompts")

        frame = pd.read_parquet(detail_path)
        if len(frame) != args.prompts_per_shard or frame["uid"].nunique() != args.prompts_per_shard:
            raise RuntimeError(f"{detail_path}: expected {args.prompts_per_shard} unique prompts")
        frame["uid"] = frame["uid"].astype(str)
        overlap = seen.intersection(frame["uid"])
        if overlap:
            raise RuntimeError(f"UID overlap at shard {shard_index}: {sorted(overlap)[:5]}")
        seen.update(frame["uid"])
        for column in ERROR_COLUMNS:
            values = frame[column].to_numpy(dtype=float)
            if not np.isfinite(values).all() or (values < 0).any():
                raise RuntimeError(f"{detail_path}: invalid values in {column}")
        frame.insert(0, "source_shard", shard_index)
        frames.append(frame)
        inputs.append(
            {
                "shard_index": shard_index,
                "summary": str(summary_path),
                "summary_sha256": sha256(summary_path),
                "detail": str(detail_path),
                "detail_sha256": sha256(detail_path),
            }
        )

    detail = pd.concat(frames, ignore_index=True)
    expected_prompts = args.num_shards * args.prompts_per_shard
    if len(detail) != expected_prompts or detail["uid"].nunique() != expected_prompts:
        raise RuntimeError("Aggregate does not contain exactly 256 disjoint prompt UIDs")
    grpo = float(detail[ERROR_COLUMNS[0]].mean())
    kae = float(detail[ERROR_COLUMNS[1]].mean())
    rfpp = float(detail[ERROR_COLUMNS[2]].mean())
    reduction = float(100.0 * (1.0 - kae / grpo)) if grpo > 0 else None
    result = {
        "dataset": args.dataset,
        "step": args.step,
        "num_shards": args.num_shards,
        "num_prompts": expected_prompts,
        "k_eval": 8,
        "m_repeats": 50,
        "grpo_mean_oracle_centered_trace_error": grpo,
        "kae_mean_oracle_centered_trace_error": kae,
        "reinforcepp_mean_oracle_centered_trace_error": rfpp,
        "kae_reduction_percent": reduction,
        "kae_reduction_vs_reinforcepp_percent": float(100.0 * (1.0 - kae / rfpp)) if rfpp > 0 else None,
        "reinforcepp_baseline": "global_batch_completion_leave_one_out",
        "sampling_plan_sha256": global_plan_identity[1],
    }

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    detail_out = output_dir / "prompt_details.parquet"
    csv_out = output_dir / "gradient_mse_summary.csv"
    json_out = output_dir / "gradient_mse_summary.json"
    manifest_out = output_dir / "aggregate_manifest.json"
    detail.to_parquet(detail_out, index=False)
    pd.DataFrame([result]).to_csv(csv_out, index=False)
    json_out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "result": result,
        "inputs": inputs,
        "outputs": {
            "prompt_details": str(detail_out),
            "prompt_details_sha256": sha256(detail_out),
            "summary_csv": str(csv_out),
            "summary_csv_sha256": sha256(csv_out),
            "summary_json": str(json_out),
            "summary_json_sha256": sha256(json_out),
        },
    }
    manifest_out.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    print(f"aggregate_manifest_sha256={sha256(manifest_out)}")


if __name__ == "__main__":
    main()
