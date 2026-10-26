"""Require complete paper-sized outputs before declaring an analysis successful."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def unique_file(root, pattern):
    files = list(root.glob(pattern))
    if len(files) != 1 or files[0].stat().st_size == 0:
        raise ValueError(f"Expected one nonempty {pattern} in {root}")
    return files[0]


def check(condition, message):
    if not condition:
        raise ValueError(message)


def finite_nonnegative(frame, columns):
    values = frame[columns].to_numpy(dtype=float)
    check(np.isfinite(values).all() and (values >= 0).all(), f"Invalid metric in {columns}")


def validate(args):
    root = args.output.resolve()
    if args.experiment == "value_mse":
        path = unique_file(root, "*_prompt_metrics.parquet")
        summary_path = unique_file(root, "*_summary.json")
        summary = json.loads(summary_path.read_text())
        frame = pd.read_parquet(path)
        methods = {"Naive", "REINFORCE++", "GRPO", "KAE", "Kalman", "Shrinkage"}
        check(summary["num_prompts"] == 256 and summary["num_excluded_prompts"] == 0, "Missing value-MSE prompts")
        check(summary["data_name"] == args.dataset and summary["target_step"] == args.step, "Wrong dataset or step")
        check(len(frame) == 256 * len(methods) and set(frame["method"]) == methods, "Incomplete method coverage")
        check(not frame.duplicated(["uid", "method"]).any(), "Duplicate prompt/method rows")
        finite_nonnegative(frame, ["mse", "variance"])
    elif args.experiment == "kernel_sensitivity":
        path = root / "sensitivity/prompt_metrics.parquet"
        summary_path = root / "sensitivity/summary.json"
        summary = json.loads(summary_path.read_text())
        frame = pd.read_parquet(path)
        receipt = json.loads((root / "sensitivity/complete.json").read_text())
        check(receipt["status"] == "complete", "Sensitivity run incomplete")
        check(summary["dataset"] == args.dataset and summary["step"] == args.step, "Wrong dataset/step")
        check(summary["config"]["repeats"] == 100 and summary["config"]["group_size"] == 8, "Wrong repeats/group size")
        check(len(frame) == 256 * 22 and frame.groupby("uid").size().eq(22).all(), "Incomplete sensitivity grid")
        check(not frame.duplicated(["uid", "method"]).any(), "Duplicate prompt/method")
        check(frame["method"].nunique() == 22, "Wrong method count")
        finite_nonnegative(frame, ["mse", "variance"])
        for name, digest in receipt["outputs"].items():
            check(hashlib.sha256((path.parent / name).read_bytes()).hexdigest() == digest, f"Output hash mismatch: {name}")
    elif args.experiment == "assumption6":
        path = root / "scores/per_completion.parquet"
        summary_path = root / "scores/summary.json"
        summary = json.loads(summary_path.read_text())
        frame = pd.read_parquet(path)
        check(len(frame) == 32768 and frame.groupby("uid").size().eq(128).all(), "Incomplete completion coverage")
        check(not frame.duplicated(["uid", "completion_id"]).any(), "Duplicate completion rows")
        finite_nonnegative(frame, ["score_norm_sq", "response_tokens", "reward"])
        check(frame["reward"].le(1).all() and frame["response_tokens"].gt(0).all(), "Invalid rewards or token counts")
        check(np.isfinite(frame["sequence_log_prob"]).all(), "Nonfinite sequence log probability")
        check(summary["parameter_scope"] == "all" and summary["score_reduction"] == "sum", "Wrong score definition")
        expected_parameters = 1543714304 if args.dataset == "gsm8k" else 7615616512
        check(summary["selected_parameter_count"] == expected_parameters, "Wrong model/parameter scope")
        check(Path(summary["checkpoint"]).resolve() == args.checkpoint.resolve(), "Wrong checkpoint")
    elif (root / "summary.json").exists():
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent / "gradient_mse"))
        from report_global_loo import validate as validate_global
        summary, frame = validate_global(root, args.dataset, args.step)
        check(Path(summary["config"]["checkpoint"]).resolve() == args.checkpoint.resolve(), "Wrong checkpoint")
        path, summary_path = root / "prompt_details.parquet", root / "summary.json"
    else:
        path = unique_file(root, "*_detail.parquet")
        summary_path = unique_file(root, "*_summary.json")
        summary = json.loads(summary_path.read_text())
        frame = pd.read_parquet(path)
        cfg, selection = summary["config"], summary["selection_stats"]
        check(summary["num_prompts_succeeded"] == 256 and summary["num_prompts_failed"] == 0, "Failed/incomplete prompts")
        check(selection["prompts_before_sharding"] == 256 and selection["selected_prompts"] == 256, "Unexpected selection")
        check(cfg["target_step_abs"] == args.step and cfg["m_repeats"] == 50, "Wrong step/repeats")
        check(cfg["num_prompt_shards"] == 1 and cfg["prompt_shard_index"] == 0, "Unexpected sharding")
        check(cfg.get("reinforcepp_baseline") == "global_batch_completion_leave_one_out", "RF++ is not global LOO")
        check(cfg.get("reinforcepp_batch_prompts") == 256 and cfg.get("reinforcepp_batch_completions") == 2048,
              "RF++ must include the full 256 x 8 batch")
        if args.dataset == "math":
            scope = cfg.get("grad_scope_info", {})
            expected = dict(grad_scope="last_n", grad_last_n_layers=5, selected_layer_ids=[23,24,25,26,27],
                            include_lm_head=True, include_model_norm=True, include_embed_tokens=False)
            check(all(scope.get(k) == v for k, v in expected.items()), "MATH scope must be last five blocks + model norm + output layer")
        check(Path(cfg["model_path"]).resolve() == args.checkpoint.resolve(), "Wrong checkpoint")
        check(len(frame) == 256, "Wrong gradient-MSE row count")
        finite_nonnegative(frame, ["trace_ref_error_grpo_no_norm", "trace_ref_error_overt_no_norm", "trace_ref_error_reinforcepp"])
    check(frame["uid"].nunique() == 256, "Expected exactly 256 unique prompts")
    return dict(status="complete", experiment=args.experiment, dataset=args.dataset, step=args.step,
                rows=len(frame), prompts=256, output=str(path),
                output_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                summary_sha256=hashlib.sha256(summary_path.read_bytes()).hexdigest())
