#!/usr/bin/env python3
"""Preview or run analysis of saved rollout pools. Never generates online samples."""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "train_scripts"))
from data_paths import data_path


def commands(args):
    root = args.rollouts.resolve()
    target = root / f"{args.dataset}_generation_k10_step_{args.step}_batchTRACE.parquet"
    output = args.output.resolve()
    prefix = [sys.executable]
    if args.experiment == "kernel_sensitivity":
        return [prefix + [str(HERE / "kernel_sensitivity/run.py"), "--dataset", args.dataset,
                         "--rollouts", str(root), "--step", str(args.step), "--repeats", "100",
                         "--seed", "1234", "--output", str(output / "sensitivity")]]
    if args.experiment == "value_mse":
        return [prefix + [str(HERE / "value_mse/run.py"), "--data-name", args.dataset,
                         "--data-root", str(root), "--steps-glob", str(root / f"{args.dataset}_generation_k10_step_*_batchTRACE.parquet"),
                         "--target-step", str(args.step), "--history-window", "4", "--repeats", "100",
                         "--true-n", "100", "--eval-n", "8", "--history-n", "8", "--kae-alpha", "0.2",
                         "--oracle-sample-tag", "", "--n-bootstrap", "0", "--seed", "1234",
                         "--output-dir", str(output)]]
    if args.checkpoint is None:
        raise ValueError("This experiment requires --checkpoint")
    checkpoint = str(args.checkpoint.resolve())
    if args.experiment == "assumption6":
        baseline = output / "kae_baselines.parquet"
        return [
            prefix + [str(HERE / "assumption6/build_baseline_table.py"), "--rollout-dir", str(root),
                      "--dataset", args.dataset, "--target-step", str(args.step), "--output", str(baseline),
                      "--oracle-n", "100", "--eval-n", "8", "--repeats", "5", "--alpha", "0.2", "--seed", "1234"],
            prefix + [str(HERE / "assumption6/verify_assumption6.py"), "--checkpoint", checkpoint,
                      "--data", str(target), "--output-dir", str(output / "scores"), "--baseline-table", str(baseline),
                      "--parameter-scope", "all", "--score-reduction", "sum", "--dtype", "bfloat16",
                      "--max-prompt-tokens", "512" if args.dataset == "gsm8k" else "1024",
                      "--max-response-tokens", "512" if args.dataset == "gsm8k" else "2048",
                      "--max-prompts", "256", "--max-completions-per-prompt", "128",
                      "--bootstrap-replicates", "0", "--device-map", "none", "--device", "cuda"],
            prefix + [str(HERE / "assumption6/report.py"), "--scores", str(output / "scores"),
                      "--output", str(output / "paper_summary.json")],
        ]
    # Use the exact persisted draw indices from the saved experiment. Selecting
    # a different compute partition must never redefine RF++'s full batch.
    plan = HERE / "gradient_mse/draws" / args.dataset / f"step_{args.step}"
    receipt = output / "checkpoint_verified.json"
    return [
        prefix + [str(HERE / "gradient_mse/verify_checkpoint.py"), "--dataset", args.dataset,
                  "--step", str(args.step), "--checkpoint", checkpoint, "--output", str(receipt)],
        prefix + [str(HERE / "gradient_mse/global_loo.py"), "--dataset", args.dataset,
                  "--step", str(args.step), "--plan", str(plan), "--checkpoint", checkpoint,
                  "--checkpoint-receipt", str(receipt), "--rollouts", str(root), "--output", str(output),
                  "--methods", "reinforcepp", "grpo_no_norm", "overt_no_norm", "--repeats", "50",
                  "--max-prompts", "256", "--reference-microbatch", "16" if args.dataset == "gsm8k" else "8",
                  "--repeat-microbatch", "0"],
    ]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=("value_mse", "kernel_sensitivity", "gradient_mse", "assumption6"), required=True)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--step", type=int, choices=(10, 90, 170), required=True)
    parser.add_argument("--rollouts", type=Path, help="Override data/paths.json rollout directory")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.rollouts = args.rollouts or data_path("rollouts", args.dataset)
    plan = commands(args)
    for command in plan:
        print(shlex.join(command), flush=True)
    if not args.execute:
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {args.output}")
    steps = range(args.step - 4, args.step + 1)
    for step in steps:
        path = args.rollouts / f"{args.dataset}_generation_k10_step_{step}_batchTRACE.parquet"
        subprocess.run([sys.executable, str(HERE / "assumption6/validate_rollout_pool.py"),
                        "--input", str(path), "--expected-step", str(step)], check=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "analysis_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    for command in plan:
        subprocess.run(command, check=True)
    from validate_output import validate
    receipt = validate(args)
    (args.output / "analysis_complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("ANALYSIS_COMPLETE", json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
