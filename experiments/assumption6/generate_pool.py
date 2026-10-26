#!/usr/bin/env python3
"""Generate a 256-by-128 response pool from a checkpoint and training trace.

Print the generation command by default; run it with --execute.
"""

import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
TOOLS = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "test_scripts"))
from evaluate_math import validate_model, sha256
from data_paths import data_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trace", type=Path, help="Override the trace path in data/paths.json")
    parser.add_argument("--train-data", type=Path, help="Override data/paths.json")
    parser.add_argument("--output", type=Path, help="Pool directory; defaults to data/rollouts/<dataset>")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    args.trace = args.trace or data_path("traces", args.dataset)
    args.train_data = args.train_data or data_path("training", args.dataset)
    args.output = args.output or data_path("rollouts", args.dataset)
    if args.step not in (6, 7, 8, 9, 10, 86, 87, 88, 89, 90, 166, 167, 168, 169, 170):
        parser.error("Use one of the fifteen saved-pool steps")
    out = args.output.resolve()
    batch = out / "prompt_batches" / f"step_{args.step}.parquet"
    result = out / f"{args.dataset}_generation_k10_step_{args.step}_batchTRACE.parquet"
    summary = result.with_suffix(".summary.json")
    record = out / "plans" / f"step_{args.step}.json"
    gsm = args.dataset == "gsm8k"
    options = {
        "model.path": str(args.checkpoint.resolve()), "data.path": str(batch),
        "data.prompt_key": "prompt", "data.output_path": str(result),
        "data.n_samples": 128, "data.batch_size": 16 if gsm else 8,
        "trainer.n_gpus_per_node": 1 if gsm else 4, "trainer.nnodes": 1,
        "rollout.temperature": 1.0, "rollout.top_p": 1, "rollout.top_k": -1,
        "rollout.gpu_memory_utilization": 0.85, "rollout.prompt_length": 512 if gsm else 1024,
        "rollout.response_length": 512 if gsm else 2048,
        "ray_kwargs.ray_init.num_cpus": 20 if gsm else 41,
        "hydra.run.dir": str(out / "hydra" / f"step_{args.step}"),
    }
    plan = [
        [sys.executable, str(TOOLS / "build_trace_batch.py"), "--trace", str(args.trace.resolve()),
         "--train-parquet", str(args.train_data.resolve()), "--step", str(args.step), "--output", str(batch)],
        [sys.executable, str(ROOT / "test_scripts/generate_math.py")] + [f"{k}={v}" for k, v in options.items()],
        [sys.executable, str(TOOLS / "validate_rollout_pool.py"), "--input", str(result),
         "--expected-step", str(args.step), "--summary", str(summary)],
    ]
    for command in plan:
        print(shlex.join(command), flush=True)
    if not args.execute:
        return
    validate_model(args.checkpoint)
    partial = result.with_name(result.stem + "_pre_reward.parquet")
    if any(path.exists() for path in (record, batch, result, partial, summary)):
        raise FileExistsError("This step already has output or a pending generation plan; inspect it before retrying")
    record.parent.mkdir(parents=True, exist_ok=True)
    receipt = dict(dataset=args.dataset, step=args.step, commands=plan, evaluation_seed=0,
                   trace_sha256=sha256(args.trace), train_data_sha256=sha256(args.train_data))
    record.write_text(json.dumps(receipt, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    for command in plan:
        subprocess.run(command, cwd=ROOT, env=env, check=True)
    receipt.update(status="complete", output_sha256=sha256(result))
    record.write_text(json.dumps(receipt, indent=2) + "\n")
    print("ASSUMPTION6_GENERATION_STEP_DONE", args.dataset, args.step, flush=True)


if __name__ == "__main__":
    main()
