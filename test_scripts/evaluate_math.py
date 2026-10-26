#!/usr/bin/env python3
"""Evaluate a merged model on the six paper math benchmarks. Preview by default."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_scripts"))
from data_paths import data_path
DATA_SHA256 = "92503399b914212d2e3d85839882bc1376a0111652fec0dcb722e86307e97617"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_model(path):
    path = Path(path).resolve()
    if not (path / "config.json").is_file():
        raise ValueError("Expected a merged Hugging Face checkpoint with config.json")
    index = path / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text()).get("weight_map", {})
        if not weight_map:
            raise ValueError("Empty model shard index")
        files = sorted(set(weight_map.values()))
    else:
        files = ["model.safetensors"]
    for filename in files:
        target = (path / filename).resolve()
        if not target.is_relative_to(path) or not target.is_file() or target.stat().st_size == 0:
            raise ValueError(f"Missing or invalid model shard: {filename}")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trained-on", choices=("math", "dapo"), required=True)
    parser.add_argument("--model", required=True, type=Path, help="Merged Hugging Face checkpoint directory")
    parser.add_argument("--data", type=Path, default=data_path("math_evaluation"))
    parser.add_argument("--run-name", required=True, help="For example dapo_kae_seed42; must end in _seed<number>")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not re.fullmatch(r"[A-Za-z0-9_-]+_seed\d+", args.run_name):
        raise ValueError("run-name must use letters/numbers/_/- and end in _seed<number>")
    length, step, seed = (2048, 200, 3407) if args.trained_on == "math" else (3072, 500, 0)
    output = args.output.resolve()
    result = output / f"generation_{args.run_name}_s{step}_temperature0.6_evalseed{seed}.parquet"
    options = {
        "model.path": str(args.model.resolve()), "data.path": str(args.data.resolve()),
        "data.prompt_key": "prompt", "data.output_path": str(result),
        "data.n_samples": 1, "data.batch_size": 1024,
        "trainer.n_gpus_per_node": 4, "trainer.nnodes": 1,
        "rollout.temperature": 0.6, "rollout.top_p": 1, "rollout.top_k": -1,
        "rollout.gpu_memory_utilization": 0.8, "rollout.prompt_length": 1024,
        "rollout.response_length": length, "ray_kwargs.ray_init.num_cpus": 48,
        "hydra.run.dir": str(output / "hydra"),
    }
    # Keep the original default-engine-seed-0 path unchanged for DAPO.
    if seed:
        options.update({"rollout._target_": "test_scripts.seeded_rollout.SeededRolloutConfig",
                        "+rollout.seed": seed})
    command = [sys.executable, str(ROOT / "test_scripts/generate_math.py")] + [
        f"{key}={value}" for key, value in options.items()]
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    if sha256(args.data) != DATA_SHA256:
        raise ValueError("Use the SHA-verified valid.all.parquet from the paper assets")
    model_files = validate_model(args.model)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    receipt = dict(command=command, protocol=options, evaluation_seed=seed, data_sha256=DATA_SHA256,
                   trained_on=args.trained_on, checkpoint_step=step, model_files=model_files)
    (output / "evaluation_start.json").write_text(json.dumps(receipt, indent=2) + "\n")
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    from summarize_math import summarize_file
    import pandas as pd

    row = summarize_file(result, strict=True)
    pd.DataFrame([row]).to_csv(output / "summary.csv", index=False)
    (output / "summary.json").write_text(json.dumps(row, indent=2, allow_nan=False) + "\n")
    receipt.update(status="complete", result_sha256=sha256(result), result=str(result))
    (output / "evaluation_complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("PAPER_EVAL_DONE", json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
