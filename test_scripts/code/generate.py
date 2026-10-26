#!/usr/bin/env python3
"""Generate HumanEval+ and MBPP+ in one GPU job. Never executes generated code."""

import argparse
import json
from pathlib import Path
import sys

from common import COUNTS, sha256, validate_frame

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train_scripts"))
from data_paths import data_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True, help="Merged local checkpoint")
    parser.add_argument("--data-root", type=Path, default=data_path("code"), help="Contains <benchmark>/test.parquet")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", required=True)
    parser.add_argument("--train-seed", type=int, required=True)
    parser.add_argument("--mode", choices=("greedy", "k16"), default="k16")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.gpus < 1:
        parser.error("--gpus must be positive")
    protocol = dict(mode=args.mode, n=16 if args.mode == "k16" else 1,
                    temperature=0.6 if args.mode == "k16" else 0.0, top_p=1.0, top_k=-1,
                    seed=0, prompt_length=1024, response_length=3072,
                    dtype="bfloat16", tensor_parallel_size=args.gpus, gpu_memory_utilization=0.8)
    print(json.dumps(dict(protocol=protocol, data_root=str(args.data_root.resolve()),
                          model=str(args.model.resolve()), output=str(args.output.resolve()),
                          problems=COUNTS, completions=sum(COUNTS.values()) * protocol["n"]), indent=2))
    if not args.execute:
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from evaluate_math import validate_model
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    validate_model(args.model)
    frames, input_hashes = {}, {}
    for bench in COUNTS:
        path = args.data_root / bench / "test.parquet"
        frames[bench] = pd.read_parquet(path)
        validate_frame(frames[bench], bench)
        input_hashes[bench] = sha256(path)
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=False)
    token_prompts = {}
    for bench, frame in frames.items():
        prompts = [tokenizer.apply_chat_template(list(row), tokenize=True, add_generation_prompt=True) for row in frame.prompt]
        if any(len(p) > protocol["prompt_length"] for p in prompts):
            raise ValueError(f"{bench}: prompt exceeds 1024 tokens; do not silently truncate")
        token_prompts[bench] = prompts
    args.output.mkdir(parents=True, exist_ok=True)
    engine = LLM(model=str(args.model), tokenizer=str(args.model), dtype="bfloat16", trust_remote_code=False,
                 tensor_parallel_size=args.gpus, distributed_executor_backend="mp", seed=0,
                 max_model_len=4096, gpu_memory_utilization=0.8)
    sampling = SamplingParams(n=protocol["n"], temperature=protocol["temperature"],
                              top_p=1.0, top_k=-1, seed=0, max_tokens=3072)
    artifacts = {}
    for bench, frame in frames.items():
        generated = engine.generate(prompt_token_ids=token_prompts[bench], sampling_params=sampling)
        if len(generated) != len(frame):
            raise ValueError("Incomplete generation")
        frame = frame.copy()
        frame["responses"] = [[x.text for x in row.outputs] for row in generated]
        frame["finish_reasons"] = [[x.finish_reason for x in row.outputs] for row in generated]
        validate_frame(frame, bench, protocol["n"])
        path = args.output / f"{bench}.parquet"
        frame.to_parquet(path, index=False)
        artifacts[bench] = dict(file=path.name, sha256=sha256(path))
        print("CODE_GENERATION_BENCHMARK_DONE", bench, flush=True)
    receipt = dict(status="complete", method=args.method, train_seed=args.train_seed,
                   model=str(args.model.resolve()), model_config_sha256=sha256(args.model / "config.json"),
                   protocol=protocol, input_sha256=input_hashes, outputs=artifacts)
    (args.output / "generation_complete.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print("CODE_GENERATION_DONE", flush=True)


if __name__ == "__main__":
    main()
