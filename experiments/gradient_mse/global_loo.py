"""Compute global-LOO gradient MSE from saved pools and immutable draws.

No generation or optimization. Reference gradients and scalar squared-distance
reductions stay on the GPU. All three methods are evaluated by default. MATH
uses the last five decoder blocks, final model norm and output layer.
"""

import argparse
import fcntl
import importlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from global_plan import batch_totals, read_plan, sha256
from paper_protocol import math_parameter_scope, reinforcepp_loo_advantages
from boundary import POLICY as TOKENIZATION_POLICY, tokenize_fixed_prompt


@torch.no_grad()
def scalar_squared_error(params, reference):
    """Same sum of squared coordinate errors, without per-method large buffers."""
    total = torch.zeros((), dtype=torch.float64, device=reference[0].device)
    for param, ref in zip(params, reference):
        # BF16 reference storage is lossless here: the source gradients are
        # already BF16. Subtraction/squaring still take place in FP32.
        ref32 = ref.float()
        difference = -ref32 if param.grad is None else param.grad.detach().float() - ref32
        total.add_(difference.square().sum(dtype=torch.float32).double())
    return total


def save_json(path, payload):
    """Atomic publication; never replace an already published artifact."""
    if path.exists():
        raise FileExistsError(path)
    temporary = path.with_name(path.name + f".tmp.{os.getpid()}")
    with temporary.open("x") as handle:
        json.dump(payload, handle, indent=2, allow_nan=False)
        handle.write("\n")
    os.rename(temporary, path)


def run(args):
    helper = importlib.import_module("gsm8k" if args.dataset == "gsm8k" else "math_mse")
    meta, plan = read_plan(args.plan)
    if meta["dataset"] != args.dataset or meta["step"] != args.step:
        raise ValueError("Plan dataset/step mismatch")
    checkpoint_manifest = json.loads(args.checkpoint_receipt.read_text())
    checkpoint_record = next(x for x in checkpoint_manifest["checkpoints"]
        if x["dataset"] == args.dataset and x["step"] == args.step)
    if checkpoint_manifest["status"] != "complete" or checkpoint_record["path"] != str(args.checkpoint.resolve()):
        raise ValueError("Missing verified checkpoint identity")
    for item in checkpoint_record["files"]:
        path = args.checkpoint / item["name"]
        stat = path.stat()
        if stat.st_size != item["bytes"] or stat.st_mtime_ns != item["mtime_ns"]:
            raise ValueError(f"Checkpoint changed since SHA verification: {path}")
    step_samples = {}
    for item in meta["inputs"]:
        path = args.rollouts / item["filename"]
        if sha256(path) != item["sha256"]:
            raise ValueError(f"Rollout SHA mismatch: {path}")
        samples, stats = helper.load_step_samples(str(path), prompt_col="prompt")
        if stats["rows_loaded"] != 256 or any(stats[k] for k in stats if k.startswith("skipped") or k == "duplicate_uid"):
            raise ValueError(f"Incomplete prompt pool: {path}, {stats}")
        if set(samples) != set(plan["uids"]):
            raise ValueError("Rollout UID set does not match the plan")
        step_samples[item["step"]] = samples
    target = step_samples[args.step]
    totals, total_n = batch_totals(plan["uids"], plan["indices"], {uid:s.rewards for uid,s in target.items()})
    np.testing.assert_array_equal(totals, plan["global_reward_sums"])
    if total_n != 2048:
        raise ValueError("RF++ statistical batch must be 256 x 8")
    selected = list(range(256))[:args.max_prompts or 256]
    repeats = args.repeats
    config = dict(dataset=args.dataset, step=args.step, checkpoint=str(args.checkpoint.resolve()),
        checkpoint_sha_manifest=checkpoint_record, plan_sha256=meta["plan_sha256"],
        methods=args.methods, prompts=len(selected), global_batch_prompts=256,
        global_batch_completions=2048, k_eval=8, m_repeats=repeats, seed=0,
        sampling_with_replacement=True, baseline="global_batch_completion_leave_one_out",
        history_steps=list(range(args.step-4,args.step)), linear_alpha=0.2, variance_floor=0.01,
        dtype="bf16", score_reduction="sum", cov_device="cuda",
        reference_pool_size=128, reference_microbatch=args.reference_microbatch or (16 if args.dataset=="gsm8k" else 8),
        repeat_microbatch=args.repeat_microbatch,
        reference_storage_dtype="source_gradient_dtype", tokenization_policy=TOKENIZATION_POLICY,
        max_cont_tokens=0, grad_scope="all" if args.dataset=="gsm8k" or args.math_full_parameters else "last5+model_norm+lm_head_no_embeddings",
        code_sha256={p.name:sha256(p) for p in [Path(__file__), Path(helper.__file__), Path(__file__).with_name("global_plan.py"), Path(__file__).with_name("paper_protocol.py"), Path(__file__).with_name("boundary.py")]})
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output/"run.lock").open("a+")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config_path = args.output/"run_config.json"
    if config_path.exists():
        if json.loads(config_path.read_text()) != config:
            raise ValueError("Refusing to mix configurations in an output directory")
    else:
        save_json(config_path, config)
    if (args.output/"summary.json").exists():
        raise FileExistsError("Run is already complete; do not duplicate it")
    print("GLOBAL_LOO_START", json.dumps(config), flush=True)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True, local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, torch_dtype=torch.bfloat16,
        trust_remote_code=True, local_files_only=True).to("cuda")
    model.eval()
    if args.dataset == "math" and not args.math_full_parameters:
        scope = math_parameter_scope()
        params, scope_info = helper.configure_grad_parameter_scope(model, scope)
        if scope_info["selected_layer_ids"] != list(range(model.config.num_hidden_layers - 5, model.config.num_hidden_layers)):
            raise ValueError("Unexpected MATH decoder-layer selection")
    else:
        params = list(model.parameters())
        scope_info = dict(grad_scope="all")
    scope_info["parameter_count"] = sum(p.numel() for p in params)
    print("GLOBAL_LOO_MODEL_READY", json.dumps(scope_info), flush=True)
    (args.output/"prompts").mkdir(exist_ok=True)
    started = time.monotonic()
    records = []
    for position, index in enumerate(selected):
        uid = str(plan["uids"][index])
        prompt_path = args.output/"prompts"/f"prompt_{index:03d}.json"
        if prompt_path.exists():
            row = json.loads(prompt_path.read_text())
            if row["uid"] != uid or row["repeats"] != repeats or row["plan_sha256"] != meta["plan_sha256"]:
                raise ValueError("Invalid resumable prompt")
            for method in args.methods:
                if not np.isfinite(row[f"trace_ref_error_{method}"]) or row[f"trace_ref_error_{method}"] < 0:
                    raise ValueError("Invalid saved gradient error")
            records.append(row)
            print("GLOBAL_LOO_SKIP", index, uid, flush=True)
            continue
        prompt_start = time.monotonic()
        torch.cuda.reset_peak_memory_stats()
        sample = target[uid]
        prompt_text, prompt_ids = helper.tokenize_prompt(tokenizer, sample.messages)
        cont_ids, repairs = tokenize_fixed_prompt(tokenizer, prompt_text, prompt_ids, sample.responses)
        if repairs:
            print("GLOBAL_LOO_BOUNDARY_REPAIR", json.dumps(dict(uid=uid, repairs=repairs)), flush=True)
        if len(cont_ids) != 128 or any(x.numel()==0 for x in cont_ids):
            raise ValueError(f"Invalid tokenization for UID {uid}")
        rewards = torch.tensor(sample.rewards, dtype=torch.float32, device="cuda")
        value = float(sample.rewards.mean())
        helper.backward_prompt_loss_microbatched(model, tokenizer, prompt_ids, cont_ids,
            rewards-value, config["reference_microbatch"])
        reference = [torch.zeros_like(p) if p.grad is None else p.grad.detach().clone() for p in params]
        model.zero_grad(set_to_none=True)
        reference_norm = sum(float(r.float().square().sum(dtype=torch.float32).item()) for r in reference)
        accumulators = {method:torch.zeros((), dtype=torch.float64, device="cuda") for method in args.methods}
        compute_start = time.monotonic()
        for repeat in range(repeats):
            indices = plan["indices"][index,repeat,-1,:]
            chosen = [cont_ids[int(k)] for k in indices]
            sampled_rewards = rewards[torch.as_tensor(indices,device="cuda")]
            # 2048/2047 is algebraically identical to excluding this completion
            # from the global reward sum. It adds no model evaluation.
            advantages = {"reinforcepp": reinforcepp_loo_advantages(sampled_rewards, float(totals[repeat]/total_n), total_n)}
            if "grpo_no_norm" in args.methods:
                advantages["grpo_no_norm"] = (8/7)*(sampled_rewards-float(sample.rewards[indices].mean()))
            if "overt_no_norm" in args.methods:
                stats = {int(step):helper.compute_mean_var(step_samples[int(step)][uid].rewards[plan["indices"][index,repeat,j]])
                         for j,step in enumerate(plan["steps"])}
                mean, _, weight_sum, _ = helper.compute_kae_baseline(args.step, stats, 0.2, 0.01)
                eff_n = 8*weight_sum
                advantages["overt_no_norm"] = (eff_n/(eff_n-1))*(sampled_rewards-mean)
            if 0 < args.repeat_microbatch < 8:
                for method in args.methods:
                    helper.backward_prompt_loss_microbatched(model, tokenizer, prompt_ids, chosen,
                        advantages[method], args.repeat_microbatch)
                    accumulators[method].add_(scalar_squared_error(params, reference))
            else:
                lp = helper.batched_logprob_continuations(model, tokenizer, prompt_ids, chosen)
                for method_index, method in enumerate(args.methods):
                    model.zero_grad(set_to_none=True)
                    (-(advantages[method]*lp).mean()).backward(retain_graph=method_index<len(args.methods)-1)
                    accumulators[method].add_(scalar_squared_error(params, reference))
                del lp
            model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        errors = {f"trace_ref_error_{method}":float(value.item()/repeats) for method,value in accumulators.items()}
        if not all(np.isfinite(v) and v>=0 for v in errors.values()) or not np.isfinite(reference_norm):
            raise ValueError(f"Nonfinite gradient error for UID {uid}")
        row = dict(uid=uid, global_index=index, repeats=repeats, plan_sha256=meta["plan_sha256"],
            reference_pool_size=128, reference_value=value, reference_grad_sq_norm=reference_norm,
            tokenization_policy=TOKENIZATION_POLICY, boundary_repairs=repairs,
            seconds=time.monotonic()-prompt_start, repeat_seconds=time.monotonic()-compute_start,
            peak_gpu_bytes=torch.cuda.max_memory_allocated(), **errors)
        save_json(prompt_path,row)
        records.append(row)
        print("GLOBAL_LOO_PROGRESS", json.dumps(dict(completed=position+1,total=len(selected),**row)),flush=True)
        del reference, accumulators, advantages
    frame = pd.DataFrame(records)
    if len(frame)!=len(selected) or frame.uid.nunique()!=len(selected):
        raise ValueError("Incorrect prompt coverage")
    output = args.output/"prompt_details.parquet"
    frame.to_parquet(output,index=False)
    summary = dict(status="complete", config=config, scope=scope_info, num_prompts_succeeded=len(frame),
        num_prompts_failed=0, detail_sha256=sha256(output), elapsed_seconds=time.monotonic()-started,
        avg_trace_ref_error={method:float(frame[f"trace_ref_error_{method}"].mean()) for method in args.methods})
    save_json(args.output/"summary.json",summary)
    print("GLOBAL_LOO_DONE", json.dumps(summary),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gsm8k","math"), required=True)
    parser.add_argument("--step", type=int, choices=(10,90,170), required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-receipt", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--methods", nargs="+", choices=("reinforcepp","grpo_no_norm","overt_no_norm"), default=["reinforcepp","grpo_no_norm","overt_no_norm"])
    parser.add_argument("--max-prompts", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--math-full-parameters", action="store_true")
    parser.add_argument("--reference-microbatch", type=int, default=0)
    parser.add_argument("--repeat-microbatch", type=int, default=0)
    args=parser.parse_args()
    if (not 1<=args.repeats<=50 or not 0<=args.max_prompts<=256 or len(args.methods)!=len(set(args.methods))
            or args.reference_microbatch<0 or args.repeat_microbatch<0):
        parser.error("Invalid prompt/repeat/method selection")
    run(args)
