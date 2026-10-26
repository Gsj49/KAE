#!/usr/bin/env python3
"""Portable paper training entry point. Print the command unless --execute is given."""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys

from data_paths import data_path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("grpo", "drgrpo", "gpg", "js", "kalman", "ema", "kae", "kae_norm", "reinforcepp")


def make_overrides(args):
    config = json.loads((ROOT / "train_scripts/configs/policy_optimization.json").read_text())
    common, dataset = config["common"], config["datasets"][args.dataset]
    dataset = dataset.copy()
    if args.purpose == "estimation":
        dataset.update(save_freq=1, test_freq=20)
    adv, norm, loss, aggregation = "grpo", False, "vanilla", "token-mean"
    extra = {}
    if args.method == "grpo":
        norm = True
    elif args.method == "drgrpo":
        aggregation = "seq-mean-token-sum-norm"
    elif args.method == "gpg":
        adv, loss = "gpg", "gpg"
    elif args.method == "js":
        adv, aggregation = "grpo_shrinkage", "seq-mean-token-mean"
    elif args.method == "kalman":
        adv, norm = "grpo_kalman", True
        extra = {"kalman_process_variance": 1e-5, "kalman_measurement_variance": 1e-2,
                 "kalman_initial_mean": 0.0, "kalman_initial_variance": 1.0}
    elif args.method == "ema":
        adv = "grpo_kae_exp"
        extra = {"ema_beta": common["ema_beta"], "ema_min_var": common["ema_min_var"],
                 "ema_leave_one_out": True}
    elif args.method in ("kae", "kae_norm"):
        adv = "grpo_kae_tri"
        extra = kae_bandwidth_options(args, common)
        if args.method == "kae_norm":
            norm = True
            extra.update(kae_std_mode="group", kae_std_fallback=0.5,
                         ema_min_var=0.01)
    elif args.method == "reinforcepp":
        adv, norm = "reinforce_plus_plus", True
    output = Path(args.output).resolve()
    options = {
        "algorithm.adv_estimator": adv, "algorithm.norm_adv_by_std_in_grpo": norm,
        "algorithm.use_kl_in_reward": False,
        "data.train_files": str(Path(args.train_data).resolve()),
        "data.val_files": str(Path(args.val_data).resolve()),
        "data.prompt_key": "prompt", "+data.seed": args.seed,
        "data.train_batch_size": common["train_batch_size"],
        "data.max_prompt_length": dataset["prompt_length"],
        "data.max_response_length": dataset["response_length"],
        "data.filter_overlong_prompts": True, "data.truncation": "error", "data.shuffle": True,
        "+data.fixed_repeat.enabled": True, "+data.fixed_repeat.repeat_count": dataset["repeat_count"],
        "+data.prompt_trace_path": str(output / "prompt_trace.jsonl"),
        "actor_rollout_ref.model.path": args.model or dataset["model"],
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "actor_rollout_ref.actor.optim.lr": common["learning_rate"],
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.lr_warmup_steps_ratio": 0.0,
        "actor_rollout_ref.actor.optim.weight_decay": 0.01,
        "actor_rollout_ref.actor.optim.betas": [0.9, 0.999],
        "actor_rollout_ref.actor.ppo_mini_batch_size": common["ppo_mini_batch_size"],
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.clip_ratio": 0.2,
        "actor_rollout_ref.actor.clip_ratio_low": 0.2,
        "actor_rollout_ref.actor.clip_ratio_high": 0.2,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": 24000,
        "actor_rollout_ref.actor.use_kl_loss": False,
        "actor_rollout_ref.actor.entropy_coeff": 0,
        "actor_rollout_ref.actor.policy_loss.loss_mode": loss,
        "actor_rollout_ref.actor.loss_agg_mode": aggregation,
        "actor_rollout_ref.actor.fsdp_config.param_offload": False,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": False,
        "actor_rollout_ref.actor.checkpoint.save_contents": ["model"],
        "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu": 32,
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization": dataset["memory_utilization"],
        "actor_rollout_ref.rollout.n": dataset["group_size"],
        "actor_rollout_ref.rollout.temperature": common["temperature"],
        "actor_rollout_ref.rollout.top_p": common["top_p"],
        "actor_rollout_ref.rollout.top_k": common["top_k"],
        "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu": 32,
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "custom_reward_function.path": str(ROOT / "verl/utils/reward_score/kae_math_verify.py"),
        "custom_reward_function.name": "compute_score",
        "trainer.critic_warmup": 0, "trainer.logger": ["console", "wandb"],
        "trainer.val_before_train": dataset["val_before_train"],
        "trainer.project_name": "KAE-paper", "trainer.experiment_name": output.name,
        "trainer.n_gpus_per_node": dataset["gpus"], "trainer.nnodes": 1,
        "trainer.total_epochs": 1000 if args.purpose == "estimation" and args.dataset == "math" else 10000,
        "trainer.total_training_steps": args.steps if args.steps is not None else dataset["steps"],
        "trainer.save_freq": args.save_freq if args.save_freq is not None else dataset["save_freq"],
        "trainer.test_freq": dataset["test_freq"],
        "trainer.default_local_dir": str(output / "checkpoints"),
        "trainer.resume_mode": "disable", "hydra.run.dir": str(output / "hydra"),
        "ray_kwargs.ray_init.num_cpus": common["ray_cpus"],
    }
    if args.dataset == "gsm8k":
        options["actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu"] = 32
    options.update({f"+algorithm.{key}": value for key, value in extra.items()})
    return options


def hydra_value(value):
    return json.dumps(value, separators=(",", ":")) if not isinstance(value, str) else value


def positive_bandwidth(value):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise argparse.ArgumentTypeError("kernel ih must be finite and positive")
    return value


def kae_bandwidth_options(args, settings):
    width = getattr(args, "kernel_ih", None)
    after = getattr(args, "kernel_ih_after", None)
    switch = getattr(args, "kernel_switch_step", None)
    width = settings["kae_ih"] if width is None else width
    after = settings.get("kae_ih_after") if after is None else after
    switch = settings.get("kae_ih_switch_step", 0) if switch is None else switch
    if switch < 0 or (switch > 0 and after is None):
        raise ValueError("A positive bandwidth switch step requires kae_ih_after")
    return {"kae_ih": positive_bandwidth(width),
            "kae_ih_after": None if after is None else positive_bandwidth(after),
            "kae_ih_switch_step": switch}


def add_bandwidth_arguments(parser, default):
    parser.add_argument("--kernel-ih", type=positive_bandwidth,
                        help=f"Triangular KAE bandwidth product i*h (default: {default}; alpha=1/(i*h))")
    parser.add_argument("--kernel-ih-after", type=positive_bandwidth,
                        help="Optional i*h after the bandwidth switch step")
    parser.add_argument("--kernel-switch-step", type=int,
                        help="Switch after this step; 0 disables switching (default)")


def validate_bandwidth_arguments(args, parser, methods):
    if any(value is not None for value in (args.kernel_ih, args.kernel_ih_after, args.kernel_switch_step)):
        if args.method not in methods:
            parser.error("Kernel bandwidth options apply only to " + ", ".join(methods))
    if args.kernel_switch_step is not None and args.kernel_switch_step < 0:
        parser.error("--kernel-switch-step must be nonnegative")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("dapo", "math", "gsm8k"), required=True)
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--purpose", choices=("policy", "estimation"), default="policy",
                        help="estimation saves every checkpoint and uses the MSE/Assumption 6 validation schedule")
    parser.add_argument("--seed", type=int, default=1, help="Training data-sampler seed")
    parser.add_argument("--train-data", help="Override the dataset path in data/paths.json")
    parser.add_argument("--val-data", help="Override the validation path in data/paths.json")
    parser.add_argument("--model", help="Local model directory or Hugging Face identifier")
    parser.add_argument("--output", required=True, help="New run directory; never reused implicitly")
    parser.add_argument("--steps", type=int, help="Explicit override, e.g. for a functional smoke")
    parser.add_argument("--save-freq", type=int, help="Use 1 to retain steps for MSE/Assumption 6")
    add_bandwidth_arguments(parser, default=10)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args(argv)
    if args.purpose == "estimation" and (args.dataset == "dapo" or args.method != "kae"):
        parser.error("The estimation-checkpoint experiment uses KAE on GSM8K or MATH")
    validate_bandwidth_arguments(args, parser, ("kae", "kae_norm"))
    if any(value is not None and value < 1 for value in (args.steps, args.save_freq)):
        parser.error("--steps and --save-freq must be positive")
    args.train_data = args.train_data or str(data_path("training", args.dataset))
    args.val_data = args.val_data or str(data_path("validation", args.purpose, args.dataset))
    return args


def launch(args, options, experiment=None):
    command = [sys.executable, "-m", "verl.trainer.main_ppo"] + [
        f"{key}={hydra_value(value)}" for key, value in options.items()]
    print(shlex.join(command), flush=True)
    if not args.execute:
        return
    from validate_data import validate

    data = validate(Path(args.train_data))
    validate(Path(args.val_data), training=False)
    output = Path(args.output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {output}")
    output.mkdir(parents=True, exist_ok=True)
    source = ROOT / "verl/trainer/ppo/core_algos.py"
    receipt = dict(command=command, options=options, train_data=data, experiment=experiment or args.purpose,
                   core_algos_sha256=hashlib.sha256(source.read_bytes()).hexdigest())
    (output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    env.setdefault("WANDB_MODE", "offline")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def main():
    args = parse_args()
    launch(args, make_overrides(args))


if __name__ == "__main__":
    main()
