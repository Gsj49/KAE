"""One-shot and single-stream presets, sharing the normal VERL launcher."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from train import (ROOT, add_bandwidth_arguments, kae_bandwidth_options, launch,
                   make_overrides, validate_bandwidth_arguments)
from data_paths import data_path


def parse_args(experiment, argv=None):
    parser = argparse.ArgumentParser(description=f"{experiment} training (preview unless --execute)")
    if experiment == "one_shot":
        parser.add_argument("--method", choices=("grpo", "pool_oracle", "kae"), required=True)
        parser.set_defaults(dataset="math")
    else:
        parser.add_argument("--method", choices=("reinforce", "reinforce_schedule", "kae"), required=True)
        parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--train-data", help="Override data/paths.json")
    parser.add_argument("--val-data", help="Override data/paths.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model")
    parser.add_argument("--seed", type=int, default=1, help="Data-sampler seed")
    parser.add_argument("--steps", type=int, help="Override for a short training smoke")
    add_bandwidth_arguments(parser, default=5)
    parser.add_argument("--execute", action="store_true")
    parser.set_defaults(purpose="policy", save_freq=None)
    args = parser.parse_args(argv)
    if args.steps is not None and args.steps < 1:
        parser.error("--steps must be positive")
    validate_bandwidth_arguments(args, parser, ("kae",))
    if experiment == "one_shot":
        args.train_data = args.train_data or str(data_path("one_shot", "prepared"))
        args.val_data = args.val_data or str(data_path("validation", "one_shot"))
    else:
        args.train_data = args.train_data or str(data_path("training", args.dataset))
        args.val_data = args.val_data or str(data_path("validation", "single_stream", args.dataset))
    return args


def options(experiment, args):
    settings = json.loads((ROOT / "train_scripts/configs/illustrations.json").read_text())[experiment]
    base = SimpleNamespace(**vars(args))
    base.method = "kae" if args.method == "kae" else "grpo"
    result = make_overrides(base)
    if experiment == "one_shot":
        preset = settings
        batch, repeat = settings["copies"], 1
        result.update({
            "algorithm.adv_estimator": {"grpo": "grpo", "pool_oracle": "grpo_pool", "kae": "grpo_kae_tri"}[args.method],
            "algorithm.norm_adv_by_std_in_grpo": settings["norm_adv_by_std_in_grpo"],
            "actor_rollout_ref.actor.loss_agg_mode": settings["loss_agg_mode"],
            "actor_rollout_ref.actor.ppo_mini_batch_size": batch,
            "data.val_batch_size": 530, "trainer.val_before_train": False,
            "+data.fixed_repeat.enabled": False,
        })
    else:
        preset = settings["datasets"][args.dataset]
        batch = settings["train_batch_size"]
        repeat = 1 if args.method == "reinforce" else settings["repeat_count"]
        result.update({
            # Upstream GRPO's G=1 branch uses a zero baseline (REINFORCE).
            "algorithm.adv_estimator": "grpo_kae_tri" if args.method == "kae" else "grpo",
            "algorithm.norm_adv_by_std_in_grpo": args.method == "reinforce",
            "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
            "actor_rollout_ref.actor.ppo_mini_batch_size": batch // 2,
            "trainer.val_before_train": preset["val_before_train"],
            "+data.fixed_repeat.enabled": args.method != "reinforce",
        })
    result.update({
        "data.train_batch_size": batch,
        "+data.fixed_repeat.repeat_count": repeat,
        "data.max_prompt_length": preset["prompt_length"],
        "data.max_response_length": preset["response_length"],
        "actor_rollout_ref.model.path": args.model or preset["model"],
        "actor_rollout_ref.rollout.n": settings["group_size"],
        "actor_rollout_ref.rollout.gpu_memory_utilization": preset["memory_utilization"],
        "trainer.n_gpus_per_node": preset["gpus"],
        "trainer.total_training_steps": args.steps or settings["steps"],
        "trainer.save_freq": settings["save_freq"], "trainer.test_freq": settings["test_freq"],
    })
    if args.method == "kae":
        result.update({f"+algorithm.{key}": value for key, value in kae_bandwidth_options(args, settings).items()})
    return result


def run(experiment):
    args = parse_args(experiment)
    if args.execute and experiment == "one_shot":
        from prepare_oneshot import validate_oneshot
        validate_oneshot(Path(args.train_data))
    launch(args, options(experiment, args), experiment=experiment)
