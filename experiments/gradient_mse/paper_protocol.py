"""Shared paper settings and full-batch sampling for gradient-MSE comparisons."""

import hashlib
import json
from types import SimpleNamespace

import numpy as np


def math_parameter_scope():
    """MATH experiment: last five blocks, final model norm and output layer."""
    return SimpleNamespace(grad_scope="last_n", grad_last_n_layers=5,
                           grad_layer_start=-1, grad_layer_end=-1,
                           include_lm_head=True, include_model_norm=True,
                           include_embed_tokens=False)


def reinforcepp_loo_advantages(rewards, batch_mean, completion_count):
    """Z - (sum_all Z - Z)/(BG-1), without another forward/backward pass."""
    if completion_count < 2:
        raise ValueError("REINFORCE++ LOO requires at least two completions")
    return (completion_count / (completion_count - 1.0)) * (rewards - batch_mean)


def prepare_full_batch(cfg, pools, steps, rng, choose, draw, batch_means):
    """Draw the entire statistical batch before selecting any compute shard.

    A shard controls only GPU work. Its RF++ baseline includes the exact same
    target-step draws from every prompt, including prompts in other shards.
    """
    if cfg.num_prompt_shards < 1 or not 0 <= cfg.prompt_shard_index < cfg.num_prompt_shards:
        raise ValueError("Invalid compute shard")
    bundles, selection = choose(
        step_to_samples=pools, required_steps=steps, target_step=cfg.target_step_abs,
        k_eval=cfg.k_eval, t_prompts=cfg.t_prompts, rng=rng,
        sampling_with_replacement=cfg.sampling_with_replacement,
        num_prompt_shards=1, prompt_shard_index=0)
    if not bundles:
        raise ValueError("No valid prompts in the full statistical batch")
    plan = draw(bundles=bundles, required_steps=steps, k_eval=cfg.k_eval,
                m_repeats=cfg.m_repeats, rng=rng, replace=cfg.sampling_with_replacement)
    means = batch_means(bundles, plan, cfg.target_step_abs)
    uids = [bundle.uid for bundle in bundles]
    uid_hash = hashlib.sha256(json.dumps(uids, separators=(",", ":")).encode()).hexdigest()
    digest = hashlib.sha256(uid_hash.encode())
    for prompt_plan in plan:
        for repeat in prompt_plan:
            for step in steps:
                digest.update(np.asarray(repeat[step], dtype="<i8").tobytes())
    count = len(bundles) * cfg.k_eval
    if count < 2:
        raise ValueError("REINFORCE++ LOO requires at least two completions")
    metadata = dict(reinforcepp_baseline="global_batch_completion_leave_one_out",
                    reinforcepp_batch_prompts=len(bundles),
                    reinforcepp_batch_completions=count,
                    reinforcepp_batch_uid_sha256=uid_hash,
                    sampling_plan_sha256=digest.hexdigest())
    indices = list(range(cfg.prompt_shard_index, len(bundles), cfg.num_prompt_shards))
    selection.update(prompts_before_sharding=len(bundles),
                     num_prompt_shards=cfg.num_prompt_shards,
                     prompt_shard_index=cfg.prompt_shard_index,
                     selected_prompts=len(indices))
    return ([bundles[i] for i in indices], selection, [plan[i] for i in indices], means, metadata)
