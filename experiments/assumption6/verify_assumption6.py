#!/usr/bin/env python3
"""Empirically check Assumption 6 on arbitrary checkpoints and rollout data.

Assumption 6 states that, conditional on a prompt X, the outcome reward Z is
uncorrelated with the squared policy-score norm

    || grad_theta log pi_theta(Y | X) ||_2^2.

The script accepts either one row per completion or packed ``responses`` and
``rewards`` list columns.  It deliberately does not encode checkpoint names or
training steps: any Hugging Face-format checkpoint and matching rollout file can
be supplied after training.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch


QUADRATIC_TERM_DEFINITION = "mean_score_norm_sq_times_baseline_mse"


def _parse_object(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    value = value.strip()
    if not value or value[0] not in "[{(":
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return value


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    raise ValueError(f"Unsupported data format: {path}")


def _infer_column(columns: Iterable[str], requested: str | None, candidates: list[str], role: str) -> str:
    columns = list(columns)
    if requested:
        if requested not in columns:
            raise ValueError(f"Requested {role} column {requested!r} is absent. Available: {columns}")
        return requested
    for candidate in candidates:
        if candidate in columns:
            return candidate
    raise ValueError(f"Cannot infer {role} column. Tried {candidates}; available: {columns}")


def _uid_from_row(row: pd.Series, uid_col: str | None, row_number: int) -> str:
    if uid_col:
        return str(row[uid_col])
    extra = _parse_object(row.get("extra_info"))
    if isinstance(extra, dict) and extra.get("index") is not None:
        return str(extra["index"])
    for candidate in ("uid", "id", "qid", "index"):
        if candidate in row and row[candidate] is not None:
            return str(row[candidate])
    return str(row_number)


def _as_list(value: Any) -> list[Any]:
    value = _parse_object(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _prompt_to_text(prompt: Any, tokenizer: Any) -> str:
    prompt = _parse_object(prompt)
    if isinstance(prompt, list) and prompt and isinstance(prompt[0], dict):
        return tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    if isinstance(prompt, dict):
        if "content" in prompt:
            return str(prompt["content"])
        return json.dumps(prompt, ensure_ascii=False)
    return str(prompt)


def load_completions(args: argparse.Namespace, tokenizer: Any) -> pd.DataFrame:
    frame = _read_table(Path(args.data))
    prompt_col = _infer_column(frame.columns, args.prompt_col, ["prompt", "prompts", "question", "input"], "prompt")
    response_col = _infer_column(
        frame.columns,
        args.response_col,
        ["response", "responses", "completion", "completions", "output"],
        "response",
    )
    reward_col = _infer_column(
        frame.columns, args.reward_col, ["reward", "rewards", "score", "scores"], "reward"
    )
    if args.uid_col and args.uid_col not in frame.columns:
        raise ValueError(f"Requested uid column {args.uid_col!r} is absent")

    records: list[dict[str, Any]] = []
    for row_number, row in frame.iterrows():
        uid = _uid_from_row(row, args.uid_col, int(row_number))
        prompt_text = _prompt_to_text(row[prompt_col], tokenizer)
        responses = _as_list(row[response_col])
        rewards = _as_list(row[reward_col])
        if len(responses) != len(rewards):
            raise ValueError(
                f"uid={uid}: response/reward counts differ ({len(responses)} versus {len(rewards)})"
            )
        for completion_id, (response, reward) in enumerate(zip(responses, rewards)):
            if response is None or reward is None:
                continue
            records.append(
                {
                    "uid": uid,
                    "completion_id": completion_id,
                    "prompt": prompt_text,
                    "response": str(response),
                    "reward": float(reward),
                }
            )
    out = pd.DataFrame(records)
    if out.empty:
        raise ValueError("No usable prompt/response/reward triples were found")
    if args.num_prompt_shards < 1:
        raise ValueError("--num-prompt-shards must be >= 1")
    if not 0 <= args.prompt_shard_index < args.num_prompt_shards:
        raise ValueError(
            "--prompt-shard-index must be in "
            f"[0, {args.num_prompt_shards}); got {args.prompt_shard_index}"
        )
    ordered_uids = out["uid"].drop_duplicates().tolist()
    uid_shards = np.array_split(np.asarray(ordered_uids, dtype=object), args.num_prompt_shards)
    selected_uids = set(uid_shards[args.prompt_shard_index].tolist())
    out = out[out["uid"].isin(selected_uids)]
    if out.empty:
        raise ValueError(
            f"Prompt shard {args.prompt_shard_index}/{args.num_prompt_shards} is empty"
        )
    if args.max_prompts is not None:
        keep = out["uid"].drop_duplicates().iloc[: args.max_prompts]
        out = out[out["uid"].isin(set(keep))]
    if args.max_completions_per_prompt is not None:
        out = out.groupby("uid", sort=False, group_keys=False).head(args.max_completions_per_prompt)
    return out.reset_index(drop=True)


def select_score_parameters(
    model: torch.nn.Module,
    scope: str,
    last_n_layers: int,
    parameter_regex: str | None,
) -> list[tuple[str, torch.nn.Parameter]]:
    named = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if scope == "all":
        selected = named
    elif scope == "regex":
        if not parameter_regex:
            raise ValueError("--parameter-regex is required with --parameter-scope regex")
        pattern = re.compile(parameter_regex)
        selected = [(name, param) for name, param in named if pattern.search(name)]
    elif scope == "last_n_layers":
        layer_ids: list[int] = []
        for name, _ in named:
            match = re.search(r"(?:layers|blocks|h)\.(\d+)\.", name)
            if match:
                layer_ids.append(int(match.group(1)))
        if not layer_ids:
            raise ValueError("Could not infer transformer layer indices; use --parameter-scope regex or all")
        kept_layers = set(sorted(set(layer_ids))[-last_n_layers:])

        def keep(name: str) -> bool:
            match = re.search(r"(?:layers|blocks|h)\.(\d+)\.", name)
            in_tail = bool(match and int(match.group(1)) in kept_layers)
            is_head_or_final_norm = "lm_head" in name or bool(
                re.search(r"(?:model|transformer)\.(?:norm|ln_f)\.", name)
            )
            return in_tail or is_head_or_final_norm

        selected = [(name, param) for name, param in named if keep(name)]
    else:
        raise ValueError(f"Unknown parameter scope: {scope}")
    if not selected:
        raise ValueError("Parameter selection is empty")
    return selected


def _encode_pair(
    tokenizer: Any,
    prompt: str,
    response: str,
    max_prompt_tokens: int,
    max_response_tokens: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    prompt_ids = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")["input_ids"][0]
    response_ids = tokenizer(response, add_special_tokens=False, return_tensors="pt")["input_ids"][0]
    prompt_ids = prompt_ids[-max_prompt_tokens:]
    response_ids = response_ids[:max_response_tokens]
    if response_ids.numel() == 0:
        raise ValueError("Completion tokenized to zero tokens")
    input_ids = torch.cat([prompt_ids, response_ids])
    labels = torch.full_like(input_ids, -100)
    labels[prompt_ids.numel() :] = response_ids
    return input_ids.unsqueeze(0), labels.unsqueeze(0)


def _input_device(model: torch.nn.Module) -> torch.device:
    return model.get_input_embeddings().weight.device


def score_norm_squared(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    response: str,
    parameters: list[tuple[str, torch.nn.Parameter]],
    max_prompt_tokens: int,
    max_response_tokens: int,
    score_reduction: str,
) -> tuple[float, float, int]:
    input_ids, labels = _encode_pair(
        tokenizer, prompt, response, max_prompt_tokens=max_prompt_tokens, max_response_tokens=max_response_tokens
    )
    input_ids = input_ids.to(_input_device(model))
    labels = labels.to(input_ids.device)
    attention_mask = torch.ones_like(input_ids)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
    logits = outputs.logits[:, :-1].float()
    shifted_labels = labels[:, 1:].to(logits.device)
    completion_mask = shifted_labels.ne(-100)
    safe_labels = shifted_labels.masked_fill(~completion_mask, 0)
    token_log_probs = torch.log_softmax(logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    sequence_log_prob = (token_log_probs * completion_mask).sum()
    token_count = int(completion_mask.sum().item())
    objective = sequence_log_prob if score_reduction == "sum" else sequence_log_prob / max(token_count, 1)
    gradients = torch.autograd.grad(
        objective,
        [param for _, param in parameters],
        allow_unused=True,
        create_graph=False,
        retain_graph=False,
    )
    norm_sq = torch.zeros((), dtype=torch.float64, device=input_ids.device)
    for gradient in gradients:
        if gradient is not None:
            norm_sq = norm_sq + gradient.detach().double().square().sum().to(norm_sq.device)
    return float(norm_sq.cpu()), float(sequence_log_prob.detach().cpu()), token_count


def _safe_corr(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return math.nan
    if method == "spearman":
        x = pd.Series(x).rank(method="average").to_numpy()
        y = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(x, y)[0, 1])


def _pooled_within_prompt_stats(per_prompt: pd.DataFrame) -> tuple[float, float]:
    cross = float(per_prompt["within_cross_product"].sum())
    reward_ss = float(per_prompt["within_reward_ss"].sum())
    score_ss = float(per_prompt["within_score_ss"].sum())
    count = int(per_prompt["n"].sum())
    covariance = cross / count if count > 0 else math.nan
    denominator = math.sqrt(reward_ss * score_ss)
    correlation = cross / denominator if denominator > 0 else math.nan
    return covariance, correlation


def _bootstrap_prompt_summary(
    per_prompt: pd.DataFrame, n_bootstrap: int, seed: int
) -> dict[str, float | int]:
    if n_bootstrap <= 0 or per_prompt.empty:
        return {"replicates": 0}
    rng = np.random.default_rng(seed)
    pooled_corr: list[float] = []
    mean_cov: list[float] = []
    for _ in range(n_bootstrap):
        sample = per_prompt.iloc[rng.integers(0, len(per_prompt), size=len(per_prompt))]
        _, correlation = _pooled_within_prompt_stats(sample)
        pooled_corr.append(correlation)
        mean_cov.append(float(sample["cov_reward_score_norm_sq"].mean()))

    def interval(values: list[float]) -> tuple[float, float]:
        array = np.asarray(values, dtype=float)
        array = array[np.isfinite(array)]
        if array.size == 0:
            return math.nan, math.nan
        lo, hi = np.percentile(array, [2.5, 97.5])
        return float(lo), float(hi)

    corr_lo, corr_hi = interval(pooled_corr)
    cov_lo, cov_hi = interval(mean_cov)
    return {
        "replicates": int(n_bootstrap),
        "pooled_within_prompt_pearson_ci95_lo": corr_lo,
        "pooled_within_prompt_pearson_ci95_hi": corr_hi,
        "mean_conditional_covariance_ci95_lo": cov_lo,
        "mean_conditional_covariance_ci95_hi": cov_hi,
    }


def _load_baselines(path: str) -> pd.DataFrame:
    baselines = _read_table(Path(path)).copy()
    # Native format produced by this experiment.
    if {"uid", "baseline_value", "oracle_value"}.issubset(baselines.columns):
        if "method" not in baselines.columns:
            baselines["method"] = "baseline"
    # Alternative long format with method, estimated mean and reference value.
    elif {"uid", "which", "mean_est", "gt_true"}.issubset(baselines.columns):
        baselines = baselines.rename(
            columns={"which": "method", "mean_est": "baseline_value", "gt_true": "oracle_value",
                     "mse": "baseline_mse"}
        )
    else:
        raise ValueError(
            "Baseline table must contain either "
            "uid,baseline_value,oracle_value,baseline_mse[,method] "
            "or uid,which,mean_est,gt_true,mse"
        )
    if "baseline_mse" not in baselines.columns:
        raise ValueError(
            "Baseline table requires baseline_mse: the mean squared error of the "
            "individual estimates, not the squared error of their mean. "
            "Rebuild the table with build_baseline_table.py."
        )
    baselines = baselines[["uid", "method", "baseline_value", "oracle_value", "baseline_mse"]].copy()
    numeric = baselines[["baseline_value", "oracle_value", "baseline_mse"]].to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (baselines["baseline_mse"] < 0).any():
        raise ValueError("Baseline moments must be finite and baseline_mse nonnegative")
    squared_bias = (baselines["baseline_value"] - baselines["oracle_value"]) ** 2
    if (baselines["baseline_mse"] + 1e-12 < squared_bias).any():
        raise ValueError("baseline_mse cannot be smaller than the squared mean error")
    baselines["uid"] = baselines["uid"].astype(str)
    baselines["method"] = baselines["method"].astype(str)
    if baselines.duplicated(["uid", "method"]).any():
        raise ValueError("Baseline table has duplicate uid/method rows")
    return baselines


def summarize(
    per_completion: pd.DataFrame,
    baseline_table: str | None,
    bootstrap_replicates: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    frame = per_completion.copy()
    rows: list[dict[str, Any]] = []
    for uid, group in frame.groupby("uid", sort=False):
        reward = group["reward"].to_numpy(dtype=float)
        norm_sq = group["score_norm_sq"].to_numpy(dtype=float)
        reward_centered = reward - reward.mean()
        norm_centered = norm_sq - norm_sq.mean()
        cross_product = float(np.sum(reward_centered * norm_centered))
        reward_ss = float(np.sum(reward_centered**2))
        score_ss = float(np.sum(norm_centered**2))
        covariance = cross_product / len(group)
        mean_score = float(norm_sq.mean())
        weighted_baseline = float(np.mean(reward * norm_sq) / mean_score) if mean_score > 0 else math.nan
        row: dict[str, Any] = {
            "uid": uid,
            "n": int(group.shape[0]),
            "reward_mean": float(reward.mean()),
            "score_norm_sq_mean": mean_score,
            "cov_reward_score_norm_sq": covariance,
            "pearson": _safe_corr(reward, norm_sq, "pearson"),
            "spearman": _safe_corr(reward, norm_sq, "spearman"),
            "score_weighted_optimal_baseline": weighted_baseline,
            "optimal_minus_value": weighted_baseline - float(reward.mean()),
            "within_cross_product": cross_product,
            "within_reward_ss": reward_ss,
            "within_score_ss": score_ss,
        }
        rows.append(row)
    per_prompt = pd.DataFrame(rows)
    pooled_covariance, pooled_correlation = _pooled_within_prompt_stats(per_prompt)
    valid_pearson = per_prompt["pearson"].dropna().to_numpy(dtype=float)
    valid_cov = per_prompt["cov_reward_score_norm_sq"].dropna().to_numpy(dtype=float)
    summary: dict[str, Any] = {
        "assumption": "Cov(Z, ||grad log pi(Y|X)||^2 | X) = 0",
        "num_prompts": int(per_prompt.shape[0]),
        "num_completions": int(frame.shape[0]),
        "num_prompts_with_identifiable_correlation": int(valid_pearson.size),
        "mean_conditional_covariance": float(valid_cov.mean()) if valid_cov.size else math.nan,
        "median_conditional_covariance": float(np.median(valid_cov)) if valid_cov.size else math.nan,
        "mean_conditional_pearson": float(valid_pearson.mean()) if valid_pearson.size else math.nan,
        "median_abs_conditional_pearson": float(np.median(np.abs(valid_pearson))) if valid_pearson.size else math.nan,
        "pooled_within_prompt_covariance": pooled_covariance,
        "pooled_within_prompt_pearson": pooled_correlation,
        "mean_abs_optimal_minus_value": float(per_prompt["optimal_minus_value"].abs().mean()),
        "fraction_abs_pearson_below_0_1": (
            float(np.mean(np.abs(valid_pearson) < 0.1)) if valid_pearson.size else math.nan
        ),
        "baseline_cross_term_computed": baseline_table is not None,
    }
    summary["prompt_bootstrap"] = _bootstrap_prompt_summary(
        per_prompt, n_bootstrap=bootstrap_replicates, seed=bootstrap_seed
    )

    cross_terms = pd.DataFrame()
    if baseline_table:
        baselines = _load_baselines(baseline_table)
        for method, group in baselines.groupby("method", sort=False):
            if not set(per_prompt["uid"]).issubset(set(group["uid"])):
                raise ValueError(f"Baseline table is missing scored prompts for {method}")
        cross_terms = per_prompt.merge(baselines, on="uid", how="inner", validate="one_to_many")
        cross_terms["baseline_error"] = cross_terms["baseline_value"] - cross_terms["oracle_value"]
        # Expand E[((Z - V) - (baseline - V))^2 q | X].
        # baseline_error estimates E[baseline - V | X]. The quadratic
        # term uses E[(baseline - V)^2 | X], including baseline variance.
        cross_terms["ae_cross_term"] = (
            -2.0
            * cross_terms["cov_reward_score_norm_sq"]
            * cross_terms["baseline_error"]
        )
        cross_terms["quadratic_value_error_term"] = (
            cross_terms["score_norm_sq_mean"] * cross_terms["baseline_mse"]
        )
        denominator = cross_terms["quadratic_value_error_term"].abs()
        cross_terms["abs_cross_over_quadratic"] = np.where(
            denominator > 0, cross_terms["ae_cross_term"].abs() / denominator, np.nan
        )
        method_summary: dict[str, Any] = {}
        for method, group in cross_terms.groupby("method", sort=False):
            method_summary[str(method)] = {
                "num_prompts": int(len(group)),
                "mean_cross_term": float(group["ae_cross_term"].mean()),
                "mean_abs_cross_term": float(group["ae_cross_term"].abs().mean()),
                "mean_quadratic_value_error_term": float(group["quadratic_value_error_term"].mean()),
                "median_abs_cross_over_quadratic": float(group["abs_cross_over_quadratic"].median()),
            }
        summary["baseline_cross_terms_by_method"] = method_summary
        summary["quadratic_term_definition"] = QUADRATIC_TERM_DEFINITION
        summary["cross_term_prompt_scope"] = "all_scored_prompts"
    return per_prompt, cross_terms, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Hugging Face model directory or repository id")
    parser.add_argument("--data", required=True, help="Parquet, CSV, JSON, or JSONL rollout data")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-col")
    parser.add_argument("--response-col")
    parser.add_argument("--reward-col")
    parser.add_argument("--uid-col")
    parser.add_argument(
        "--baseline-table",
        help=(
            "Optional long table with uid,baseline_value,oracle_value[,method], or the paper MSE "
            "format uid,which,mean_est,gt_true"
        ),
    )
    parser.add_argument("--parameter-scope", choices=["all", "last_n_layers", "regex"], default="last_n_layers")
    parser.add_argument("--last-n-layers", type=int, default=5)
    parser.add_argument("--parameter-regex")
    parser.add_argument("--score-reduction", choices=["sum", "mean"], default="sum")
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--max-response-tokens", type=int, default=2048)
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--max-completions-per-prompt", type=int)
    parser.add_argument("--num-prompt-shards", type=int, default=1)
    parser.add_argument("--prompt-shard-index", type=int, default=0)
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="bfloat16")
    parser.add_argument("--device-map", default="auto", help="Transformers device_map; use 'none' for one device")
    parser.add_argument("--device", default="cuda", help="Used only when --device-map none")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-seed", type=int, default=20260817)
    return parser.parse_args()


def main() -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = {
        "auto": "auto",
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[args.dtype]
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
    load_kwargs: dict[str, Any] = {"dtype": dtype, "trust_remote_code": True}
    if args.device_map.lower() != "none":
        load_kwargs["device_map"] = args.device_map
    model = AutoModelForCausalLM.from_pretrained(args.checkpoint, **load_kwargs)
    if args.device_map.lower() == "none":
        model.to(args.device)
    model.eval()
    model.config.use_cache = False

    completions = load_completions(args, tokenizer)
    selected = select_score_parameters(
        model,
        scope=args.parameter_scope,
        last_n_layers=args.last_n_layers,
        parameter_regex=args.parameter_regex,
    )
    selected_parameter_count = sum(param.numel() for _, param in selected)
    print(
        f"[info] prompts={completions['uid'].nunique()} completions={len(completions)} "
        f"selected_tensors={len(selected)} selected_parameters={selected_parameter_count:,}"
    )

    rows: list[dict[str, Any]] = []
    for row_number, row in completions.iterrows():
        model.zero_grad(set_to_none=True)
        norm_sq, log_prob, token_count = score_norm_squared(
            model,
            tokenizer,
            prompt=row["prompt"],
            response=row["response"],
            parameters=selected,
            max_prompt_tokens=args.max_prompt_tokens,
            max_response_tokens=args.max_response_tokens,
            score_reduction=args.score_reduction,
        )
        rows.append(
            {
                "uid": row["uid"],
                "completion_id": int(row["completion_id"]),
                "reward": float(row["reward"]),
                "score_norm_sq": norm_sq,
                "sequence_log_prob": log_prob,
                "response_tokens": token_count,
            }
        )
        if (row_number + 1) % 10 == 0 or row_number + 1 == len(completions):
            print(f"[progress] {row_number + 1}/{len(completions)}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_completion = pd.DataFrame(rows)
    per_prompt, cross_terms, summary = summarize(
        per_completion,
        args.baseline_table,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_seed=args.bootstrap_seed,
    )
    summary.update(
        checkpoint=args.checkpoint,
        data=args.data,
        parameter_scope=args.parameter_scope,
        last_n_layers=args.last_n_layers if args.parameter_scope == "last_n_layers" else None,
        parameter_regex=args.parameter_regex,
        score_reduction=args.score_reduction,
        selected_parameter_tensors=len(selected),
        selected_parameter_count=selected_parameter_count,
        seed=args.seed,
        num_prompt_shards=args.num_prompt_shards,
        prompt_shard_index=args.prompt_shard_index,
        prompt_uids=completions["uid"].drop_duplicates().tolist(),
    )
    per_completion.to_parquet(output_dir / "per_completion.parquet", index=False)
    per_prompt.to_csv(output_dir / "per_prompt.csv", index=False)
    if not cross_terms.empty:
        cross_terms.to_csv(output_dir / "baseline_cross_terms.csv", index=False)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False, allow_nan=True)
    print(f"[saved] {output_dir}")


if __name__ == "__main__":
    main()
