import argparse
import ast
import glob
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from paper_protocol import prepare_full_batch, reinforcepp_loo_advantages
from boundary import POLICY as TOKENIZATION_POLICY, tokenize_fixed_prompt

STEP_PAT = re.compile(r"step[_-]?(\d+)")
UID_PRIORITY = ("uid", "id", "qid", "gidx", "pos")
LAYER_PAT = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")

@dataclass(frozen=True)
class RunConfig:
    samples_glob: str
    target_step_abs: int
    history_steps: List[int]
    k_eval: int
    m_repeats: int
    t_prompts: int
    model_path: str
    prompt_col: str
    linear_alpha: float
    ema_min_var: float
    epsilon: float
    device: str
    cov_device: str
    dtype: str
    seed: int
    out_dir: str
    trust_remote_code: bool
    logprob_micro_batch: int
    max_cont_tokens: int
    grad_scope: str
    grad_last_n_layers: int
    grad_layer_start: int
    grad_layer_end: int
    include_lm_head: bool
    include_model_norm: bool
    include_embed_tokens: bool
    enable_gradient_checkpointing: bool
    disable_kv_cache: bool
    sampling_with_replacement: bool
    num_prompt_shards: int
    prompt_shard_index: int

@dataclass
class StepSample:
    messages: Optional[List[Dict[str, Any]]]
    responses: List[str]
    rewards: np.ndarray

@dataclass
class PromptBundle:
    uid: str
    messages: List[Dict[str, Any]]
    step_samples: Dict[int, StepSample]

class RunningTraceReferenceError:
    """
    Tracks oracle-centered mean squared gradient error:
      trace_err = (1 / M) sum_m ||g_m - g_ref||_2^2.
    """
    def __init__(
        self,
        params: Sequence[torch.nn.Parameter],
        reference: Sequence[torch.Tensor],
        device: str,
    ):
        self.params = list(params)
        self.device = device
        self.n = 0
        self.reference = list(reference)
        self.sqerr_sum: List[torch.Tensor] = []
        for ref in self.reference:
            self.sqerr_sum.append(torch.zeros_like(ref, dtype=torch.float32, device=device))

    @torch.no_grad()
    def update_from_grads(self) -> None:
        self.n += 1
        for i, p in enumerate(self.params):
            if p.grad is None:
                diff = -self.reference[i]
            else:
                g = p.grad.detach().to(dtype=torch.float32, device=self.reference[i].device)
                diff = g - self.reference[i]
            self.sqerr_sum[i].add_(diff * diff)

    @torch.no_grad()
    def trace_ref_error(self) -> float:
        if self.n < 1:
            return 0.0
        trace_val = 0.0
        for i in range(len(self.params)):
            trace_val += (self.sqerr_sum[i] / self.n).sum().item()
        return trace_val

    @torch.no_grad()
    def trace_cov(self) -> float:
        return self.trace_ref_error()


@torch.no_grad()
def grad_sq_norm(params: Sequence[torch.nn.Parameter]) -> float:
    total = 0.0
    for p in params:
        if p.grad is not None:
            g = p.grad.detach().to(dtype=torch.float32)
            total += float((g * g).sum().item())
    return total


@torch.no_grad()
def capture_current_grads(
    params: Sequence[torch.nn.Parameter],
    device: str,
) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    for p in params:
        if p.grad is None:
            out.append(torch.zeros_like(p, dtype=torch.float32, device=device))
        else:
            out.append(p.grad.detach().to(dtype=torch.float32, device=device).clone())
    return out

def extract_step_num(path: str) -> Optional[int]:
    match = STEP_PAT.search(path)
    if match is None:
        return None
    return int(match.group(1))

def order_paths_by_step(paths: Sequence[str]) -> List[Tuple[int, str]]:
    out: List[Tuple[int, str]] = []
    for p in paths:
        step = extract_step_num(p)
        if step is None:
            continue
        out.append((step, p))
    out.sort(key=lambda x: x[0])
    return out

def maybe_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        if isinstance(x, float) and math.isnan(x):
            return None
        return float(x)
    except (TypeError, ValueError):
        return None

def to_list(x: Any) -> List[Any]:
    if isinstance(x, list):
        return x
    if isinstance(x, tuple):
        return list(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return []

def parse_extra_info_index(extra_info: Any) -> Optional[str]:
    if extra_info is None:
        return None
    if isinstance(extra_info, dict):
        idx = extra_info.get("index")
        return str(idx) if idx is not None else None
    if hasattr(extra_info, "get"):
        idx = extra_info.get("index")
        return str(idx) if idx is not None else None
    if isinstance(extra_info, str):
        parsed: Any = None
        try:
            parsed = json.loads(extra_info)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(extra_info)
            except (ValueError, SyntaxError):
                parsed = None
        if isinstance(parsed, dict) and parsed.get("index") is not None:
            return str(parsed["index"])
    return None

def resolve_uid(row: Mapping[str, Any]) -> Optional[str]:
    if "extra_info" in row:
        idx = parse_extra_info_index(row["extra_info"])
        if idx is not None:
            return idx

    for col in UID_PRIORITY:
        if col in row:
            val = row[col]
            fval = maybe_float(val)
            if fval is not None and float(fval).is_integer():
                return str(int(fval))
            if val is not None and not (isinstance(val, float) and math.isnan(val)):
                return str(val)
    return None


def parse_layer_id_from_param_name(name: str) -> Optional[int]:
    match = LAYER_PAT.search(name)
    if match is None:
        return None
    return int(match.group(1))


def configure_grad_parameter_scope(
    model: torch.nn.Module,
    cfg: RunConfig,
) -> Tuple[List[torch.nn.Parameter], Dict[str, Any]]:
    named_params = list(model.named_parameters())
    all_layer_ids: List[int] = sorted(
        {lid for name, _ in named_params for lid in [parse_layer_id_from_param_name(name)] if lid is not None}
    )
    selected_layer_ids: Optional[Set[int]] = None
    if cfg.grad_scope == "all":
        selected_layer_ids = None
    elif cfg.grad_scope == "last_n":
        if not all_layer_ids:
            raise ValueError("No transformer layer ids found in model parameters; cannot use grad_scope=last_n.")
        if cfg.grad_last_n_layers <= 0:
            raise ValueError("grad_last_n_layers must be > 0 when grad_scope=last_n.")
        selected_layer_ids = set(all_layer_ids[-cfg.grad_last_n_layers :])
    elif cfg.grad_scope == "range":
        if cfg.grad_layer_start < 0 or cfg.grad_layer_end < 0:
            raise ValueError("grad_layer_start and grad_layer_end must be >= 0 when grad_scope=range.")
        if cfg.grad_layer_start > cfg.grad_layer_end:
            raise ValueError("grad_layer_start must be <= grad_layer_end.")
        selected_layer_ids = set(range(cfg.grad_layer_start, cfg.grad_layer_end + 1))
    else:
        raise ValueError(f"Unsupported grad_scope={cfg.grad_scope}")

    tracked_params: List[torch.nn.Parameter] = []
    tracked_param_names: List[str] = []
    for name, param in named_params:
        keep = False
        if selected_layer_ids is None:
            keep = True
        else:
            layer_id = parse_layer_id_from_param_name(name)
            if layer_id is not None and layer_id in selected_layer_ids:
                keep = True
            if cfg.include_lm_head and (name == "lm_head.weight" or name.startswith("lm_head.")):
                keep = True
            if cfg.include_model_norm and (name == "model.norm.weight" or name == "model.norm.bias" or name.startswith("model.norm.")):
                keep = True
            if cfg.include_embed_tokens and ("embed_tokens" in name):
                keep = True
        param.requires_grad_(keep)
        if keep:
            tracked_params.append(param)
            tracked_param_names.append(name)

    if len(tracked_params) == 0:
        raise ValueError("No trainable parameters selected for gradient variance computation.")
    scope_info: Dict[str, Any] = {
        "grad_scope": cfg.grad_scope,
        "grad_last_n_layers": cfg.grad_last_n_layers,
        "grad_layer_start": cfg.grad_layer_start,
        "grad_layer_end": cfg.grad_layer_end,
        "include_lm_head": cfg.include_lm_head,
        "include_model_norm": cfg.include_model_norm,
        "include_embed_tokens": cfg.include_embed_tokens,
        "num_params_total": len(named_params),
        "num_params_tracked": len(tracked_params),
        "all_layer_ids_minmax": [int(min(all_layer_ids)), int(max(all_layer_ids))] if all_layer_ids else None,
        "selected_layer_ids": sorted(int(x) for x in selected_layer_ids) if selected_layer_ids is not None else "all",
        "tracked_param_name_preview": tracked_param_names[:32],
    }
    return tracked_params, scope_info

def normalize_messages(raw: Any) -> Optional[List[Dict[str, Any]]]:
    if raw is None:
        return None
    if isinstance(raw, np.ndarray):
        raw = raw.tolist()
    if isinstance(raw, str):
        parsed: Any = None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            try:
                parsed = ast.literal_eval(raw)
            except (ValueError, SyntaxError):
                return None
        raw = parsed
    if not isinstance(raw, list):
        return None
    msgs: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
            if role is None or content is None:
                continue
            msgs.append({"role": str(role), "content": content if isinstance(content, str) else str(content)})
    return msgs if msgs else None

def normalize_responses_rewards(responses_raw: Any, rewards_raw: Any) -> Tuple[List[str], np.ndarray]:
    responses = to_list(responses_raw)
    rewards = to_list(rewards_raw)
    n = min(len(responses), len(rewards))
    out_texts: List[str] = []
    out_rewards: List[float] = []
    for i in range(n):
        reward_val = maybe_float(rewards[i])
        if reward_val is None:
            continue
        resp = responses[i]
        if resp is None:
            continue
        out_texts.append(str(resp))
        out_rewards.append(float(reward_val))
    return out_texts, np.asarray(out_rewards, dtype=np.float32)

def load_step_samples(path: str, prompt_col: str) -> Tuple[Dict[str, StepSample], Dict[str, int]]:
    df = pd.read_parquet(path)
    required = {"responses", "rewards"}
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path} missing required columns: {missing}")
    if prompt_col not in df.columns:
        raise ValueError(f"{path} missing prompt column: {prompt_col}")
    cols = df.columns.tolist()
    uid_to_sample: Dict[str, StepSample] = {}
    skipped_no_uid = 0
    skipped_bad_messages = 0
    skipped_empty_rr = 0
    duplicate_uid = 0
    for values in df.itertuples(index=False, name=None):
        row = dict(zip(cols, values))
        uid = resolve_uid(row)
        if uid is None:
            skipped_no_uid += 1
            continue
        if uid in uid_to_sample:
            duplicate_uid += 1
            continue
        messages = normalize_messages(row.get(prompt_col))
        if messages is None:
            skipped_bad_messages += 1
            continue
        responses, rewards = normalize_responses_rewards(row.get("responses"), row.get("rewards"))
        if rewards.size == 0 or len(responses) == 0:
            skipped_empty_rr += 1
            continue
        uid_to_sample[uid] = StepSample(messages=messages, responses=responses, rewards=rewards)
    stats = {
        "rows_total": len(df),
        "rows_loaded": len(uid_to_sample),
        "skipped_no_uid": skipped_no_uid,
        "skipped_bad_messages": skipped_bad_messages,
        "skipped_empty_rr": skipped_empty_rr,
        "duplicate_uid": duplicate_uid,
    }
    return uid_to_sample, stats

def build_history_steps(target_step_abs: int, history_steps: Optional[List[int]], history_len: int) -> List[int]:
    if history_steps:
        hs = sorted(set(int(s) for s in history_steps))
    else:
        start = max(1, target_step_abs - int(history_len))
        hs = list(range(start, target_step_abs))
    for s in hs:
        if s >= target_step_abs:
            raise ValueError(f"history step {s} must be < target_step_abs={target_step_abs}")
    return hs

def pad_and_stack(seqs: List[torch.Tensor], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    batch_size = len(seqs)
    max_len = max(int(s.numel()) for s in seqs) if batch_size > 0 else 0
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long)
    attn_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i, seq in enumerate(seqs):
        seq_len = int(seq.numel())
        if seq_len > 0:
            input_ids[i, :seq_len] = seq
            attn_mask[i, :seq_len] = 1
    return input_ids, attn_mask

def batched_logprob_continuations(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    cont_ids_list: List[torch.Tensor],
) -> torch.Tensor:
    device = model.device
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    full_sequences: List[torch.Tensor] = []
    prompt_len = int(prompt_ids.numel())
    for cont in cont_ids_list:
        full_sequences.append(torch.cat([prompt_ids, cont], dim=0))
    input_ids, attn_mask = pad_and_stack(full_sequences, pad_id=pad_id)
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)
    logits = model(input_ids=input_ids, attention_mask=attn_mask).logits
    logp_all = torch.log_softmax(logits, dim=-1)
    lp_list: List[torch.Tensor] = []
    for i, cont in enumerate(cont_ids_list):
        cont_len = int(cont.numel())
        if cont_len == 0:
            lp_list.append(torch.zeros((), device=device, dtype=torch.float32))
            continue
        pred = logp_all[i, prompt_len - 1 : prompt_len + cont_len - 1, :]
        tgt = cont.to(device).unsqueeze(1)
        lp = pred.gather(1, tgt).squeeze(1).sum()
        lp_list.append(lp)
    return torch.stack(lp_list, dim=0)

def tokenize_prompt(tokenizer: AutoTokenizer, messages: List[Dict[str, Any]]) -> Tuple[str, torch.Tensor]:
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    enc = tokenizer(prompt_text, return_tensors="pt")
    return prompt_text, enc["input_ids"][0].detach().cpu()

def tokenize_responses_as_continuations(
    tokenizer: AutoTokenizer,
    prompt_text: str,
    prompt_ids: torch.Tensor,
    responses: Sequence[str],
) -> Tuple[List[torch.Tensor], int]:
    """Keep the conditioning prompt fixed and losslessly repair boundary merges."""
    continuations, repairs = tokenize_fixed_prompt(tokenizer, prompt_text, prompt_ids, responses)
    return continuations, len(repairs)


def maybe_truncate_cont_ids(
    cont_ids: Sequence[torch.Tensor],
    max_cont_tokens: int,
) -> Tuple[List[torch.Tensor], int]:
    if max_cont_tokens <= 0:
        return [x.to(torch.long) for x in cont_ids], 0
    out: List[torch.Tensor] = []
    truncated = 0
    for seq in cont_ids:
        if int(seq.numel()) > max_cont_tokens:
            out.append(seq[:max_cont_tokens].to(torch.long))
            truncated += 1
        else:
            out.append(seq.to(torch.long))
    return out, truncated


def backward_prompt_loss_microbatched(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_ids: torch.Tensor,
    cont_ids_list: Sequence[torch.Tensor],
    adv: torch.Tensor,
    micro_batch_size: int,
) -> None:
    total_n = len(cont_ids_list)
    if total_n == 0:
        return
    mb = int(micro_batch_size) if int(micro_batch_size) > 0 else total_n
    mb = min(mb, total_n)
    model.zero_grad(set_to_none=True)
    denom = float(total_n)
    for start in range(0, total_n, mb):
        end = min(start + mb, total_n)
        chunk_cont_ids = list(cont_ids_list[start:end])
        lp_chunk = batched_logprob_continuations(
            model=model,
            tokenizer=tokenizer,
            prompt_ids=prompt_ids,
            cont_ids_list=chunk_cont_ids,
        )
        adv_chunk = adv[start:end]
        chunk_loss = -(adv_chunk * lp_chunk).sum() / denom
        chunk_loss.backward()
def compute_mean_var(rewards: np.ndarray) -> Tuple[float, float]:
    mean_val = float(np.mean(rewards))
    var_val = float(np.mean((rewards - mean_val) ** 2))
    return mean_val, var_val

def compute_kae_baseline(
    target_step: int,
    step_stats: Mapping[int, Tuple[float, float]],
    linear_alpha: float,
    ema_min_var: float,
) -> Tuple[float, float, float, Dict[int, float]]:
    weighted: List[Tuple[float, float, float, int]] = []
    for step, (mu, var) in step_stats.items():
        delta = target_step - step
        weight = max(1.0 - linear_alpha * float(delta), 0.0)
        if weight <= 0.0:
            continue
        weighted.append((weight, mu, max(var, ema_min_var), step))
    if not weighted:
        mu_t, var_t = step_stats[target_step]
        return mu_t, max(var_t, ema_min_var), 1.0, {target_step: 1.0}
    weight_sum = float(sum(w for w, _, _, _ in weighted))
    mu_bar = float(sum(w * mu for w, mu, _, _ in weighted) / weight_sum)
    var_bar = float(sum(w * var for w, _, var, _ in weighted) / weight_sum)
    step_weight = {step: float(w) for w, _, _, step in weighted}
    return mu_bar, var_bar, weight_sum, step_weight

def choose_prompt_bundles(
    step_to_samples: Mapping[int, Dict[str, StepSample]],
    required_steps: Sequence[int],
    target_step: int,
    k_eval: int,
    t_prompts: int,
    rng: np.random.Generator,
    sampling_with_replacement: bool,
    num_prompt_shards: int,
    prompt_shard_index: int,
) -> Tuple[List[PromptBundle], Dict[str, int]]:
    uid_sets = [set(step_to_samples[s].keys()) for s in required_steps]
    common_uids = set.intersection(*uid_sets) if uid_sets else set()
    skipped_too_short = 0
    skipped_bad_prompt = 0
    bundles: List[PromptBundle] = []
    for uid in sorted(common_uids):
        target_sample = step_to_samples[target_step][uid]
        if target_sample.messages is None:
            skipped_bad_prompt += 1
            continue
        step_payload: Dict[int, StepSample] = {}
        ok = True
        for step in required_steps:
            sample = step_to_samples[step][uid]
            if sampling_with_replacement:
                if sample.rewards.size == 0 or len(sample.responses) == 0:
                    skipped_too_short += 1
                    ok = False
                    break
            else:
                if sample.rewards.size < k_eval or len(sample.responses) < k_eval:
                    skipped_too_short += 1
                    ok = False
                    break
            step_payload[step] = sample
        if not ok:
            continue
        bundles.append(
            PromptBundle(
                uid=uid,
                messages=target_sample.messages,
                step_samples=step_payload,
            )
        )
    rng.shuffle(bundles)
    prompts_before_sharding = len(bundles)
    if num_prompt_shards < 1:
        raise ValueError("num_prompt_shards must be >= 1")
    if prompt_shard_index < 0 or prompt_shard_index >= num_prompt_shards:
        raise ValueError(
            f"prompt_shard_index={prompt_shard_index} must be in [0, {num_prompt_shards})"
        )
    bundles = bundles[prompt_shard_index::num_prompt_shards]
    if t_prompts > 0:
        bundles = bundles[: min(t_prompts, len(bundles))]
    stats = {
        "common_uid": len(common_uids),
        "prompts_before_sharding": prompts_before_sharding,
        "num_prompt_shards": num_prompt_shards,
        "prompt_shard_index": prompt_shard_index,
        "selected_prompts": len(bundles),
        "skipped_too_short": skipped_too_short,
        "skipped_bad_prompt": skipped_bad_prompt,
    }
    return bundles, stats

def build_sampling_plan(
    bundles: Sequence[PromptBundle],
    required_steps: Sequence[int],
    k_eval: int,
    m_repeats: int,
    rng: np.random.Generator,
    replace: bool,
) -> List[List[Dict[int, np.ndarray]]]:
    """
    Pre-sample indices for every (prompt, repeat, step), so all methods share
    exactly the same sampled responses within each repeat.
    """
    plan: List[List[Dict[int, np.ndarray]]] = []
    for bundle in bundles:
        prompt_plan: List[Dict[int, np.ndarray]] = []
        for _ in range(m_repeats):
            step_plan: Dict[int, np.ndarray] = {}
            for step in required_steps:
                sample = bundle.step_samples[step]
                idx = rng.choice(sample.rewards.size, size=k_eval, replace=replace)
                step_plan[step] = np.asarray(idx, dtype=np.int64)
            prompt_plan.append(step_plan)
        plan.append(prompt_plan)
    return plan

def compute_reinforcepp_batch_mean_baselines(
    bundles: Sequence[PromptBundle],
    sampling_plan: Sequence[Sequence[Mapping[int, np.ndarray]]],
    target_step: int,
) -> List[float]:
    """
    Full-batch mean before the completion-level LOO correction, for repeat m:
      b_m = mean_g(mu_{g,m}),
    where mu_{g,m} is the sampled group mean of prompt g at target step.
    """
    if len(bundles) == 0:
        return []
    m_repeats = len(sampling_plan[0]) if len(sampling_plan[0]) > 0 else 0
    out: List[float] = []
    for m in range(m_repeats):
        mu_list: List[float] = []
        for prompt_idx, bundle in enumerate(bundles):
            idx = sampling_plan[prompt_idx][m][target_step]
            rewards_step = bundle.step_samples[target_step].rewards[idx]
            mu_s, _ = compute_mean_var(rewards_step)
            mu_list.append(mu_s)
        out.append(float(np.mean(mu_list)))
    return out

def run_experiment(cfg: RunConfig) -> Dict[str, Any]:
    degenerate_var_eps = 1e-12
    zero_adv_eps = 1e-12
    rng = np.random.default_rng(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
    os.makedirs(cfg.out_dir, exist_ok=True)
    all_paths = glob.glob(cfg.samples_glob)
    ordered = order_paths_by_step(all_paths)
    if len(ordered) == 0:
        raise ValueError(f"No files matched samples_glob={cfg.samples_glob}")
    step_to_path = {step: path for step, path in ordered}
    required_steps = sorted(cfg.history_steps + [cfg.target_step_abs])
    for step in required_steps:
        if step not in step_to_path:
            raise ValueError(f"Missing parquet for step={step}; check samples_glob and step ids")
    step_to_samples: Dict[int, Dict[str, StepSample]] = {}
    step_load_stats: Dict[int, Dict[str, int]] = {}
    for step in required_steps:
        samples, st = load_step_samples(step_to_path[step], prompt_col=cfg.prompt_col)
        step_to_samples[step] = samples
        step_load_stats[step] = st
    bundles, selection_stats, sampling_plan, reinforcepp_batch_baselines, batch_metadata = prepare_full_batch(
        cfg, step_to_samples, required_steps, rng, choose_prompt_bundles,
        build_sampling_plan, compute_reinforcepp_batch_mean_baselines)
    if len(bundles) == 0:
        raise ValueError("No valid prompts after filtering; check uid alignment and rewards length")
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=cfg.trust_remote_code)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    torch_dtype = torch.float16 if cfg.dtype == "fp16" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_path,
        torch_dtype=torch_dtype,
        trust_remote_code=cfg.trust_remote_code,
    ).to(cfg.device)
    # Optional low-memory switches (default off to keep original behavior).
    if cfg.enable_gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if cfg.disable_kv_cache and hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False
    model.eval()
    tracked_params, grad_scope_info = configure_grad_parameter_scope(model=model, cfg=cfg)
    prompt_rows: List[Dict[str, Any]] = []
    failed_prompts: List[Dict[str, str]] = []
    tokenization_prefix_mismatch_count = 0
    reference_tokenization_prefix_mismatch_count = 0
    cont_truncated_count = 0
    reference_cont_truncated_count = 0
    repeat_total_count = 0
    repeat_grpo_degenerate_count = 0
    repeat_grpo_zero_adv_count = 0
    repeat_kae_zero_adv_count = 0
    prompt_any_grpo_degenerate_count = 0
    prompt_all_grpo_degenerate_count = 0
    prompt_any_grpo_zero_adv_count = 0
    prompt_any_kae_zero_adv_count = 0
    method_names = ["naive", "reinforcepp", "grpo_no_norm", "overt_no_norm"]
    pbar = tqdm(enumerate(bundles), total=len(bundles), desc="prompts", dynamic_ncols=True)
    for prompt_idx, bundle in pbar:
        try:
            prompt_text, prompt_ids = tokenize_prompt(tokenizer, bundle.messages)
            target_full_sample = bundle.step_samples[cfg.target_step_abs]
            reference_rewards_np = target_full_sample.rewards.astype(np.float32)
            reference_responses = target_full_sample.responses
            reference_value = float(np.mean(reference_rewards_np))
            reference_cont_ids_raw, reference_mismatch_n = tokenize_responses_as_continuations(
                tokenizer=tokenizer,
                prompt_text=prompt_text,
                prompt_ids=prompt_ids,
                responses=reference_responses,
            )
            reference_cont_ids, reference_truncated_n = maybe_truncate_cont_ids(
                cont_ids=reference_cont_ids_raw,
                max_cont_tokens=cfg.max_cont_tokens,
            )
            reference_tokenization_prefix_mismatch_count += int(reference_mismatch_n)
            reference_cont_truncated_count += int(reference_truncated_n)
            reference_rewards = torch.tensor(reference_rewards_np, dtype=torch.float32, device=cfg.device)
            reference_adv = reference_rewards - reference_value
            backward_prompt_loss_microbatched(
                model=model,
                tokenizer=tokenizer,
                prompt_ids=prompt_ids,
                cont_ids_list=reference_cont_ids,
                adv=reference_adv,
                micro_batch_size=cfg.logprob_micro_batch,
            )
            reference_grad_sq_norm = grad_sq_norm(tracked_params)
            shared_reference = capture_current_grads(tracked_params, device=cfg.cov_device)
            stats_map = {
                name: RunningTraceReferenceError(
                    tracked_params,
                    reference=shared_reference,
                    device=cfg.cov_device,
                )
                for name in method_names
            }
            model.zero_grad(set_to_none=True)
            weight_sums: List[float] = []
            prompt_grpo_degenerate_count = 0
            prompt_grpo_zero_adv_count = 0
            prompt_kae_zero_adv_count = 0
            prompt_plan = sampling_plan[prompt_idx]
            for repeat_idx in range(cfg.m_repeats):
                step_stats: Dict[int, Tuple[float, float]] = {}
                target_rewards: Optional[np.ndarray] = None
                target_responses: Optional[List[str]] = None
                for step in required_steps:
                    sample = bundle.step_samples[step]
                    idx = prompt_plan[repeat_idx][step]
                    rewards_step = sample.rewards[idx]
                    mu_s, var_s = compute_mean_var(rewards_step)
                    step_stats[step] = (mu_s, var_s)
                    if step == cfg.target_step_abs:
                        target_rewards = rewards_step.astype(np.float32)
                        target_responses = [sample.responses[int(i)] for i in idx]
                if target_rewards is None or target_responses is None:
                    raise RuntimeError("target step sampling failed unexpectedly")
                mu_t, var_t = step_stats[cfg.target_step_abs]
                mu_kae, _, weight_sum, _ = compute_kae_baseline(
                    target_step=cfg.target_step_abs,
                    step_stats=step_stats,
                    linear_alpha=cfg.linear_alpha,
                    ema_min_var=cfg.ema_min_var,
                )
                weight_sums.append(weight_sum)
                cont_ids_raw, mismatch_n = tokenize_responses_as_continuations(
                    tokenizer=tokenizer,
                    prompt_text=prompt_text,
                    prompt_ids=prompt_ids,
                    responses=target_responses,
                )
                cont_ids, truncated_n = maybe_truncate_cont_ids(
                    cont_ids=cont_ids_raw,
                    max_cont_tokens=cfg.max_cont_tokens,
                )
                tokenization_prefix_mismatch_count += int(mismatch_n)
                cont_truncated_count += int(truncated_n)
                rewards_t = torch.tensor(target_rewards, dtype=torch.float32, device=cfg.device)
                reinforcepp_baseline = float(reinforcepp_batch_baselines[repeat_idx])
                k_float = float(cfg.k_eval)
                grpo_factor = (k_float / (k_float - 1.0)) if cfg.k_eval > 1 else 1.0
                eff_n = k_float * float(weight_sum)
                kae_factor = (eff_n / (eff_n - 1.0)) if eff_n > 1.0 + 1e-8 else 1.0
                adv_reinforcepp = reinforcepp_loo_advantages(
                    rewards_t, reinforcepp_baseline, batch_metadata["reinforcepp_batch_completions"])
                adv_grpo_no_norm = grpo_factor * (rewards_t - mu_t)
                adv_kae_no_norm = kae_factor * (rewards_t - mu_kae)
                adv_naive = rewards_t
                repeat_total_count += 1
                is_grpo_degenerate = bool(var_t <= degenerate_var_eps)
                if is_grpo_degenerate:
                    repeat_grpo_degenerate_count += 1
                    prompt_grpo_degenerate_count += 1
                max_abs_grpo_adv = float(torch.max(torch.abs(adv_grpo_no_norm)).detach().item())
                max_abs_kae_adv = float(torch.max(torch.abs(adv_kae_no_norm)).detach().item())
                if max_abs_grpo_adv <= zero_adv_eps:
                    repeat_grpo_zero_adv_count += 1
                    prompt_grpo_zero_adv_count += 1
                if max_abs_kae_adv <= zero_adv_eps:
                    repeat_kae_zero_adv_count += 1
                    prompt_kae_zero_adv_count += 1
                # For the repeat group (K=8), compute log-probabilities once and
                # reuse the same graph across methods; advantages are constants.
                lp = batched_logprob_continuations(model, tokenizer, prompt_ids, cont_ids)
                loss_naive = -(adv_naive * lp).mean()
                loss_reinforcepp = -(adv_reinforcepp * lp).mean()
                loss_grpo_no_norm = -(adv_grpo_no_norm * lp).mean()
                loss_kae_no_norm = -(adv_kae_no_norm * lp).mean()
                model.zero_grad(set_to_none=True)
                loss_naive.backward(retain_graph=True)
                stats_map["naive"].update_from_grads()
                model.zero_grad(set_to_none=True)
                loss_reinforcepp.backward(retain_graph=True)
                stats_map["reinforcepp"].update_from_grads()
                model.zero_grad(set_to_none=True)
                loss_grpo_no_norm.backward(retain_graph=True)
                stats_map["grpo_no_norm"].update_from_grads()
                model.zero_grad(set_to_none=True)
                loss_kae_no_norm.backward()
                stats_map["overt_no_norm"].update_from_grads()
                model.zero_grad(set_to_none=True)
            if prompt_grpo_degenerate_count > 0:
                prompt_any_grpo_degenerate_count += 1
            if prompt_grpo_degenerate_count == cfg.m_repeats:
                prompt_all_grpo_degenerate_count += 1
            if prompt_grpo_zero_adv_count > 0:
                prompt_any_grpo_zero_adv_count += 1
            if prompt_kae_zero_adv_count > 0:
                prompt_any_kae_zero_adv_count += 1
            row = {
                "prompt_index": prompt_idx,
                "uid": bundle.uid,
                "reference_pool_size": int(reference_rewards_np.size),
                "reference_value": float(reference_value),
                "reference_grad_sq_norm": float(reference_grad_sq_norm),
                "trace_ref_error_naive": float(stats_map["naive"].trace_ref_error()),
                "trace_ref_error_reinforcepp": float(stats_map["reinforcepp"].trace_ref_error()),
                "trace_ref_error_grpo_no_norm": float(stats_map["grpo_no_norm"].trace_ref_error()),
                "trace_ref_error_overt_no_norm": float(stats_map["overt_no_norm"].trace_ref_error()),
                "trace_naive": float(stats_map["naive"].trace_ref_error()),
                "trace_reinforcepp": float(stats_map["reinforcepp"].trace_ref_error()),
                "trace_grpo_no_norm": float(stats_map["grpo_no_norm"].trace_ref_error()),
                "trace_overt_no_norm": float(stats_map["overt_no_norm"].trace_ref_error()),
                "mean_weight_sum": float(np.mean(weight_sums)) if weight_sums else float("nan"),
                "repeat_grpo_degenerate_count": int(prompt_grpo_degenerate_count),
                "repeat_grpo_degenerate_rate": float(prompt_grpo_degenerate_count / cfg.m_repeats),
                "repeat_grpo_zero_adv_count": int(prompt_grpo_zero_adv_count),
                "repeat_grpo_zero_adv_rate": float(prompt_grpo_zero_adv_count / cfg.m_repeats),
                "repeat_overt_zero_adv_count": int(prompt_kae_zero_adv_count),
                "repeat_overt_zero_adv_rate": float(prompt_kae_zero_adv_count / cfg.m_repeats),
            }
            prompt_rows.append(row)
            pbar.set_postfix(
                {
                    "naive": f"{row['trace_ref_error_naive']:.2e}",
                    "rfpp": f"{row['trace_ref_error_reinforcepp']:.2e}",
                    "grpo": f"{row['trace_ref_error_grpo_no_norm']:.2e}",
                    "overt": f"{row['trace_ref_error_overt_no_norm']:.2e}",
                }
            )
        except Exception as exc:  # noqa: BLE001
            failed_prompts.append({"uid": bundle.uid, "error": str(exc)})
            if torch.cuda.is_available() and str(exc).lower().find("out of memory") >= 0:
                torch.cuda.empty_cache()
            continue
    if len(prompt_rows) == 0:
        raise RuntimeError("All prompts failed during experiment run")
    detail_df = pd.DataFrame(prompt_rows)
    avg_trace_ref_error = {
        "naive": float(detail_df["trace_ref_error_naive"].mean()),
        "reinforcepp": float(detail_df["trace_ref_error_reinforcepp"].mean()),
        "grpo_no_norm": float(detail_df["trace_ref_error_grpo_no_norm"].mean()),
        "overt_no_norm": float(detail_df["trace_ref_error_overt_no_norm"].mean()),
    }
    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = (
        f"overt_traceerr_step{cfg.target_step_abs}_k{cfg.k_eval}_m{cfg.m_repeats}"
        f"_t{len(detail_df)}_{run_stamp}"
    )
    summary_path = os.path.join(cfg.out_dir, f"{base_name}_summary.json")
    detail_json_path = os.path.join(cfg.out_dir, f"{base_name}_detail.json")
    detail_parquet_path = os.path.join(cfg.out_dir, f"{base_name}_detail.parquet")
    detail_csv_fallback_path = os.path.join(cfg.out_dir, f"{base_name}_detail.csv")
    plot_path = os.path.join(cfg.out_dir, f"{base_name}_boxplot.png")
    summary_payload: Dict[str, Any] = {
        "config": {
            "samples_glob": cfg.samples_glob,
            "target_step_abs": cfg.target_step_abs,
            "history_steps": cfg.history_steps,
            "k_eval": cfg.k_eval,
            "m_repeats": cfg.m_repeats,
            "t_prompts_requested": cfg.t_prompts,
            "num_prompt_shards": int(cfg.num_prompt_shards),
            "prompt_shard_index": int(cfg.prompt_shard_index),
            "model_path": cfg.model_path,
            "prompt_col": cfg.prompt_col,
            "linear_alpha": cfg.linear_alpha,
            "ema_min_var": cfg.ema_min_var,
            "epsilon": cfg.epsilon,
            **batch_metadata,
            "tokenization_policy": TOKENIZATION_POLICY,
            "compare_norm_and_no_norm": False,
            "sampling_with_replacement": bool(cfg.sampling_with_replacement),
            "reference_gradient_center": "shared oracle target-step group mean over the full target-step sample pool",
            "device": cfg.device,
            "cov_device": cfg.cov_device,
            "dtype": cfg.dtype,
            "seed": cfg.seed,
            "trust_remote_code": cfg.trust_remote_code,
            "tokenization_prefix_mismatch_count": int(tokenization_prefix_mismatch_count),
            "reference_tokenization_prefix_mismatch_count": int(reference_tokenization_prefix_mismatch_count),
            "cont_truncated_count": int(cont_truncated_count),
            "reference_cont_truncated_count": int(reference_cont_truncated_count),
            "logprob_micro_batch": int(cfg.logprob_micro_batch),
            "repeat_logprob_mode": "shared_forward_retain_graph",
            "max_cont_tokens": int(cfg.max_cont_tokens),
            "grad_scope_info": grad_scope_info,
            "enable_gradient_checkpointing": bool(cfg.enable_gradient_checkpointing),
            "disable_kv_cache": bool(cfg.disable_kv_cache),
            "degenerate_var_eps": float(degenerate_var_eps),
            "zero_adv_eps": float(zero_adv_eps),
        },
        "required_steps": required_steps,
        "step_load_stats": step_load_stats,
        "selection_stats": selection_stats,
        "num_prompts_succeeded": int(len(detail_df)),
        "num_prompts_failed": int(len(failed_prompts)),
        "avg_trace_ref_error": avg_trace_ref_error,
        "avg_trace_cov": avg_trace_ref_error,
        "repeat_zero_signal_stats": {
            "repeat_total": int(repeat_total_count),
            "repeat_grpo_degenerate_count": int(repeat_grpo_degenerate_count),
            "repeat_grpo_degenerate_rate": (
                float(repeat_grpo_degenerate_count / repeat_total_count) if repeat_total_count > 0 else float("nan")
            ),
            "repeat_grpo_zero_adv_count": int(repeat_grpo_zero_adv_count),
            "repeat_grpo_zero_adv_rate": (
                float(repeat_grpo_zero_adv_count / repeat_total_count) if repeat_total_count > 0 else float("nan")
            ),
            "repeat_overt_zero_adv_count": int(repeat_kae_zero_adv_count),
            "repeat_overt_zero_adv_rate": (
                float(repeat_kae_zero_adv_count / repeat_total_count) if repeat_total_count > 0 else float("nan")
            ),
            "prompt_any_grpo_degenerate_count": int(prompt_any_grpo_degenerate_count),
            "prompt_any_grpo_degenerate_rate": (
                float(prompt_any_grpo_degenerate_count / len(detail_df)) if len(detail_df) > 0 else float("nan")
            ),
            "prompt_all_grpo_degenerate_count": int(prompt_all_grpo_degenerate_count),
            "prompt_all_grpo_degenerate_rate": (
                float(prompt_all_grpo_degenerate_count / len(detail_df)) if len(detail_df) > 0 else float("nan")
            ),
            "prompt_any_grpo_zero_adv_count": int(prompt_any_grpo_zero_adv_count),
            "prompt_any_grpo_zero_adv_rate": (
                float(prompt_any_grpo_zero_adv_count / len(detail_df)) if len(detail_df) > 0 else float("nan")
            ),
            "prompt_any_overt_zero_adv_count": int(prompt_any_kae_zero_adv_count),
            "prompt_any_overt_zero_adv_rate": (
                float(prompt_any_kae_zero_adv_count / len(detail_df)) if len(detail_df) > 0 else float("nan")
            ),
        },
        "output_files": {
            "summary_json": summary_path,
            "detail_json": detail_json_path,
            "detail_parquet": detail_parquet_path,
            "plot_png": plot_path,
        },
    }
    if failed_prompts:
        summary_payload["failed_prompts"] = failed_prompts
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    detail_records = detail_df.to_dict(orient="records")
    with open(detail_json_path, "w", encoding="utf-8") as f:
        json.dump(detail_records, f, indent=2, ensure_ascii=False)
    parquet_ok = True
    try:
        detail_df.to_parquet(detail_parquet_path, index=False)
    except Exception:  # noqa: BLE001
        parquet_ok = False
        detail_df.to_csv(detail_csv_fallback_path, index=False)
        summary_payload["output_files"]["detail_csv_fallback"] = detail_csv_fallback_path
    plot_arrays = [
        detail_df["trace_ref_error_naive"].to_numpy(dtype=float),
        detail_df["trace_ref_error_reinforcepp"].to_numpy(dtype=float),
        detail_df["trace_ref_error_grpo_no_norm"].to_numpy(dtype=float),
        detail_df["trace_ref_error_overt_no_norm"].to_numpy(dtype=float),
    ]
    plot_labels = ["Naive", "REINFORCE++", "GRPO", "KAE"]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.boxplot(plot_arrays, showfliers=True, flierprops={"markersize": 2, "alpha": 0.3})
    means = [float(np.nanmean(arr)) if arr.size > 0 else float("nan") for arr in plot_arrays]
    for i, mean_val in enumerate(means, start=1):
        if np.isfinite(mean_val):
            ax.text(i, mean_val, f"{mean_val:.3g}", color="blue", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(np.arange(1, len(plot_labels) + 1))
    ax.set_xticklabels(plot_labels, rotation=20)
    ax.set_ylabel("oracle-centered trace error over full trainable parameters")
    ax.set_title(
        "Gradient estimation error comparison @ single step "
        f"(step={cfg.target_step_abs}, K={cfg.k_eval}, M={cfg.m_repeats})"
    )
    ax.grid(True, axis="y", alpha=0.35)
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    if not parquet_ok:
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_payload, f, indent=2, ensure_ascii=False)
    return summary_payload

def parse_args() -> RunConfig:
    ap = argparse.ArgumentParser(
        description=(
            "Math-oriented low-memory oracle-centered gradient error comparison for "
            "Naive + REINFORCE++(batch-mean baseline) + GRPO/OVERT (no-norm)."
        )
    )
    ap.add_argument("--samples_glob", type=str, required=True, help="Glob for sample parquet files across steps.")
    ap.add_argument("--target_step_abs", type=int, required=True, help="Target absolute step.")
    ap.add_argument(
        "--history_steps",
        type=int,
        nargs="*",
        default=None,
        help="Explicit history steps; if omitted, use history_len before target.",
    )
    ap.add_argument("--history_len", type=int, default=3, help="History length when --history_steps is not set.")
    ap.add_argument("--K_eval", type=int, default=8, help="Per-repeat sample size K for each step.")
    ap.add_argument("--M", type=int, default=50, help="Number of repeats per prompt for oracle-centered gradient error estimation.")
    ap.add_argument("--T", type=int, default=64, help="Max prompts to evaluate.")
    ap.add_argument("--num_prompt_shards", type=int, default=1)
    ap.add_argument("--prompt_shard_index", type=int, default=0)
    ap.add_argument(
        "--sampling_with_replacement",
        dest="sampling_with_replacement",
        action="store_true",
        default=True,
        help="Sample each Monte Carlo group with replacement from the cached reward/response pool (default: on).",
    )
    ap.add_argument(
        "--sampling_without_replacement",
        dest="sampling_with_replacement",
        action="store_false",
        help="Disable replacement and recover the old without-replacement Monte Carlo sampling.",
    )
    ap.add_argument("--model_path", type=str, required=True, help="HF model/checkpoint path for logprob+backward.")
    ap.add_argument("--prompt_col", type=str, default="prompt", help="Prompt column name in sample parquet.")
    ap.add_argument("--linear_alpha", type=float, default=0.2, help="Linear temporal decay alpha.")
    ap.add_argument("--ema_min_var", type=float, default=1e-2, help="Variance floor.")
    ap.add_argument("--epsilon", type=float, default=1e-6, help="Numerical epsilon in normalization.")
    ap.add_argument("--device", type=str, default="cuda", help="Torch device.")
    ap.add_argument(
        "--cov_device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
        help="Device for oracle-centered gradient-error accumulators (default cpu to reduce GPU memory).",
    )
    ap.add_argument(
        "--logprob_micro_batch",
        type=int,
        default=0,
        help="Reference-gradient continuation micro-batch size; repeats use shared-forward logprob reuse.",
    )
    ap.add_argument(
        "--max_cont_tokens",
        type=int,
        default=0,
        help="Max continuation tokens per sampled response (<=0 means no truncation).",
    )
    ap.add_argument(
        "--grad_scope",
        type=str,
        choices=["all", "last_n", "range"],
        default="last_n",
        help="Parameter scope for gradient variance stats. Use last_n/range to reduce memory.",
    )
    ap.add_argument("--grad_last_n_layers", type=int, default=5, help="MATH default: last five decoder blocks.")
    ap.add_argument("--grad_layer_start", type=int, default=-1, help="Used when grad_scope=range.")
    ap.add_argument("--grad_layer_end", type=int, default=-1, help="Used when grad_scope=range.")
    ap.add_argument("--include_lm_head", dest="include_lm_head", action="store_true", default=True)
    ap.add_argument("--no_include_lm_head", dest="include_lm_head", action="store_false")
    ap.add_argument("--include_model_norm", dest="include_model_norm", action="store_true", default=True)
    ap.add_argument("--no_include_model_norm", dest="include_model_norm", action="store_false")
    ap.add_argument("--include_embed_tokens", dest="include_embed_tokens", action="store_true", default=False)
    ap.add_argument("--no_include_embed_tokens", dest="include_embed_tokens", action="store_false")
    ap.add_argument(
        "--enable_gradient_checkpointing",
        dest="enable_gradient_checkpointing",
        action="store_true",
        default=False,
        help="Enable gradient checkpointing (default off, to match original behavior).",
    )
    ap.add_argument(
        "--disable_kv_cache",
        dest="disable_kv_cache",
        action="store_true",
        default=False,
        help="Disable model.config.use_cache (default off, to match original behavior).",
    )
    ap.add_argument("--dtype", type=str, choices=["fp16", "bf16"], default="bf16", help="Model dtype.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed.")
    ap.add_argument(
        "--out_dir",
        type=str,
        required=True,
        help="Output directory.",
    )
    ap.add_argument("--trust_remote_code", dest="trust_remote_code", action="store_true", default=True)
    ap.add_argument("--no_trust_remote_code", dest="trust_remote_code", action="store_false")
    args = ap.parse_args()
    history_steps = build_history_steps(
        target_step_abs=args.target_step_abs,
        history_steps=args.history_steps,
        history_len=args.history_len,
    )
    return RunConfig(
        samples_glob=args.samples_glob,
        target_step_abs=args.target_step_abs,
        history_steps=history_steps,
        k_eval=args.K_eval,
        m_repeats=args.M,
        t_prompts=args.T,
        model_path=args.model_path,
        prompt_col=args.prompt_col,
        linear_alpha=max(min(float(args.linear_alpha), 1.0), 1e-6),
        ema_min_var=float(args.ema_min_var),
        epsilon=float(args.epsilon),
        device=args.device,
        cov_device=args.cov_device,
        dtype=args.dtype,
        seed=int(args.seed),
        out_dir=args.out_dir,
        trust_remote_code=bool(args.trust_remote_code),
        logprob_micro_batch=int(args.logprob_micro_batch),
        max_cont_tokens=int(args.max_cont_tokens),
        grad_scope=str(args.grad_scope),
        grad_last_n_layers=int(args.grad_last_n_layers),
        grad_layer_start=int(args.grad_layer_start),
        grad_layer_end=int(args.grad_layer_end),
        include_lm_head=bool(args.include_lm_head),
        include_model_norm=bool(args.include_model_norm),
        include_embed_tokens=bool(args.include_embed_tokens),
        enable_gradient_checkpointing=bool(args.enable_gradient_checkpointing),
        disable_kv_cache=bool(args.disable_kv_cache),
        sampling_with_replacement=bool(args.sampling_with_replacement),
        num_prompt_shards=int(args.num_prompt_shards),
        prompt_shard_index=int(args.prompt_shard_index),
    )

def main() -> None:
    cfg = parse_args()
    summary = run_experiment(cfg)
    print(f"[done] saved summary: {summary['output_files']['summary_json']}")
    print(f"[done] avg oracle-centered trace error: {summary['avg_trace_ref_error']}")

if __name__ == "__main__":
    main()
