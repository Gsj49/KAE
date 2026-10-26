"""Stable prompt identifiers and checkpoint state for historical baselines."""

import json
from pathlib import Path

import numpy as np

from verl.trainer.ppo import core_algos


def historical_indices(non_tensor_batch):
    """Return stable prompt IDs for matching observations across updates."""
    extra = non_tensor_batch.get("extra_info")
    if extra is None:
        raise ValueError("KAE/EMA requires a stable extra_info.index for each prompt copy")
    result = []
    for row in extra:
        if not isinstance(row, dict) or row.get("index") is None:
            raise ValueError("Missing stable extra_info.index; prepare the training data first")
        index = row["index"]
        if isinstance(index, np.generic):
            index = index.item()
        if not isinstance(index, (str, int)):
            raise TypeError("extra_info.index must be a string or integer")
        result.append(index)
    # Different copies of a one-shot prompt must have different indices.
    return np.asarray(result, dtype=object)


def pooled_indices(non_tensor_batch):
    """Same-question copies share pool_id, but retain distinct history indices."""
    extra = non_tensor_batch.get("extra_info")
    if extra is None:
        raise ValueError("Pooled one-shot baseline requires extra_info.pool_id")
    result = []
    for row in extra:
        value = row.get("pool_id") if isinstance(row, dict) else None
        if isinstance(value, np.generic):
            value = value.item()
        if not isinstance(value, (str, int)):
            raise ValueError("Missing string/integer extra_info.pool_id")
        result.append(value)
    return np.asarray(result, dtype=object)


def save_history(checkpoint_dir, estimator):
    rows = []
    for key, entry in core_algos._KAE_CACHE.items():
        if isinstance(key, np.generic):
            key = key.item()
        rows.append(dict(key=key, means=entry.means, variances=entry.vars, last_step=entry.last_step))
    path = Path(checkpoint_dir) / "kae_history.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(dict(version=1, estimator=estimator, rows=rows), allow_nan=False))
    temporary.replace(path)


def record_prompt_batch(path, non_tensor_batch, step, repeat_count):
    indices = historical_indices(non_tensor_batch).tolist()
    record = dict(global_step=int(step), sticky_k=int(repeat_count), uid_list=indices,
                  num_prompts=len(indices))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(record) + "\n")


def load_history(checkpoint_dir, estimator):
    path = Path(checkpoint_dir) / "kae_history.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot resume historical baseline without {path}")
    state = json.loads(path.read_text())
    if state["version"] != 1 or state["estimator"] != estimator:
        raise ValueError("Historical baseline checkpoint is incompatible")
    cache = {}
    for row in state["rows"]:
        if row["key"] in cache or len(row["means"]) != len(row["variances"]):
            raise ValueError("Invalid history checkpoint")
        if not np.isfinite(row["means"] + row["variances"]).all():
            raise ValueError("Non-finite history checkpoint")
        entry = core_algos._KAEHistory()
        entry.means, entry.vars, entry.last_step = row["means"], row["variances"], int(row["last_step"])
        cache[row["key"]] = entry
    core_algos._KAE_CACHE.clear()
    core_algos._KAE_CACHE.update(cache)
