"""Code-benchmark input checks and metrics (does not execute candidate code)."""

import hashlib
import json
from pathlib import Path

import numpy as np

COUNTS = {"humaneval_plus": 164, "mbpp_plus": 378}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def gold_record(row, benchmark):
    gold = row["reward_model"]["ground_truth"]
    gold = json.loads(gold) if isinstance(gold, str) else gold
    if not isinstance(gold, dict) or not isinstance(gold.get("test"), str) or not gold["test"].strip():
        raise ValueError("Missing code test harness; this is an input error, not reward zero")
    if benchmark == "humaneval_plus" and not gold.get("entry_point"):
        raise ValueError("HumanEval+ needs an entry_point")
    for key in ("entry_point", "prompt_prefix"):
        if key in gold and not isinstance(gold[key], str):
            raise ValueError(f"Invalid {key}")
    return gold


def validate_frame(frame, benchmark, n=None, expected_count=None):
    expected = COUNTS[benchmark] if expected_count is None else expected_count
    if len(frame) != expected or "reward_model" not in frame:
        raise ValueError(f"Expected {expected} {benchmark} rows with test harnesses")
    for _, row in frame.iterrows():
        gold_record(row, benchmark)
        if n is not None:
            responses = row.get("responses")
            if not isinstance(responses, (list, np.ndarray)) or len(responses) != n:
                raise ValueError(f"Expected {n} responses per problem")
            if not all(isinstance(x, str) for x in responses):
                raise ValueError("Invalid response type")


def metrics(rewards):
    values = np.asarray(rewards, dtype=float)
    if values.ndim != 2 or not values.size or not np.isfinite(values).all() or not np.isin(values, [0, 1]).all():
        raise ValueError("Expected a complete binary problem-by-completion matrix")
    n = values.shape[1]
    # Exactly K draws per problem: pass@K is the indicator of any passing draw.
    return {f"pass@{n}": float(values.max(axis=1).mean()), f"avg@{n}": float(values.mean())}
