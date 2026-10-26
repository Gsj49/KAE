"""Validate preprocessed VERL parquet files without rewriting their identities."""

import argparse
import hashlib
import json
from pathlib import Path


def validate(path, training=True):
    import numpy as np
    import pandas as pd

    frame = pd.read_parquet(path)
    if frame.empty or not {"prompt", "reward_model", "data_source", "extra_info"} <= set(frame.columns):
        raise ValueError(f"Incomplete training/evaluation schema: {path}")
    indices = []
    for extra in frame["extra_info"]:
        if not isinstance(extra, dict) or extra.get("index") is None:
            raise ValueError("Every row needs a persistent extra_info.index")
        indices.append(str(extra["index"]))
    if training and len(set(indices)) != len(indices):
        raise ValueError("Training row indices must be unique, including separate one-shot prompt copies")
    for reward in frame["reward_model"]:
        if not isinstance(reward, dict) or "ground_truth" not in reward:
            raise ValueError("Missing ground truth")
    for prompt in frame["prompt"]:
        if isinstance(prompt, np.ndarray):
            prompt = prompt.tolist()
        if not isinstance(prompt, list) or not prompt:
            raise ValueError("Expected a nonempty chat-format prompt")
        if any(not isinstance(message, dict) or not isinstance(message.get("content"), str)
               or not isinstance(message.get("role"), str) for message in prompt):
            raise ValueError("Every prompt message needs role/content strings")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return dict(rows=len(frame), unique_indices=len(set(indices)), sha256=digest.hexdigest())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--evaluation", action="store_true")
    args = parser.parse_args()
    print(json.dumps(validate(args.path, training=not args.evaluation), indent=2))
