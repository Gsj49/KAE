#!/usr/bin/env python3
"""Copy one preprocessed question B times, keeping copy and pooling IDs separate."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from validate_data import validate
from data_paths import data_path


def canonical(value):
    return json.dumps(value, sort_keys=True, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x.item())


def validate_oneshot(path, copies=128):
    validate(path)
    frame = pd.read_parquet(path)
    if len(frame) != copies:
        raise ValueError(f"Expected exactly {copies} copies in the one-shot batch")
    for column in ("prompt", "reward_model", "data_source"):
        if len({canonical(row) for row in frame[column]}) != 1:
            raise ValueError(f"Copies do not share the same {column}")
    pools = [row.get("pool_id") for row in frame.extra_info]
    if any(not isinstance(x, (str, int)) for x in pools) or len(set(pools)) != 1:
        raise ValueError("Copies must share one explicit extra_info.pool_id")


def prepare(source, output, row, copies=128):
    if copies < 2 or row < 0:
        raise ValueError("Copies must be at least two and row nonnegative")
    if output.exists() or output.with_suffix(".manifest.json").exists():
        raise FileExistsError(output)
    validate(source, training=False)
    frame = pd.read_parquet(source)
    if row >= len(frame):
        raise IndexError(row)
    original = frame.iloc[row].to_dict()
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    pool = f"oneshot:{digest[:16]}:row{row}"
    records = []
    for index in range(copies):
        record = copy.deepcopy(original)
        record["extra_info"] = dict(record["extra_info"], index=f"{pool}:copy{index}", pool_id=pool)
        records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(output, index=False)
    validate_oneshot(output, copies)
    manifest = dict(source=str(source.resolve()), source_sha256=digest, source_row=row, copies=copies,
                    pool_id=pool, output_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=data_path("one_shot", "source"))
    parser.add_argument("--row", type=int, required=True, help="Zero-based row for the question to repeat")
    parser.add_argument("--copies", type=int, default=128)
    parser.add_argument("--output", type=Path, default=data_path("one_shot", "prepared"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.source, args.output, args.row, args.copies), indent=2))


if __name__ == "__main__":
    main()
