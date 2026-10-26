#!/usr/bin/env python3
"""Aggregate completed code summaries across three training runs per method."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from common import COUNTS, metrics, sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("summaries", nargs="+", type=Path)
    parser.add_argument("--seeds", type=int, nargs=3, default=[1, 42, 420])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(set(args.seeds)) != 3:
        parser.error("Expected three distinct training seeds")
    if args.output.exists():
        raise FileExistsError(args.output)
    rows, sources, protocols, identities = [], [], set(), set()
    for path in args.summaries:
        record = json.loads(path.read_text())
        if record.get("status") != "complete" or record.get("smoke"):
            raise ValueError("Incomplete/smoke output is not a scientific result")
        generation = record["generation"]
        identity = (generation["method"], generation["train_seed"])
        if identity in identities:
            raise ValueError(f"Duplicate run: {identity}")
        identities.add(identity)
        protocols.add(json.dumps(dict(protocol=generation["protocol"], data=generation["input_sha256"],
                                      worker=record["worker_sha256"], common=record["common_sha256"],
                                      timeout=record["timeout"]), sort_keys=True))
        measured = {}
        for bench in COUNTS:
            output = path.parent / f"{bench}.parquet"
            if sha256(output) != record["outputs"][bench]["sha256"]:
                raise ValueError("Score output checksum mismatch")
            frame = pd.read_parquet(output)
            from common import validate_frame
            validate_frame(frame, bench, generation["protocol"]["n"])
            measured[bench] = metrics(frame.rewards.tolist())
            if measured[bench] != record["metrics"][bench]:
                raise ValueError("Metric mismatch")
        measured["Avg"] = {key: np.mean([measured[b][key] for b in COUNTS]) for key in measured[next(iter(COUNTS))]}
        for bench, values in measured.items():
            for metric, value in values.items():
                rows.append(dict(method=identity[0], train_seed=identity[1], benchmark=bench, metric=metric, value=value))
        sources.append(dict(summary=str(path.resolve()), sha256=sha256(path)))
    if len(protocols) != 1:
        raise ValueError("Do not mix greedy and K16 or different data/scoring protocols")
    detail = pd.DataFrame(rows)
    for method, part in detail.groupby("method"):
        if set(part.train_seed) != set(args.seeds):
            raise ValueError(f"Incomplete three-run coverage: {method}")
    aggregate = detail.groupby(["method", "benchmark", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    aggregate["se"] = aggregate["std"] / np.sqrt(aggregate["count"])
    args.output.mkdir(parents=True)
    detail.to_csv(args.output / "runs.csv", index=False)
    aggregate.to_csv(args.output / "mean_sd_se.csv", index=False)
    (args.output / "manifest.json").write_text(json.dumps(dict(sources=sources, protocol=json.loads(protocols.pop()),
        outputs={name: sha256(args.output / name) for name in ("runs.csv", "mean_sd_se.csv")}), indent=2) + "\n")
    print(aggregate.to_string(index=False))


if __name__ == "__main__":
    main()
