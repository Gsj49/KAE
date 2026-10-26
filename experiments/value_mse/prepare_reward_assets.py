#!/usr/bin/env python3
"""Make lossless, text-free reward inputs from SHA-verified saved rollout pools.

This changes storage only: no sampling, scoring or estimator calculation occurs.
The compact files cannot replace full rollouts for gradient/score-norm analyses.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
STEPS = (6, 7, 8, 9, 10, 86, 87, 88, 89, 90, 166, 167, 168, 169, 170)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reward_table(path, expected_prompts=256, expected_samples=128):
    table = pq.read_table(path, columns=["extra_info", "rewards"])
    identities = table["extra_info"].to_pylist()
    if any(not isinstance(info, dict) or info.get("index") is None for info in identities):
        raise ValueError(f"Missing explicit prompt identity: {path}")
    indices = [info["index"] for info in identities]
    uids = [str(index) for index in indices]
    rewards = table["rewards"].to_pylist()
    if len(uids) != expected_prompts or len(set(uids)) != expected_prompts:
        raise ValueError(f"Expected {expected_prompts} unique prompts: {path}")
    digest = hashlib.sha256()
    digest.update(b"kae-ordered-uid-float64-rewards-v1\0")
    for uid, values in zip(uids, rewards):
        values = np.asarray(values, dtype="<f8")
        if values.shape != (expected_samples,) or not np.isfinite(values).all():
            raise ValueError(f"Invalid reward count or nonfinite rewards: {path}/{uid}")
        if ((values < 0) | (values > 1)).any():
            raise ValueError(f"Reward outside [0, 1]: {path}/{uid}")
        encoded = uid.encode("utf-8")
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
        digest.update(values.tobytes())
    # Retain the original index type and reward values/order. Strip all text,
    # pandas metadata and nonessential extra_info fields (e.g. question/answer).
    index_type = table.schema.field("extra_info").type.field("index").type
    compact = pa.table({
        "extra_info": pa.array([{"index": index} for index in indices],
                               type=pa.struct([pa.field("index", index_type)])),
        "rewards": pa.array(rewards, type=pa.list_(pa.float64())),
    })
    return compact, digest.hexdigest(), set(uids)


def build(source_root, output_root, source_manifest):
    records = json.loads(source_manifest.read_text())["assets"]
    expected = {(dataset, step) for dataset in ("gsm8k", "math") for step in STEPS}
    observed = [(record["dataset"], record["step"]) for record in records]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Source manifest must contain exactly the 30 required pools")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    prepared, uid_sets = [], {}
    # Validate the complete source set before writing derived data.
    for record in records:
        dataset, step, name = record["dataset"], record["step"], record["filename"]
        if Path(name).name != name or name != f"{dataset}_generation_k10_step_{step}_batchTRACE.parquet":
            raise ValueError("Unexpected source filename")
        source = source_root / dataset / name
        if source.stat().st_size != record["bytes"] or sha256(source) != record["sha256"]:
            raise ValueError(f"Original source size/hash mismatch: {source}")
        compact, semantic_hash, uids = reward_table(source)
        prepared.append((record, compact, semantic_hash))
        uid_sets[dataset, step] = uids
    for dataset in ("gsm8k", "math"):
        for target in (10, 90, 170):
            if any(uid_sets[dataset, step] != uid_sets[dataset, target]
                   for step in range(target - 4, target + 1)):
                raise ValueError(f"UID coverage differs across history: {dataset}/{target}")
    output_root.mkdir(parents=True)
    assets = []
    for record, compact, semantic_hash in prepared:
        relative = Path(record["dataset"]) / record["filename"]
        destination = output_root / relative
        destination.parent.mkdir(exist_ok=True)
        pq.write_table(compact, destination, compression="zstd", compression_level=9)
        restored, restored_hash, _ = reward_table(destination)
        if not compact.equals(restored) or restored_hash != semantic_hash:
            raise ValueError(f"Reward-only round-trip changed data: {destination}")
        assets.append(dict(dataset=record["dataset"], step=record["step"],
            path=relative.as_posix(), bytes=destination.stat().st_size,
            sha256=sha256(destination), source_bytes=record["bytes"],
            source_sha256=record["sha256"], ordered_rewards_sha256=semantic_hash,
            rows=256, rewards_per_prompt=128))
    manifest = dict(format="kae-reward-only-v1", status="complete",
        source_manifest_sha256=sha256(source_manifest),
        description="Lossless projection of the saved retrained Assumption-6 pools; no resampling or rescoring.",
        columns={"extra_info.index": "Original prompt identity; unique within dataset/step.",
                 "rewards": "128 float64 correctness rewards, in the original completion order. Position is the completion index."},
        scope="Value-MSE and reward-only kernel sensitivity. Not gradient-MSE, score-norm correlation, or reward-verifier reproduction.",
        target_steps=[10, 90, 170], history_steps_per_target=4,
        source_total_bytes=sum(row["source_bytes"] for row in assets),
        compact_total_bytes=sum(row["bytes"] for row in assets),
        sampling_note="All 128 rewards are retained. Existing analysis scripts select reference and estimation subsets at run time.",
        assets=assets)
    (output_root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def archive(output_root, archive_path):
    readme = ROOT / "experiments/value_mse/REWARD_ASSETS.md"
    # All paths are explicit. Do not accidentally include the full rollout pools.
    members = [(path, "data/value_mse/" + path.relative_to(output_root).as_posix())
               for path in sorted(output_root.rglob("*")) if path.is_file()]
    members += [(readme, "README.md")]
    members += [(ROOT / name, name) for name in (
        "experiments/value_mse/run.py", "experiments/value_mse/baseline_estimators.py",
        "experiments/value_mse/prepare_reward_assets.py", "LICENSE", "NOTICE")]
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as zipped:
        for source, name in members:
            zipped.write(source, name)
    with zipfile.ZipFile(archive_path) as zipped:
        if zipped.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
    return dict(bytes=archive_path.stat().st_size, sha256=sha256(archive_path),
                files=len(members), path=str(archive_path))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=ROOT / "data/rollouts")
    parser.add_argument("--output-root", type=Path, default=ROOT / "data/value_mse")
    parser.add_argument("--source-manifest", type=Path, default=ROOT / "data/rollout_assets.json")
    parser.add_argument("--zip-output", type=Path)
    args = parser.parse_args()
    if args.zip_output is not None and args.zip_output.exists():
        raise FileExistsError(args.zip_output)
    manifest = build(args.source_root, args.output_root, args.source_manifest)
    result = dict(status="complete", files=len(manifest["assets"]),
                  source_bytes=manifest["source_total_bytes"], reward_bytes=manifest["compact_total_bytes"])
    if args.zip_output is not None:
        result["archive"] = archive(args.output_root, args.zip_output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
