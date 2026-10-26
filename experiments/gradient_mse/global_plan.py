"""CPU-only, immutable full-batch draw plans for the RF++ LOO recovery.

Reconstruct the recorded 16-shard experiment's draws, then combine all 256
prompts BEFORE computing RF++ baselines. Compute shards never define a batch.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(2 ** 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def legacy_draws(sorted_uids, pool_sizes, *, seed=0, repeats=50, group_size=8, legacy_shards=16):
    """Exactly reproduce choose_prompt_bundles + build_sampling_plan.

    pool_sizes has shape [sorted UID, history steps followed by target step].
    The historical number of shards affects RNG replay only, not RF++'s batch.
    """
    if len(sorted_uids) != len(set(sorted_uids)) or list(sorted_uids) != sorted(sorted_uids):
        raise ValueError("UIDs must be unique and lexically sorted")
    sizes = np.asarray(pool_sizes)
    if sizes.ndim != 2 or sizes.shape[0] != len(sorted_uids) or (sizes <= 0).any():
        raise ValueError("Invalid pool sizes")
    if not 1 <= legacy_shards <= len(sorted_uids):
        raise ValueError("Invalid historical shard count")
    lookup = {uid: i for i, uid in enumerate(sorted_uids)}
    order = list(sorted_uids)
    np.random.default_rng(seed).shuffle(order)
    draws = np.empty((len(order), repeats, sizes.shape[1], group_size), dtype=np.int64)
    for shard in range(legacy_shards):
        rng = np.random.default_rng(seed)
        shuffled = list(sorted_uids)
        rng.shuffle(shuffled)
        assert shuffled == order
        for global_index in range(shard, len(order), legacy_shards):
            uid = order[global_index]
            for repeat in range(repeats):
                for step_index, size in enumerate(sizes[lookup[uid]]):
                    draws[global_index, repeat, step_index] = rng.choice(int(size), group_size, replace=True)
    return order, draws


def batch_totals(uids, draws, target_rewards):
    """One global reward sum and completion count for each MC repeat."""
    count = len(uids) * draws.shape[-1]
    if count < 2:
        raise ValueError("LOO requires at least two completions")
    totals = np.zeros(draws.shape[1], dtype=np.float64)
    for index, uid in enumerate(uids):
        rewards = np.asarray(target_rewards[uid], dtype=np.float64)
        totals += rewards[draws[index, :, -1, :]].sum(axis=1)
    return totals, count


def prepare(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    steps = list(range(args.step - 4, args.step + 1))
    pools, inputs = {}, []
    for step in steps:
        path = args.rollouts / f"{args.dataset}_generation_k10_step_{step}_batchTRACE.parquet"
        frame = pd.read_parquet(path)
        ids = frame.extra_info.map(lambda value: str(value["index"]))
        if len(frame) != 256 or ids.nunique() != 256:
            raise ValueError(f"Expected exactly 256 unique prompts: {path}")
        records = {}
        for uid, (_, row) in zip(ids, frame.iterrows()):
            rewards = np.asarray(row.rewards, dtype=np.float64)
            if (rewards.shape != (128,) or len(row.responses) != 128
                    or not np.isfinite(rewards).all() or ((rewards < 0) | (rewards > 1)).any()
                    or any(not isinstance(x, str) or not x for x in row.responses)
                    or len(row.prompt) == 0):
                raise ValueError(f"Invalid complete pool: {path}, UID={uid}")
            records[uid] = rewards
        pools[step] = records
        inputs.append(dict(step=step, filename=path.name, sha256=sha256(path), bytes=path.stat().st_size))
    uids = sorted(pools[args.step])
    if any(set(pool) != set(uids) for pool in pools.values()):
        raise ValueError("History and target UID sets differ")
    sizes = np.array([[len(pools[step][uid]) for step in steps] for uid in uids])
    order, indices = legacy_draws(uids, sizes, legacy_shards=args.legacy_shards)
    totals, count = batch_totals(order, indices, pools[args.step])
    archive = None
    if args.old_details:
        old = pd.read_parquet(args.old_details)
        if len(old) != 256 or old.uid.astype(str).nunique() != 256:
            raise ValueError("Old details do not cover 256 unique prompts")
        for _, row in old.iterrows():
            global_index = int(row.prompt_index) * args.legacy_shards + int(row.source_shard)
            if order[global_index] != str(row.uid):
                raise ValueError("Reconstructed UID order differs from the archived experiment")
        archive = dict(path=str(args.old_details), sha256=sha256(args.old_details), uid_order_matches=True)
    args.output.mkdir(parents=True)
    path = args.output / "draws.npz"
    np.savez_compressed(path, uids=np.asarray(order), steps=np.asarray(steps), indices=indices,
                        global_reward_sums=totals, global_completion_count=np.asarray(count))
    manifest = dict(schema=1, dataset=args.dataset, step=args.step, prompts=256, group_size=8,
                    repeats=50, seed=0, sampling_with_replacement=True,
                    legacy_sampling_shards=args.legacy_shards,
                    baseline="global_batch_completion_leave_one_out", global_completion_count=count,
                    plan_sha256=sha256(path), inputs=inputs, historical_details=archive,
                    builder_sha256=sha256(Path(__file__)), numpy_version=np.__version__)
    (args.output / "plan.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("GLOBAL_LOO_PLAN_READY", json.dumps(manifest), flush=True)


def read_plan(directory):
    directory = Path(directory)
    meta = json.loads((directory / "plan.json").read_text())
    if sha256(directory / "draws.npz") != meta["plan_sha256"]:
        raise ValueError("Sampling-plan SHA mismatch")
    with np.load(directory / "draws.npz", allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    if arrays["indices"].shape != (256, 50, 5, 8) or len(set(arrays["uids"])) != 256:
        raise ValueError("Incomplete global plan")
    if int(arrays["global_completion_count"]) != 2048:
        raise ValueError("RF++ must use 256 x 8 completions")
    return meta, arrays


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--step", type=int, choices=(10, 90, 170), required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--legacy-shards", type=int, default=16)
    parser.add_argument("--old-details", type=Path)
    prepare(parser.parse_args())
