"""Verify a downloaded checkpoint against the pinned public asset manifest."""

import argparse
import json
import os
from pathlib import Path

from global_plan import sha256

DEFAULT_MANIFEST = Path(__file__).resolve().parent / "checkpoint_assets.json"


def verify(checkpoint, dataset, step, manifest_path=DEFAULT_MANIFEST):
    checkpoint = Path(checkpoint).resolve()
    manifest = json.loads(Path(manifest_path).read_text())
    matches = [c for c in manifest["checkpoints"] if c["dataset"] == dataset and c["step"] == step]
    if len(matches) != 1:
        raise ValueError("Expected exactly one dataset/step checkpoint")
    record = matches[0]
    files = []
    for item in record["files"]:
        relative = Path(item["name"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid checkpoint-relative filename")
        path = checkpoint / relative
        before = path.stat()
        if before.st_size != item["bytes"] or sha256(path) != item["sha256"]:
            raise ValueError(f"Checkpoint size/SHA mismatch: {path}")
        after = path.stat()
        if (after.st_size, after.st_mtime_ns) != (before.st_size, before.st_mtime_ns):
            raise ValueError(f"Checkpoint changed during verification: {path}")
        files.append(dict(name=item["name"], bytes=after.st_size,
                          sha256=item["sha256"], mtime_ns=after.st_mtime_ns))
    return dict(status="complete", repo_id=manifest["repo_id"], revision=manifest["revision"],
                asset_manifest_sha256=sha256(manifest_path), checkpoints=[
                    dict(dataset=dataset, step=step, path=str(checkpoint), files=files)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=("gsm8k", "math"), required=True)
    parser.add_argument("--step", type=int, choices=(10, 90, 170), required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    receipt = verify(args.checkpoint, args.dataset, args.step, args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_name(args.output.name + f".tmp.{os.getpid()}")
    with temp.open("x") as handle:
        json.dump(receipt, handle, indent=2, allow_nan=False)
        handle.write("\n")
    temp.rename(args.output)
    print("CHECKPOINT_VERIFIED", args.dataset, args.step, flush=True)


if __name__ == "__main__":
    main()
