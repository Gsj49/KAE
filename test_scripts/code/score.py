#!/usr/bin/env python3
"""Score both code benchmarks in an isolated CPU job; fail if isolation is unavailable."""

import argparse
import json
import os
from pathlib import Path
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile

from common import COUNTS, metrics, sha256, validate_frame

HERE = Path(__file__).resolve().parent


def sandbox_command(args, work):
    inner = ["--input", "/work/input", "--output", "/work/output",
             "--workers", str(args.workers), "--timeout", "15"]
    if args.smoke:
        inner.append("--smoke")
    if args.backend == "docker":
        if not args.image or not re.fullmatch(r"(?:[\w./:-]+@)?sha256:[0-9a-f]{64}", args.image):
            raise ValueError("Use an immutable Docker image ID sha256:... or name@sha256:...")
        return ["docker", "run", "--rm", "--init", "--cidfile", str(work.parent / "container.cid"),
                "--pull=never", "--network=none", "--read-only",
                "--cap-drop=ALL", "--security-opt=no-new-privileges", "--user=65534:65534",
                "--pids-limit=512", "--memory=64g", f"--cpus={args.cpus}",
                "--tmpfs", "/tmp:rw,nosuid,nodev,size=2g",
                "--env", "KAE_CODE_SANDBOX=1", "--env", "OMP_NUM_THREADS=1",
                "--env", "OPENBLAS_NUM_THREADS=1", "--env", "HOME=/tmp",
                "--mount", f"type=bind,source={work},target=/work", "--workdir=/work",
                args.image, "python", "-s", "/opt/scorer/worker.py"] + inner
    prefix = Path(sys.prefix).resolve()
    if prefix in (Path("/"), Path("/home"), Path.home()) or not (prefix / "bin/python").is_file():
        raise ValueError("Bubblewrap requires a dedicated Python environment")
    command = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv"]
    for path in ("/usr", "/bin", "/lib", "/lib64"):
        if Path(path).exists():
            command += ["--ro-bind", path, path]
    if not prefix.is_relative_to("/usr"):
        command += ["--ro-bind", str(prefix), str(prefix)]
    command += ["--ro-bind", str(HERE), "/scorer", "--bind", str(work), "/work",
                "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "--chdir", "/work",
                "--setenv", "HOME", "/tmp", "--setenv", "PATH", f"{prefix}/bin:/usr/bin:/bin",
                "--setenv", "KAE_CODE_SANDBOX", "1", "--setenv", "OMP_NUM_THREADS", "1",
                "--setenv", "OPENBLAS_NUM_THREADS", "1",
                str(prefix / "bin/python"), "-s", "/scorer/worker.py"]
    return command + inner


def run_sandbox(command, log, container_id_file=None):
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Scoring interrupted by signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    finally:
        signal.signal(signal.SIGTERM, previous)
        # A Docker daemon can outlive its CLI. Clean up only this invocation's
        # recorded container, including after a Slurm SIGTERM or Ctrl-C.
        if container_id_file is not None and container_id_file.is_file():
            identifier = container_id_file.read_text().strip()
            if re.fullmatch(r"[0-9a-f]{64}", identifier):
                subprocess.run(["docker", "rm", "-f", identifier], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=20, check=False)


def fixtures(root):
    import pandas as pd
    outputs = {}
    for bench in COUNTS:
        test = "def check(candidate):\n    assert candidate(2) == 3\n" if bench == "humaneval_plus" else "assert add_one(2) == 3\n"
        gold = json.dumps(dict(entry_point="add_one", test=test, prompt_prefix=""))
        frame = pd.DataFrame([dict(reward_model={"ground_truth": gold},
                                   responses=["def add_one(x):\n return x+1", "def add_one(x):\n return x-1"])])
        path = root / f"{bench}.parquet"
        frame.to_parquet(path, index=False)
        outputs[bench] = dict(file=path.name, sha256=sha256(path))
    receipt = dict(status="complete", protocol={"n": 2, "mode": "smoke"}, outputs=outputs)
    (root / "generation_complete.json").write_text(json.dumps(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, help="Directory containing generation_complete.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("docker", "bwrap"), required=True)
    parser.add_argument("--image", help="Immutable Docker image ID/digest built from Dockerfile")
    parser.add_argument("--cpus", type=int, default=72)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--smoke", action="store_true", help="Two known answers per benchmark, still inside the sandbox")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.workers <= args.cpus or args.workers > 128:
        parser.error("Use 1 <= workers <= CPUs, at most 128 workers")
    if not args.smoke and args.input is None:
        parser.error("--input is required except for --smoke")
    print(shlex.join(sandbox_command(args, Path("/tmp/kae-code-disposable-work"))))
    if not args.execute:
        return
    if args.backend == "bwrap" and sys.platform != "linux":
        raise RuntimeError("Bubblewrap needs Linux")
    if shutil.which(args.backend) is None:
        raise RuntimeError(f"{args.backend} is unavailable; no host-execution fallback")
    allocated = os.environ.get("SLURM_CPUS_PER_TASK")
    if allocated and args.cpus > int(allocated):
        raise ValueError("Requested CPU budget exceeds this Slurm allocation")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    import pandas as pd
    args.output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="kae-code-") as temporary:
        work = Path(temporary) / "work"
        work.mkdir(mode=0o755)
        source, scored = work / "input", work / "output"
        source.mkdir(mode=0o755)
        scored.mkdir(mode=0o777)
        scored.chmod(0o777)  # disposable directory, for the unprivileged container user
        if args.smoke:
            generation = fixtures(source)
        else:
            generation = json.loads((args.input / "generation_complete.json").read_text())
            if generation.get("status") != "complete" or generation["protocol"]["n"] not in (1, 16):
                raise ValueError("Incomplete or unsupported generation receipt")
            for bench in COUNTS:
                path = args.input / f"{bench}.parquet"
                if sha256(path) != generation["outputs"][bench]["sha256"]:
                    raise ValueError("Generation checksum mismatch")
                validate_frame(pd.read_parquet(path), bench, generation["protocol"]["n"])
                shutil.copyfile(path, source / path.name)
            shutil.copyfile(args.input / "generation_complete.json", source / "generation_complete.json")
        for path in source.iterdir():
            path.chmod(0o644)
        command = sandbox_command(args, work)
        with (args.output / "scoring.log").open("w") as log:
            run_sandbox(command, log, Path(temporary) / "container.cid" if args.backend == "docker" else None)
        result = json.loads((scored / "summary.json").read_text())
        if result.get("status") != "complete" or result["generation"] != generation:
            raise ValueError("Missing or mismatched scoring result")
        for bench in COUNTS:
            path = scored / f"{bench}.parquet"
            frame = pd.read_parquet(path)
            validate_frame(frame, bench, generation["protocol"]["n"], expected_count=1 if args.smoke else None)
            expected = metrics(frame.rewards.tolist())
            if expected != result["metrics"][bench] or sha256(path) != result["outputs"][bench]["sha256"]:
                raise ValueError("Scoring result integrity failure")
            shutil.copyfile(path, args.output / path.name)
        result["sandbox"] = dict(backend=args.backend, image=args.image, cpus=args.cpus)
        result["worker_sha256"] = sha256(HERE / "worker.py")
        result["common_sha256"] = sha256(HERE / "common.py")
        (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("CODE_SCORE_DONE", args.output, flush=True)


if __name__ == "__main__":
    main()
