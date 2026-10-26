"""Container/namespace-only scoring worker. Call score.py, not this file directly."""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import signal
import sys

from common import COUNTS, gold_record, metrics, sha256, validate_frame

CODE_BLOCK = re.compile(r"```(?:python)?\s*\n?(.*?)```", re.DOTALL)


def extract_code(text):
    blocks = CODE_BLOCK.findall(text)
    return blocks[-1] if blocks else text


def candidate_child(bench, code, gold, connection):
    # The outer sandbox supplies filesystem/network isolation. This fresh child
    # supplies a deadline and separate Python state for each response.
    os.setsid()
    import resource
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))
    os.system = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("blocked"))
    try:
        with open(os.devnull, "w") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        namespace = {"__name__": "__candidate__"}
        entry = gold.get("entry_point", "")
        if bench == "humaneval_plus":
            try:
                exec(code, namespace)
            except Exception:
                namespace = {"__name__": "__candidate__"}
            if entry not in namespace:
                namespace = {"__name__": "__candidate__"}
                exec(gold.get("prompt_prefix", "") + "\n" + code, namespace)
            if entry not in namespace:
                raise ValueError("Entry point not defined")
            exec(gold["test"], namespace)
            namespace["check"](namespace[entry])
        else:
            exec(code, namespace)
            if entry and entry not in namespace:
                raise ValueError("Entry point not defined")
            exec(gold["test"], namespace)
        connection.send(True)
    except BaseException:
        connection.send(False)
    finally:
        connection.close()


def score_one(task):
    bench, row, completion, response, gold, timeout = task
    # This function runs inside a spawned worker, not in a multithreaded parent.
    context = mp.get_context("fork")
    read_end, write_end = context.Pipe(duplex=False)
    child = context.Process(target=candidate_child, args=(bench, extract_code(response), gold, write_end))
    child.start()
    write_end.close()
    value = False
    try:
        child.join(timeout)
        if not child.is_alive() and read_end.poll(0.1):
            try:
                value = bool(read_end.recv())
            except EOFError:
                pass
    finally:
        # Remove any descendants as well as a timed-out candidate.
        try:
            os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        if child.is_alive():
            child.kill()
        child.join()
        read_end.close()
        child.close()
    return bench, row, completion, float(value)


def main():
    if sys.platform != "linux" or os.environ.get("KAE_CODE_SANDBOX") != "1":
        raise RuntimeError("Use score.py with Docker or Bubblewrap; never run candidate code on the host")
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=15)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    import pandas as pd
    receipt = json.loads((args.input / "generation_complete.json").read_text())
    n = receipt["protocol"]["n"]
    frames, rewards, jobs = {}, {}, []
    for bench in COUNTS:
        path = args.input / f"{bench}.parquet"
        if sha256(path) != receipt["outputs"][bench]["sha256"]:
            raise ValueError("Generation checksum mismatch")
        frame = pd.read_parquet(path)
        validate_frame(frame, bench, n, expected_count=1 if args.smoke else None)
        frames[bench], rewards[bench] = frame, [[None] * n for _ in range(len(frame))]
        jobs.append([(bench, i, j, response, gold_record(row, bench), args.timeout)
                     for i, (_, row) in enumerate(frame.iterrows()) for j, response in enumerate(row.responses)])
    # Alternate datasets so the same CPU allocation scores both concurrently.
    tasks = [group[i] for i in range(max(map(len, jobs))) for group in jobs if i < len(group)]
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=mp.get_context("spawn")) as pool:
        futures = [pool.submit(score_one, task) for task in tasks]
        for done, future in enumerate(as_completed(futures), 1):
            bench, row, completion, score = future.result()
            rewards[bench][row][completion] = score
            if done % 128 == 0 or done == len(tasks):
                print("CODE_SCORE_PROGRESS", done, len(tasks), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    summary, outputs = {}, {}
    for bench, frame in frames.items():
        summary[bench] = metrics(rewards[bench])
        if args.smoke and rewards[bench] != [[1.0, 0.0]]:
            raise ValueError(f"Sandbox fixture failed: {bench} {rewards[bench]}")
        frame["rewards"] = rewards[bench]
        path = args.output / f"{bench}.parquet"
        frame.to_parquet(path, index=False)
        outputs[bench] = dict(file=path.name, sha256=sha256(path))
    summary["Avg"] = {key: sum(summary[b][key] for b in COUNTS) / len(COUNTS)
                      for key in summary[next(iter(COUNTS))]}
    result = dict(status="complete", generation=receipt, metrics=summary, outputs=outputs,
                  workers=args.workers, timeout=args.timeout, smoke=args.smoke)
    (args.output / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print("CODE_SCORE_DONE", flush=True)


if __name__ == "__main__":
    main()
