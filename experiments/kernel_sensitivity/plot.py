#!/usr/bin/env python3
"""Plot completed kernel-sensitivity summaries; never runs an experiment."""

import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summaries", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory for PDF/PNG and source hashes")
    args = parser.parse_args()
    summaries = [json.loads(path.read_text()) for path in args.summaries]
    if len({s["dataset"] for s in summaries}) != 1 or len({s["step"] for s in summaries}) != len(summaries):
        raise ValueError("Use one dataset and one summary per step")
    configs = [{k: v for k, v in s["config"].items() if k != "history_steps"} for s in summaries]
    if any(config != configs[0] for config in configs):
        raise ValueError("Cannot combine different experimental configurations")
    for path in args.summaries:
        receipt = json.loads((path.parent / "complete.json").read_text())
        if receipt["status"] != "complete" or receipt["outputs"]["summary.json"] != hashlib.sha256(path.read_bytes()).hexdigest():
            raise ValueError(f"Missing or mismatched completion receipt: {path}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(args.output)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summaries.sort(key=lambda s: s["step"])
    fig, axes = plt.subplots(len(summaries), 2, figsize=(9, 2.7 * len(summaries)), squeeze=False, sharey="row")
    for row, summary in enumerate(summaries):
        for col, family in enumerate(("triangular", "exponential")):
            ax = axes[row, col]
            entries = sorted((item for item in summary["method_summary"] if item["family"] == family),
                             key=lambda item: item["bandwidth_H"])
            ax.plot([item["bandwidth_H"] for item in entries], [item["mean_mse"] for item in entries],
                    marker="o", markersize=3, color="#246b8e", label="KAE")
            for name, color, style in (("GRPO", "#b76b18", "--"), ("REINFORCE++", "#666666", ":")):
                entry = next(item for item in summary["method_summary"] if item["method"] == name)
                ax.axhline(entry["mean_mse"], color=color, linestyle=style, label=name)
            ax.set(title=f"Step {summary['step']}: {family}", xlabel=r"Bandwidth $H=i h_i$")
            if col == 0:
                ax.set_ylabel("Mean prompt MSE")
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", alpha=0.2)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    args.output.mkdir(parents=True, exist_ok=True)
    for extension in ("pdf", "png"):
        fig.savefig(args.output / f"kernel_sensitivity.{extension}", dpi=220, bbox_inches="tight")
    plt.close(fig)
    (args.output / "sources.json").write_text(json.dumps([
        dict(summary=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        for path in args.summaries], indent=2) + "\n")


if __name__ == "__main__":
    main()
