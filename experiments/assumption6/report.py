"""Report prompt-level correlation summaries and the signed KAE cross term."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(directory, baseline_table=None):
    if baseline_table is not None:
        from verify_assumption6 import summarize as score_summary

        prompts, _, summary = score_summary(
            pd.read_parquet(directory / "per_completion.parquet"), str(baseline_table),
            bootstrap_replicates=0, bootstrap_seed=0,
        )
    else:
        prompts = pd.read_csv(directory / "per_prompt.csv")
        summary = json.loads((directory / "summary.json").read_text())
    nontrivial = prompts.loc[np.isfinite(prompts["pearson"])]
    correlations = nontrivial["pearson"]
    result = dict(prompts=len(prompts), identifiable_prompts=len(correlations),
                  mean_abs_correlation=float(correlations.abs().mean()),
                  sd_abs_correlation=float(correlations.abs().std(ddof=1)),
                  negative_fraction=float((correlations < 0).mean()),
                  mean_abs_bstar_minus_value=float(nontrivial["optimal_minus_value"].abs().mean()))
    methods = summary.get("baseline_cross_terms_by_method", {})
    for name, value in methods.items():
        if name.startswith("KAE"):
            if summary.get("quadratic_term_definition") != "mean_score_norm_sq_times_baseline_mse":
                raise ValueError(
                    "The score summary does not record the mean-squared-error quadratic term. "
                    "Supply --baseline-table to recompute it from saved scores."
                )
            linear = value["mean_cross_term"]
            quadratic = value["mean_quadratic_value_error_term"]
            result.update(kae_method=name, L=linear, Q=quadratic, L_over_Q=linear / quadratic if quadratic else None,
                          quadratic_term_definition=summary["quadratic_term_definition"],
                          cross_term_prompt_scope="all_scored_prompts")
    return {key: (value if not isinstance(value, float) or np.isfinite(value) else None)
            for key, value in result.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-table", type=Path,
                        help="Recompute cross terms on CPU from saved per-completion scores and baseline moments")
    args = parser.parse_args()
    result = summarize(args.scores, args.baseline_table)
    with args.output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    print(json.dumps(result, indent=2))
