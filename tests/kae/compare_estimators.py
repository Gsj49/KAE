#!/usr/bin/env python3
"""Compare fixed-bandwidth estimator bodies against a supplied source file."""

import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from test_paper_components import estimator_namespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-file", type=Path, required=True)
    args = parser.parse_args()
    names = ("compute_grpo_kae_tri_outcome_advantage", "compute_grpo_kae_exp_outcome_advantage",
             "compute_grpo_kalman_outcome_advantage", "compute_grpo_shrinkage_outcome_advantage")
    cases = 0
    maximum = 0.0
    for name in names:
        for normalize in (False, True):
            for trial in range(5):
                old = estimator_namespace(args.reference_file)
                new = estimator_namespace()
                width = (2.5, 5.0, 10.0, 5.0, 10.0)[trial]
                cfg = SimpleNamespace(kae_ih=width, ema_beta=0.2, ema_leave_one_out=True,
                                      ema_min_var=0.01, kae_std_mode="group" if trial % 2 else "weighted",
                                      kae_std_fallback=0.5)
                reference_cfg = SimpleNamespace(**vars(cfg), linear_alpha=1.0 / width)
                generator = np.random.default_rng(trial)
                for position in range(1, 20):
                    step = position if position < 12 else position + 3
                    rewards = torch.tensor(generator.integers(0, 2, (12, 3)), dtype=torch.float32)
                    mask = torch.ones_like(rewards)
                    mask[:, -1] = 0
                    kwargs = dict(index=np.repeat(["a", "b", "c"], 4),
                                  norm_adv_by_std_in_grpo=normalize, global_steps=step)
                    original = old[name](rewards.clone(), mask, config=reference_cfg, **kwargs)
                    candidate = new[name](rewards.clone(), mask, config=cfg, **kwargs)
                    for expected, actual in zip(original, candidate):
                        if not torch.isfinite(actual).all():
                            raise AssertionError(f"Nonfinite output in {name}")
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        maximum = max(maximum, float((actual - expected).abs().max()))
                    cases += 1
    print(json.dumps(dict(reference_sha256=hashlib.sha256(args.reference_file.read_bytes()).hexdigest(),
                          cases=cases, max_absolute_difference=maximum, status="passed"), indent=2))


if __name__ == "__main__":
    main()
