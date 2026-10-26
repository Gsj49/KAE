"""CPU checks for the saved-reward sensitivity entry point."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "experiments/kernel_sensitivity"
spec = importlib.util.spec_from_file_location("kernel_run", HERE / "run.py")
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)
from estimators import (
    ALPHAS, BETAS, bandwidth, ema_weighted_mean_with_current_weight as exponential,
    linear_weighted_mean_with_current_weight as triangular,
    loo_baselines_from_clean_mean as loo,
)


def pools(n=3):
    rng = np.random.default_rng(17)
    return [{f"u{i}": rng.binomial(1, 0.1 + 0.15 * d, 128).astype(np.float32)
             for i in range(n)} for d in range(5)]


class KernelSensitivityTest(unittest.TestCase):
    def test_weights(self):
        values = [0.1, 0.3, 0.5, 0.7, 0.9]
        mean, current = triangular(values, 0.2)
        weights = np.array([0.2, 0.4, 0.6, 0.8, 1.0]) / 3
        self.assertAlmostEqual(mean, np.dot(values, weights), places=6)
        self.assertAlmostEqual(current, 1 / 3, places=6)
        mean, current = exponential(values, 0.2)
        weights = np.array([0.8**4, 0.2 * 0.8**3, 0.2 * 0.8**2, 0.2 * 0.8, 0.2])
        self.assertAlmostEqual(mean, np.dot(values, weights), places=6)
        self.assertAlmostEqual(current, 0.2, places=6)

    def test_loo_matches_explicit_weight_removal(self):
        rewards = np.array([0, 0, 1, 1, 1, 0, 1, 0], dtype=np.float32)
        means = [0.3, 0.2, 0.4, 0.6, float(rewards.mean())]
        for smoother, parameter in [(triangular, a) for a in ALPHAS] + [(exponential, b) for b in BETAS]:
            mean, weight = smoother(means, parameter)
            expected = [(mean * 8 - weight * reward) / (8 - weight) for reward in rewards]
            np.testing.assert_allclose(loo(rewards, mean, weight), expected, rtol=2e-7, atol=1e-7)

    def test_bandwidth_and_current_only_limit(self):
        self.assertEqual(bandwidth("triangular", 0.1, 10), (10.0, 1.0))
        self.assertAlmostEqual(bandwidth("triangular", 0.1, 90)[1], 1 / 9)
        self.assertEqual(bandwidth("exponential", 1.0, 170), (0.0, 0.0))
        for smoother in (triangular, exponential):
            self.assertAlmostEqual(smoother([0.1, 0.2, 0.3, 0.4, 0.8], 1.0)[0], 0.8)
            self.assertEqual(smoother([0.1, 0.2, 0.3, 0.4, 0.8], 1.0)[1], 1.0)

    def test_paired_reproducible_draws_and_grid(self):
        frame, draws = kernel.analyze(pools(), 10, repeats=3)
        again, again_draws = kernel.analyze(pools(), 10, repeats=3)
        pd.testing.assert_frame_equal(frame, again)
        self.assertEqual(len(frame), 3 * 22)
        self.assertTrue(np.isfinite(frame[["mse", "variance"]]).all().all())
        for key in draws:
            np.testing.assert_array_equal(draws[key], again_draws[key])
        for ref, current in zip(draws["reference_indices"], draws["current_indices"]):
            self.assertFalse(np.isin(current, ref).any())
            self.assertTrue(all(len(set(row)) == 8 for row in current))
        pivot = frame.pivot(index="uid", columns="method", values="mse")
        np.testing.assert_array_equal(pivot.GRPO, pivot["triangular:1"])
        np.testing.assert_array_equal(pivot.GRPO, pivot["exponential:1"])

    def test_no_silent_uid_intersection(self):
        data = pools()
        data[0].pop("u0")
        with self.assertRaises(ValueError):
            kernel.analyze(data, 10)

    def test_rfpp_matches_explicit_global_completion_deletion(self):
        data = pools()
        frame, draws = kernel.analyze(data, 10, repeats=3)
        uids = draws["uids"]
        current = np.array([data[-1][uid][indices] for uid, indices in zip(uids, draws["current_indices"])])
        for i, uid in enumerate(uids):
            estimates = []
            for repeat in range(3):
                flat = current[:, repeat, :].reshape(-1).astype(float)
                estimates.extend(np.delete(flat, i * 8 + g).mean() for g in range(8))
            reference = data[-1][uid][draws["reference_indices"][i]].mean()
            expected = np.mean((np.asarray(estimates, dtype=np.float32) - reference) ** 2)
            actual = frame.loc[(frame.uid == uid) & (frame.method == "REINFORCE++"), "mse"].item()
            self.assertAlmostEqual(actual, expected, places=7)

    def test_independent_reference(self):
        oracle = {uid: np.ones(100, dtype=np.float32) for uid in pools()[-1]}
        frame, draws = kernel.analyze(pools(), 10, repeats=2, oracle=oracle)
        self.assertTrue(frame.reference_value.eq(1).all())
        np.testing.assert_array_equal(draws["reference_indices"][0], np.arange(100))

    def test_strict_reader(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pool.parquet"
            frame = pd.DataFrame(dict(extra_info=[{"index": "a"}, {"index": "a"}],
                                      rewards=[np.zeros(128), np.zeros(128)]))
            frame.to_parquet(path)
            with self.assertRaises(ValueError):
                kernel.load_pool(path, 2, 128)
            frame.at[1, "extra_info"] = {"index": "b"}
            frame.at[1, "rewards"] = np.full(128, np.nan)
            frame.to_parquet(path)
            with self.assertRaises(ValueError):
                kernel.load_pool(path, 2, 128)

    def test_full_coverage_synthetic_smoke(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for step, data in zip(range(6, 11), pools(256)):
                pd.DataFrame(dict(extra_info=[{"index": uid} for uid in data],
                                  rewards=list(data.values()))).to_parquet(
                    root / f"math_generation_k10_step_{step}_batchTRACE.parquet")
            cmd = [sys.executable, str(HERE / "run.py"), "--dataset", "math", "--step", "10",
                   "--rollouts", directory, "--repeats", "2", "--output", str(root / "result")]
            result = subprocess.run(cmd, text=True, capture_output=True, check=True)
            self.assertIn("KERNEL_SENSITIVITY_DONE", result.stdout)
            receipt = json.loads((root / "result/complete.json").read_text())
            self.assertEqual(receipt["rows"], 5632)
            summary = json.loads((root / "result/summary.json").read_text())
            self.assertEqual(summary["config"]["repeats"], 2)
            self.assertEqual(len(summary["inputs"]), 5)
            self.assertTrue(summary["config"]["rfpp_leave_one_out"])
            self.assertNotEqual(subprocess.run(cmd, capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
