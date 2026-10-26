"""CPU checks for global, completion-level RF++ leave-one-out baselines."""

import sys
from pathlib import Path
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/value_mse"))
from baseline_estimators import global_batch_leave_one_out


class GlobalLOOTest(unittest.TestCase):
    def test_matches_explicit_deletion(self):
        groups = {"a": np.array([0., 1., 1.]), "b": np.array([0., 0.]), "c": np.array([1., 0., 1.])}
        values = np.concatenate(list(groups.values()))
        actual = np.concatenate(list(global_batch_leave_one_out(groups).values()))
        expected = np.array([np.delete(values, i).mean() for i in range(len(values))])
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-15)

    def test_full_batch_not_shard_mean(self):
        groups = {"a": np.zeros(8), "b": np.ones(8)}
        full = global_batch_leave_one_out(groups)
        local = global_batch_leave_one_out({"a": groups["a"]})
        np.testing.assert_allclose(full["a"], 8 / 15)
        np.testing.assert_allclose(local["a"], 0)

    def test_advantage_is_cheap_scalar_correction(self):
        rng = np.random.default_rng(0)
        groups = {str(i): rng.integers(0, 2, size=8).astype(float) for i in range(256)}
        baseline = global_batch_leave_one_out(groups)
        values = np.concatenate(list(groups.values()))
        for key, rewards in groups.items():
            np.testing.assert_allclose(rewards - baseline[key],
                (2048 / 2047) * (rewards - values.mean()), rtol=1e-14, atol=1e-15)

    def test_invalid_data(self):
        for groups in ({"a": np.array([1.])}, {"a": np.array([0., np.nan])}):
            with self.assertRaises(ValueError):
                global_batch_leave_one_out(groups)

    def test_direct_loo_and_scalar_correction_have_same_gradient(self):
        import torch
        rng = np.random.default_rng(4)
        all_rewards = rng.integers(0, 2, 2048).astype(float)
        rewards = torch.tensor(all_rewards[:8], dtype=torch.float64)
        theta = torch.tensor([0.2, -0.1, 0.3], dtype=torch.float64, requires_grad=True)
        features = torch.tensor(rng.normal(size=(8, 3)), dtype=torch.float64)
        lp = features @ theta
        baseline = (all_rewards.sum() - rewards) / 2047
        direct = torch.autograd.grad(-((rewards - baseline) * lp).mean(), theta, retain_graph=True)[0]
        factored = torch.autograd.grad(-((2048/2047) * (rewards - all_rewards.mean()) * lp).mean(), theta)[0]
        torch.testing.assert_close(direct, factored, rtol=1e-12, atol=1e-14)


if __name__ == "__main__":
    unittest.main()
