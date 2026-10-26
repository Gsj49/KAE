import unittest

import numpy as np

from baseline_estimators import kalman_posterior_baselines, leave_one_out_mean, shrinkage_baselines


class BaselineEstimatorTest(unittest.TestCase):
    def test_leave_one_out(self):
        np.testing.assert_allclose(leave_one_out_mean(np.array([1.0, 0.0, 1.0])), [0.5, 1.0, 0.5])

    def test_kalman_matches_scalar_recursion(self):
        baseline, variance = kalman_posterior_baselines(np.array([1.0, 0.0]), 1e-5, 1e-2, 0.0, 1.0)
        predicted = 1.0 + 1e-5
        gain = predicted / (predicted + 1e-2)
        self.assertAlmostEqual(baseline[0], gain)
        self.assertAlmostEqual(variance[0], (1.0 - gain) * predicted)
        self.assertTrue(np.all(variance > 0))

    def test_shrinkage_is_two_level_loo(self):
        groups = {"a": np.array([0.0, 1.0]), "b": np.array([1.0, 1.0]), "c": np.array([0.0, 0.0])}
        baselines, lambdas = shrinkage_baselines(groups)
        self.assertEqual(set(baselines), set(groups))
        self.assertTrue(all(0.0 <= value <= 1.0 for value in lambdas.values()))
        self.assertEqual(baselines["a"].shape, (2,))


if __name__ == "__main__":
    unittest.main()
