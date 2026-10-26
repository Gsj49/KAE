"""CPU checks of baseline moments, cross terms and paper-table summaries."""

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = ROOT / "experiments/assumption6"
sys.path.insert(0, str(SCRIPTS))
from build_baseline_table import baseline_moments
from verify_assumption6 import QUADRATIC_TERM_DEFINITION, _load_baselines, summarize

spec = importlib.util.spec_from_file_location("assumption6_report", SCRIPTS / "report.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class Assumption6MomentsTest(unittest.TestCase):
    def test_mse_includes_variance_with_population_divisor(self):
        moments = baseline_moments([0.0, 1.0, 0.0, 1.0], 0.25)
        self.assertEqual(moments["baseline_samples"], 4)
        self.assertAlmostEqual(moments["baseline_value"], 0.5)
        self.assertAlmostEqual(moments["baseline_variance"], 0.25)
        self.assertAlmostEqual(moments["baseline_mse"], 0.3125)
        self.assertAlmostEqual(moments["baseline_mse"],
            moments["baseline_variance"] + (moments["baseline_value"] - 0.25) ** 2)

    def test_zero_mean_error_can_have_positive_quadratic_term(self):
        moments = baseline_moments([0.0, 1.0], 0.5)
        self.assertEqual(moments["baseline_value"], moments["oracle_value"])
        self.assertEqual(moments["baseline_mse"], 0.25)

    def test_invalid_samples_rejected(self):
        for values, reference in [([], 0), ([np.nan], 0), ([[1]], 0), ([1], np.inf)]:
            with self.assertRaises(ValueError):
                baseline_moments(values, reference)

    @staticmethod
    def scores():
        rows = []
        for uid, rewards, scores in [
            ("x", [0, 0, 1, 1], [4, 2, 1, 1]),
            ("y", [1, 1, 1, 1], [2, 3, 4, 5]),
        ]:
            rows.extend(dict(uid=uid, completion_id=i, reward=z, score_norm_sq=q)
                        for i, (z, q) in enumerate(zip(rewards, scores)))
        return pd.DataFrame(rows)

    @staticmethod
    def baselines():
        return pd.DataFrame([
            dict(uid="x", method="KAE(alpha=0.2)", **baseline_moments([0.0, 0.5], 0.5)),
            dict(uid="y", method="KAE(alpha=0.2)", **baseline_moments([0.5, 1.0], 1.0)),
        ])

    def test_cross_term_sign_and_ratio_of_means(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baselines.parquet"
            self.baselines().to_parquet(path, index=False)
            prompts, terms, summary = summarize(self.scores(), str(path), 0, 0)
        x = terms.set_index("uid").loc["x"]
        self.assertAlmostEqual(x.cov_reward_score_norm_sq, -0.5)
        self.assertAlmostEqual(x.baseline_error, -0.25)
        self.assertAlmostEqual(x.ae_cross_term, -0.25)
        self.assertAlmostEqual(x.quadratic_value_error_term, 0.25)
        stats = summary["baseline_cross_terms_by_method"]["KAE(alpha=0.2)"]
        self.assertEqual(stats["num_prompts"], 2)
        self.assertAlmostEqual(stats["mean_cross_term"], -0.125)
        self.assertAlmostEqual(stats["mean_quadratic_value_error_term"], 0.34375)
        self.assertEqual(summary["quadratic_term_definition"], QUADRATIC_TERM_DEFINITION)
        # Directly average over baseline draws and independent score/reward samples.
        z = np.array([0, 0, 1, 1])
        q = np.array([4, 2, 1, 1])
        direct = np.mean([np.mean(q * ((z - b)**2 - (z - 0.5)**2)) for b in (0, 0.5)])
        self.assertAlmostEqual(direct, x.ae_cross_term + x.quadratic_value_error_term)

    def test_mean_only_baseline_table_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baselines.csv"
            self.baselines().drop(columns="baseline_mse").to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "requires baseline_mse"):
                _load_baselines(str(path))

    def test_long_format_preserves_mse(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baselines.csv"
            self.baselines().rename(columns={"method": "which", "baseline_value": "mean_est",
                "oracle_value": "gt_true", "baseline_mse": "mse"}).to_csv(path, index=False)
            actual = _load_baselines(str(path))
        np.testing.assert_array_equal(actual.baseline_mse, self.baselines().baseline_mse)

    def test_inconsistent_mse_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baselines.csv"
            table = self.baselines()
            table["baseline_mse"] = 0.0
            table.to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "squared mean error"):
                _load_baselines(str(path))

    def test_missing_prompt_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "baselines.csv"
            self.baselines().iloc[:1].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "missing scored prompts"):
                summarize(self.scores(), str(path), 0, 0)

    def test_report_uses_nontrivial_prompts_but_all_prompt_cross_terms(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            path = root / "baselines.parquet"
            self.baselines().to_parquet(path, index=False)
            self.scores().to_parquet(root / "per_completion.parquet", index=False)
            prompts, _, summary = summarize(self.scores(), str(path), 0, 0)
            prompts.to_csv(root / "per_prompt.csv", index=False)
            (root / "summary.json").write_text(json.dumps(summary))
            result = report.summarize(root)
            recomputed = report.summarize(root, path)
        self.assertEqual(result, recomputed)
        self.assertEqual(result["prompts"], 2)
        self.assertEqual(result["identifiable_prompts"], 1)
        self.assertAlmostEqual(result["mean_abs_bstar_minus_value"], 0.25)
        self.assertAlmostEqual(result["L_over_Q"], -0.125 / 0.34375)

    def test_report_rejects_summary_with_old_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            prompts, _, _ = summarize(self.scores(), None, 0, 0)
            prompts.to_csv(root / "per_prompt.csv", index=False)
            (root / "summary.json").write_text(json.dumps({"baseline_cross_terms_by_method": {
                "KAE(alpha=0.2)": {"mean_cross_term": 1, "mean_quadratic_value_error_term": 2}}}))
            with self.assertRaisesRegex(ValueError, "mean-squared-error quadratic term"):
                report.summarize(root)


if __name__ == "__main__":
    unittest.main()
