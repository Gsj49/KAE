"""CPU-only checks for the two appendix tables and their bundled inputs."""

import copy
import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'experiments' / name / 'run.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MC = load('mc_error')
COST = load('computing_cost')


class SummaryTablesTest(unittest.TestCase):
    def test_mc_matches_paper(self):
        counts = MC.read_counts(ROOT / 'data/mc_error/reference_counts.csv')
        self.assertEqual(len(counts), 1536)
        expected = [
            ('0.02725', '0.02862', '0.04800'), ('0.01221', '0.00000', '0.03840'),
            ('0.00633', '0.00000', '0.01960'), ('0.03364', '0.03363', '0.04841'),
            ('0.01622', '0.01197', '0.04358'), ('0.01046', '0.00000', '0.03416')]
        for row, reference in zip(MC.summarize(counts), expected):
            self.assertEqual(row['prompts'], 256)
            self.assertEqual(tuple(f'{row[k]:.5f}' for k in
                                   ('mean_mc_se', 'median_mc_se', 'p90_mc_se')), reference)

    def test_mc_rejects_bad_counts_or_coverage(self):
        counts = MC.read_counts(ROOT / 'data/mc_error/reference_counts.csv')
        for rows in (counts[:-1], counts + [counts[0]]):
            with self.assertRaises(ValueError):
                MC.summarize(rows)
        for field, value in [('n', 128), ('successes', -1), ('successes', 101), ('successes', float('nan'))]:
            rows = copy.deepcopy(counts)
            rows[0][field] = value
            with self.assertRaises(ValueError):
                MC.summarize(rows)

    def test_computing_cost_matches_paper(self):
        expected = [
            ('13.31', '88.00', '22.87', '3298', '90.68'),
            ('13.24', '87.78', '22.04', '3298', '90.68'),
            ('12.97', '85.80', '17.70', '3261', '90.69'),
            ('13.84', '91.23', '109.79', '3220', '90.70'),
            ('13.28', '88.09', '38.96', '3274', '90.68'),
            ('12.99', '86.09', '29.16', '3249', '90.68')]
        rows = COST.summarize(ROOT / 'data/computing_cost')
        self.assertEqual([r['method'] for r in rows], list(COST.METHODS))
        for row, reference in zip(rows, expected):
            values = (f"{row['training_loop_hours']:.2f}", f"{row['training_step_seconds']:.2f}",
                      f"{row['advantage_ms']:.2f}", f"{row['tokens_per_second_per_gpu']:.0f}",
                      f"{row['sampled_gpu_memory_gib']:.2f}")
            self.assertEqual(values, reference)

    def test_computing_cost_rejects_incomplete_steps(self):
        original = COST.read_csv

        def incomplete(path):
            rows = original(path)
            return rows[:-1] if Path(path).name == 'steps.csv' else rows

        with patch.object(COST, 'read_csv', incomplete), self.assertRaises(ValueError):
            COST.summarize(ROOT / 'data/computing_cost')
        for value in ('nan', 'inf', '-1'):
            with self.assertRaises(ValueError):
                COST.number(value)

    def test_commands_work_outside_repo_and_preserve_outputs(self):
        for name, prefix in [('mc_error', 'reference_mc'), ('computing_cost', 'computing_cost')]:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                output = Path(temporary) / 'output'
                command = [sys.executable, str(ROOT / 'experiments' / name / 'run.py'),
                           '--output', str(output)]
                subprocess.run(command, cwd=temporary, check=True, capture_output=True)
                for suffix in ('summary.csv', 'table.tex'):
                    self.assertGreater((output / f'{prefix}_{suffix}').stat().st_size, 0)
                before = (output / f'{prefix}_table.tex').read_bytes()
                rerun = subprocess.run(command, cwd=temporary, capture_output=True)
                self.assertNotEqual(rerun.returncode, 0)
                self.assertEqual((output / f'{prefix}_table.tex').read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
