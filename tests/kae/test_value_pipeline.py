"""Synthetic-data CPU smoke for the complete saved-pool value-MSE entry point."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


class ValuePipelineTest(unittest.TestCase):
    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "matplotlib is needed for the value-MSE plot")
    def test_complete_value_run_and_overwrite_guard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pools = root / "pools"
            pools.mkdir()
            rng = np.random.default_rng(24)
            for step in range(6, 11):
                rows = [dict(extra_info={"index": str(i)}, responses=["answer"] * 128,
                             rewards=rng.integers(0, 2, 128).astype(float).tolist()) for i in range(256)]
                pd.DataFrame(rows).to_parquet(pools / f"gsm8k_generation_k10_step_{step}_batchTRACE.parquet")
            output = root / "analysis"
            command = [sys.executable, str(ROOT / "experiments/run.py"), "--experiment", "value_mse",
                       "--dataset", "gsm8k", "--step", "10", "--rollouts", str(pools),
                       "--output", str(output), "--execute"]
            env = dict(os.environ, MPLCONFIGDIR=str(root / "matplotlib"), MPLBACKEND="Agg")
            result = subprocess.run(command, capture_output=True, text=True, env=env, timeout=180)
            self.assertEqual(result.returncode, 0, result.stderr[-6000:])
            receipt = json.loads((output / "analysis_complete.json").read_text())
            self.assertEqual(receipt["status"], "complete")
            self.assertEqual(receipt["prompts"], 256)
            self.assertEqual(receipt["rows"], 1536)
            before = (output / "analysis_complete.json").read_bytes()
            rerun = subprocess.run(command, capture_output=True, text=True, env=env, timeout=30)
            self.assertNotEqual(rerun.returncode, 0)
            self.assertIn("Refusing to overwrite", rerun.stderr)
            self.assertEqual((output / "analysis_complete.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
