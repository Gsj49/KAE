"""Default data locations, overrides and previews from outside the checkout."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from test_paper_components import ROOT
from data_paths import data_path, data_root
from train import parse_args as policy_args
from illustrations import parse_args as illustration_args


class DataLayoutTest(unittest.TestCase):
    def setUp(self):
        self.environment = patch.dict(os.environ, {"KAE_DATA_ROOT": str(ROOT / "data")})
        self.environment.start()
        self.addCleanup(self.environment.stop)

    def test_default_training_and_validation(self):
        for dataset, filename in (("dapo", "dapo/train.parquet"), ("math", "math/train_level_ge3.parquet"), ("gsm8k", "gsm8k/train.parquet")):
            args = policy_args(["--dataset", dataset, "--method", "kae", "--output", "results/demo"])
            self.assertEqual(Path(args.train_data), ROOT / "data" / filename)
            self.assertEqual(Path(args.val_data), ROOT / "data" / ("gsm8k/test.parquet" if dataset == "gsm8k" else "openr1/valid.all.parquet"))
        args = policy_args(["--dataset", "math", "--purpose", "estimation", "--method", "kae", "--output", "results/demo"])
        self.assertEqual(Path(args.val_data), ROOT / "data/math/test.parquet")

    def test_explicit_paths_win(self):
        args = policy_args(["--dataset", "math", "--method", "kae", "--output", "results/demo",
                            "--train-data", "custom/train.parquet", "--val-data", "custom/valid.parquet"])
        self.assertEqual(args.train_data, "custom/train.parquet")
        self.assertEqual(args.val_data, "custom/valid.parquet")

    def test_illustration_defaults(self):
        args = illustration_args("one_shot", ["--method", "kae", "--output", "results/demo"])
        self.assertEqual(Path(args.train_data), ROOT / "data/oneshot/question_128copies.parquet")
        self.assertEqual(Path(args.val_data), ROOT / "data/math/test.parquet")
        args = illustration_args("single_stream", ["--dataset", "math", "--method", "kae", "--output", "results/demo"])
        self.assertEqual(Path(args.train_data), ROOT / "data/math/train_level_ge3.parquet")
        self.assertEqual(Path(args.val_data), ROOT / "data/math/test.parquet")

    def test_external_data_root(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {"KAE_DATA_ROOT": directory}):
            self.assertEqual(data_root(), Path(directory).resolve())
            self.assertEqual(data_path("code"), Path(directory).resolve() / "code")

    def test_previews_work_from_another_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            cases = [
                (["train_scripts/train.py", "--dataset", "dapo", "--method", "kae", "--output", "demo"], "data/dapo/train.parquet"),
                (["test_scripts/evaluate_math.py", "--trained-on", "dapo", "--model", "model", "--run-name", "dapo_kae_seed1", "--output", "demo"], "data/openr1/valid.all.parquet"),
                (["experiments/run.py", "--experiment", "value_mse", "--dataset", "math", "--step", "10", "--output", "demo"], "data/rollouts/math"),
                (["experiments/assumption6/generate_pool.py", "--dataset", "gsm8k", "--step", "10", "--checkpoint", "model"], "data/traces/gsm8k_prompt_trace.jsonl"),
            ]
            for argv, expected in cases:
                result = subprocess.run([sys.executable, str(ROOT / argv[0]), *argv[1:]], cwd=directory,
                                        text=True, capture_output=True, check=True)
                self.assertIn(str(ROOT / expected), result.stdout)
            result = subprocess.run([sys.executable, str(ROOT / "test_scripts/code/generate.py"),
                                     "--model", "model", "--method", "kae", "--train-seed", "1", "--output", "demo"],
                                    cwd=directory, text=True, capture_output=True, check=True)
            self.assertEqual(json.loads(result.stdout)["completions"], 8672)
            self.assertFalse((Path(directory) / "demo").exists())

    def test_asset_checker_distinguishes_missing_empty_and_present(self):
        spec = importlib.util.spec_from_file_location("check_data", ROOT / "data/check.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture"
            self.assertEqual(module.inspect(dict(path=path))["status"], "missing")
            path.write_bytes(b"")
            self.assertEqual(module.inspect(dict(path=path))["status"], "empty")
            path.write_bytes(b"fixture")
            self.assertEqual(module.inspect(dict(path=path))["status"], "present")

    def test_asset_paths_do_not_require_manifests(self):
        spec = importlib.util.spec_from_file_location("check_data", ROOT / "data/check.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as directory, patch.object(module, "ROOT", Path(directory)):
            assets = module.assets()
        records = [row for row in assets if row["group"] in ("training", "math-eval")]
        self.assertEqual(len(records), 6)
        for row in records:
            self.assertNotIn("sha256", row)
            self.assertNotIn("expected_bytes", row)
        self.assertEqual(sum(row["group"] == "rollouts" for row in assets), 30)
        self.assertEqual(sum(row["group"] == "code" for row in assets), 2)


if __name__ == "__main__":
    unittest.main()
