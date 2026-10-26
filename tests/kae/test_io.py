"""CPU checks for identities, history state, model completeness and scoring I/O."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd

from test_paper_components import ROOT, estimator_namespace


def module_at(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class InputOutputTest(unittest.TestCase):
    def test_history_round_trip_and_missing_id(self):
        core = SimpleNamespace(**estimator_namespace())
        tree = ast.parse((ROOT / "verl/trainer/ppo/kae_state.py").read_text())
        namespace = dict(core_algos=core, json=json, Path=Path, np=np)
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "kae_state.py", "exec"), namespace)
        indices = namespace["historical_indices"]({"extra_info": [{"index": np.int64(5)}, {"index": "p6"}]})
        self.assertEqual(indices.tolist(), [5, "p6"])
        with self.assertRaises(ValueError):
            namespace["historical_indices"]({})
        entry = core._KAEHistory()
        entry.means, entry.vars, entry.last_step = [0.25, 0.5], [0.1, 0.2], 7
        core._KAE_CACHE[5] = entry
        with tempfile.TemporaryDirectory() as temporary:
            namespace["save_history"](temporary, "grpo_kae_tri")
            core._KAE_CACHE.clear()
            namespace["load_history"](temporary, "grpo_kae_tri")
            restored = core._KAE_CACHE[5]
            self.assertEqual((restored.means, restored.vars, restored.last_step), ([0.25, 0.5], [0.1, 0.2], 7))
            with self.assertRaises(ValueError):
                namespace["load_history"](temporary, "grpo_kae_exp")

    def test_model_requires_all_indexed_shards(self):
        evaluate = module_at("evaluate", "test_scripts/evaluate_math.py")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "config.json").write_text("{}")
            (root / "model.safetensors.index.json").write_text(json.dumps({
                "weight_map": {"a": "model-1.safetensors", "b": "model-2.safetensors"}}))
            (root / "model-1.safetensors").write_bytes(b"test fixture")
            with self.assertRaises(ValueError):
                evaluate.validate_model(root)
            (root / "model-2.safetensors").write_bytes(b"test fixture")
            self.assertEqual(len(evaluate.validate_model(root)), 2)

    def test_math_metric_counts_and_macro_average(self):
        summarize = module_at("summarize", "test_scripts/summarize_math.py")
        rows = []
        for source, count in summarize.EXPECTED_COUNTS.items():
            copies = summarize.EXPECTED_MULTIPLICITY[source]
            for index in range(count):
                rows.append(dict(data_source=source, extra_info={"index": index // copies},
                                 responses=["answer"], rewards=[0.5]))
        frame = pd.DataFrame(rows)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "generation_dapo_kae_seed42_s500_temperature0.6_evalseed0.parquet"
            frame.to_parquet(path)
            row = summarize.summarize_file(path, strict=True)
            self.assertEqual(row["avg"], 0.5)
            self.assertEqual(row["train_seed"], 42)
            with self.assertRaises(ValueError):
                summarize.validate_validall(frame.iloc[:-1], path, strict=True)
            with self.assertRaises(ValueError):
                summarize.reward_scalar([float("nan")])

    def test_generation_score_call_contract(self):
        tree = ast.parse((ROOT / "test_scripts/generate_math.py").read_text())
        calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                 and isinstance(node.func, ast.Name) and node.func.id == "compute_score"]
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0].args), 3)  # data_source, response, ground truth

    def test_sampler_module_import(self):
        # Import the complete module, including annotations and dependencies.
        sys.path.insert(0, str(ROOT))
        from verl.experimental.dataset.sampler import FixedRepeatBatchSampler
        self.assertTrue(callable(FixedRepeatBatchSampler))


if __name__ == "__main__":
    unittest.main()
