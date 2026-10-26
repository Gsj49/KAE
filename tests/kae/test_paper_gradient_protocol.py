"""Synthetic CPU checks of the paper scope, global LOO, and strict aggregation."""

import ast
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from test_gradient_global_plan import historical_functions

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/gradient_mse"))
from paper_protocol import math_parameter_scope, prepare_full_batch, reinforcepp_loo_advantages


class PaperGradientProtocolTest(unittest.TestCase):
    def test_full_batch_loo_invariant_to_compute_shards(self):
        rng = np.random.default_rng(57)
        steps = [6, 7, 8, 9, 10]
        pools = {step: {str(i): SimpleNamespace(messages=[{"role": "user", "content": str(i)}],
                    responses=["response"] * 128, rewards=rng.integers(0, 2, 128).astype(float))
                       for i in range(12)} for step in steps}
        for filename in ("gsm8k.py", "math_mse.py"):
            funcs = historical_functions(ROOT / "experiments/gradient_mse" / filename)

            def prepare(shards, shard):
                cfg = SimpleNamespace(seed=0, target_step_abs=10, k_eval=8, t_prompts=12,
                                      m_repeats=3, sampling_with_replacement=True,
                                      num_prompt_shards=shards, prompt_shard_index=shard)
                return prepare_full_batch(cfg, pools, steps, np.random.default_rng(0),
                    funcs["choose_prompt_bundles"], funcs["build_sampling_plan"],
                    funcs["compute_reinforcepp_batch_mean_baselines"])

            full, stats, plan, means, metadata = prepare(1, 0)
            self.assertEqual(metadata["reinforcepp_batch_completions"], 96)
            seen = set()
            for shard in range(3):
                local, selection, local_plan, local_means, local_meta = prepare(3, shard)
                self.assertEqual(local_meta, metadata)
                self.assertEqual(local_means, means)
                self.assertEqual(selection["prompts_before_sharding"], 12)
                self.assertEqual(selection["selected_prompts"], 4)
                for i, bundle in enumerate(local):
                    self.assertNotIn(bundle.uid, seen)
                    seen.add(bundle.uid)
                    index = shard + 3 * i
                    self.assertEqual(full[index].uid, bundle.uid)
                    for repeat in range(3):
                        for step in steps:
                            np.testing.assert_array_equal(local_plan[i][repeat][step], plan[index][repeat][step])
                        values = np.concatenate([b.step_samples[10].rewards[plan[j][repeat][10]]
                                                 for j, b in enumerate(full)])
                        current = values[index * 8:(index + 1) * 8]
                        actual = reinforcepp_loo_advantages(current, means[repeat], 96)
                        expected = current - (values.sum() - current) / 95
                        np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=1e-15)
            self.assertEqual(len(seen), 12)

    def test_last_five_blocks_norm_and_head(self):
        class Toy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torch.nn.Module()
                self.model.layers = torch.nn.ModuleList([torch.nn.Linear(3, 3) for _ in range(7)])
                self.model.norm = torch.nn.LayerNorm(3)
                self.model.embed_tokens = torch.nn.Embedding(5, 3)
                self.lm_head = torch.nn.Linear(3, 5)
        filename = ROOT / "experiments/gradient_mse/math_mse.py"
        tree = ast.parse(filename.read_text())
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and
                 node.name in ("configure_grad_parameter_scope", "parse_layer_id_from_param_name")]
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes
        namespace = {"LAYER_PAT": re.compile(r"(?:^|\.)layers\.(\d+)\.")}
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(filename), "exec"), namespace)
        model = Toy()
        tracked, info = namespace["configure_grad_parameter_scope"](model, math_parameter_scope())
        self.assertEqual(info["selected_layer_ids"], [2,3,4,5,6])
        actual = {name for name, p in model.named_parameters() if p.requires_grad}
        self.assertEqual(actual, {f"model.layers.{i}.{p}" for i in range(2,7) for p in ("weight","bias")}
                         | {"model.norm.weight", "model.norm.bias", "lm_head.weight", "lm_head.bias"})
        self.assertEqual(len(tracked), 14)

    def test_loo_gradient_matches_explicit_deletion(self):
        torch.manual_seed(4)
        rewards = torch.randint(0, 2, (2048,)).double()
        theta = torch.randn(3, dtype=torch.float64, requires_grad=True)
        scores = torch.randn(8, 3, dtype=torch.float64) @ theta
        expected = rewards[:8] - (rewards.sum() - rewards[:8]) / 2047
        direct = torch.autograd.grad(-(expected * scores).mean(), theta, retain_graph=True)[0]
        actual = reinforcepp_loo_advantages(rewards[:8], rewards.mean(), 2048)
        derived = torch.autograd.grad(-(actual * scores).mean(), theta)[0]
        torch.testing.assert_close(direct, derived, rtol=1e-12, atol=1e-14)

    def test_paper_launcher_uses_a100_global_plan_and_shared_forward(self):
        result = subprocess.run([sys.executable, str(ROOT / "experiments/run.py"),
            "--experiment", "gradient_mse", "--dataset", "math", "--step", "10",
            "--checkpoint", "model", "--output", "results/preview"], capture_output=True, text=True, check=True)
        self.assertIn("gradient_mse/global_loo.py", result.stdout)
        self.assertIn("experiments/gradient_mse/draws/math/step_10", result.stdout)
        self.assertIn("--reference-microbatch 8 --repeat-microbatch 0", result.stdout)
        self.assertIn("--methods reinforcepp grpo_no_norm overt_no_norm", result.stdout)
        self.assertIn("gradient_mse/verify_checkpoint.py", result.stdout)
        self.assertNotIn("--math-full-parameters", result.stdout)

    def test_aggregator_rejects_old_scope_and_shard_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            formal = base / "formal"
            shard = formal / "shard_0_of_1"
            shard.mkdir(parents=True)
            config = dict(target_step_abs=10, k_eval=8, m_repeats=50,
                num_prompt_shards=1, prompt_shard_index=0, sampling_with_replacement=True,
                linear_alpha=0.2, ema_min_var=0.01, dtype="bf16", seed=0,
                history_steps=[6, 7, 8, 9], model_path=str(base / "model"), logprob_micro_batch=8,
                reinforcepp_baseline="global_batch_completion_leave_one_out",
                reinforcepp_batch_prompts=256, reinforcepp_batch_completions=2048,
                reinforcepp_batch_uid_sha256="1" * 64, sampling_plan_sha256="2" * 64,
                grad_scope_info=dict(vars(math_parameter_scope()), selected_layer_ids=[23,24,25,26,27]))
            summary = dict(config=config, selection_stats=dict(prompts_before_sharding=256, selected_prompts=256),
                           num_prompts_succeeded=256, num_prompts_failed=0)
            frame = pd.DataFrame(dict(uid=[str(i) for i in range(256)],
                trace_ref_error_grpo_no_norm=np.ones(256)*2,
                trace_ref_error_overt_no_norm=np.ones(256), trace_ref_error_reinforcepp=np.ones(256)*3))
            frame.to_parquet(shard / "run_detail.parquet")
            def run(name):
                (shard / "run_summary.json").write_text(json.dumps(summary))
                return subprocess.run([sys.executable, str(ROOT / "experiments/gradient_mse/aggregate.py"),
                    "--formal-dir", str(formal), "--dataset", "math", "--step", "10", "--checkpoint", str(base / "model"),
                    "--output-dir", str(base / name)], text=True, capture_output=True)
            valid = run("valid")
            self.assertEqual(valid.returncode, 0, valid.stderr)
            result = json.loads((base / "valid/gradient_mse_summary.json").read_text())
            self.assertEqual(result["reinforcepp_mean_oracle_centered_trace_error"], 3)
            config["grad_scope_info"]["grad_last_n_layers"] = 1
            self.assertNotEqual(run("wrong_scope").returncode, 0)
            config["grad_scope_info"]["grad_last_n_layers"] = 5
            config["reinforcepp_baseline"] = "shard_mean"
            self.assertNotEqual(run("wrong_rfpp").returncode, 0)

    def test_release_contains_no_mse_sample_parquets(self):
        self.assertEqual(list((ROOT / "data/rollouts").rglob("*.parquet")), [])
        self.assertEqual(list((ROOT / "data/value_mse").rglob("*.parquet")), [])


if __name__ == "__main__":
    unittest.main()
