"""CPU-only tests for the actual estimator bodies and portable entry points.

AST loading avoids importing VERL's optional CUDA/vision stack. It does not
replace a real Linux/Ray/FSDP training smoke.
"""

import ast
from contextlib import redirect_stderr
from collections import defaultdict
from collections.abc import Iterator
from enum import Enum
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import torch
from torch.utils.data import BatchSampler, Sampler, SequentialSampler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train_scripts"))
from train import make_overrides, parse_args


# Test-only translation lets numerical checks read the original experiment
# source without keeping its old estimator names in the public runtime.
LEGACY_ESTIMATOR_NAMES = {
    "grpo_linear_group_std_fallback": "kae_std_fallback",
    "grpo_linear_std_mode": "kae_std_mode",
    "GRPO_EMA_v2": "GRPO_KAE_EXP",
    "GRPO_LINEAR": "GRPO_KAE_TRI",
    "grpo_ema_v2": "grpo_kae_exp",
    "grpo_linear": "grpo_kae_tri",
    "_GRPO_EMA_V2_CACHE": "_KAE_CACHE",
    "_EMAEntryV2": "_KAEHistory",
    "_ema_v2_get": "_get_kae_history",
}


def estimator_namespace(path=ROOT / "verl/trainer/ppo/core_algos.py"):
    names = {"AdvantageEstimator", "register_adv_est", "get_adv_estimator_fn", "_KAEHistory",
             "_get_kae_history", "compute_grpo_kae_tri_outcome_advantage", "compute_grpo_kae_exp_outcome_advantage",
             "compute_grpo_kalman_outcome_advantage", "compute_grpo_shrinkage_outcome_advantage",
             "compute_grpo_outcome_advantage"}
    nodes = []
    source = path.read_text()
    for old, new in LEGACY_ESTIMATOR_NAMES.items():
        source = source.replace(old, new)
    for node in ast.parse(source).body:
        if getattr(node, "name", "") in names:
            nodes.append(node)
    ns = dict(np=np, torch=torch, Enum=Enum, defaultdict=defaultdict, ADV_ESTIMATOR_REGISTRY={},
              _KAE_CACHE={})
    module = ast.Module(body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)] + nodes,
                        type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), ns)
    return ns


def sampler_class():
    path = ROOT / "verl/experimental/dataset/sampler.py"
    node = next(n for n in ast.parse(path.read_text()).body if getattr(n, "name", "") == "FixedRepeatBatchSampler")
    ns = dict(Sampler=Sampler, BatchSampler=BatchSampler, Iterator=Iterator)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    return ns["FixedRepeatBatchSampler"]


class EstimatorsTest(unittest.TestCase):
    def setUp(self):
        self.ns = estimator_namespace()

    def test_kernel_names_registered(self):
        for kernel in ("tri", "exp"):
            name = f"grpo_kae_{kernel}"
            self.assertIs(self.ns["get_adv_estimator_fn"](name),
                          self.ns[f"compute_{name}_outcome_advantage"])

    def call(self, name, rewards, step=1, **config):
        rewards = torch.tensor(rewards, dtype=torch.float64)
        mask = torch.ones_like(rewards)
        return self.ns[name](rewards, mask, np.repeat(["a", "b"], 4),
                             norm_adv_by_std_in_grpo=False, config=SimpleNamespace(**config),
                             global_steps=step)[0][:, 0]

    def test_triangle_matches_direct_weighted_leave_one_out(self):
        for width in (2.5, 5.0, 10.0):
            with self.subTest(kernel_ih=width):
                self.ns = estimator_namespace()
                history = []
                for step in range(1, 13):
                    rewards = np.random.default_rng(step).integers(0, 2, 8).astype(float)
                    history.append(rewards.reshape(2, 4).mean(1))
                    actual = self.call("compute_grpo_kae_tri_outcome_advantage", rewards[:, None], step,
                                       kae_ih=width)
                    weights = np.maximum(1 - np.arange(step - 1, -1, -1) / width, 0)
                    mean = weights @ np.array(history) / weights.sum()
                    expected = (rewards - np.repeat(mean, 4)) / (1 - 1 / (4 * weights.sum()))
                    np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_fixed_bandwidth_is_independent_of_absolute_step(self):
        results = []
        for offset in (0, 30, 100):
            self.ns = estimator_namespace()
            results.append([self.call("compute_grpo_kae_tri_outcome_advantage",
                                     np.random.default_rng(step).integers(0, 2, (8, 1)),
                                     step + offset, kae_ih=10.0)
                            for step in range(1, 13)])
        for shifted in results[1:]:
            np.testing.assert_array_equal(np.array(shifted), np.array(results[0]))

    def test_invalid_bandwidth_is_rejected(self):
        for width in (0, -1, float("nan"), float("inf")):
            with self.subTest(kernel_ih=width), self.assertRaises(ValueError):
                self.call("compute_grpo_kae_tri_outcome_advantage", [[1.0]] * 8, kae_ih=width)

    def test_optional_bandwidth_switch(self):
        history = []
        for step in range(1, 8):
            rewards = np.random.default_rng(step).integers(0, 2, 8).astype(float)
            history.append(rewards.reshape(2, 4).mean(1))
            actual = self.call("compute_grpo_kae_tri_outcome_advantage", rewards[:, None], step,
                               kae_ih=5.0, kae_ih_after=10.0, kae_ih_switch_step=2)
            width = 5.0 if step <= 2 else 10.0
            weights = np.maximum(1 - np.arange(step - 1, -1, -1) / width, 0)
            mean = weights @ np.array(history) / weights.sum()
            expected = (rewards - np.repeat(mean, 4)) / (1 - 1 / (4 * weights.sum()))
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_zero_switch_step_keeps_initial_bandwidth(self):
        values = []
        for after in (None, 5.0):
            self.ns = estimator_namespace()
            values.append([self.call("compute_grpo_kae_tri_outcome_advantage",
                                     np.random.default_rng(step).integers(0, 2, (8, 1)), step,
                                     kae_ih=10.0, kae_ih_after=after, kae_ih_switch_step=0)
                           for step in range(0, 12)])
        np.testing.assert_array_equal(np.array(values[0]), np.array(values[1]))

    def test_history_resets_after_gap(self):
        self.call("compute_grpo_kae_tri_outcome_advantage", [[1.0]] * 8, 1)
        actual = self.call("compute_grpo_kae_tri_outcome_advantage", [[0.0]] * 8, 9)
        np.testing.assert_array_equal(actual, np.zeros(8))

    def test_ema_exact_completion_weight(self):
        history = []
        for step in range(1, 7):
            rewards = np.random.default_rng(step).integers(0, 2, 8).astype(float)
            history.append(rewards.reshape(2, 4).mean(1))
            mean = history[0].copy()
            for x in history[1:]:
                mean = 0.8 * mean + 0.2 * x
            expected = (rewards - np.repeat(mean, 4)) / (1 - (1 if step == 1 else 0.2) / 4)
            actual = self.call("compute_grpo_kae_exp_outcome_advantage", rewards[:, None], step,
                               ema_beta=0.2, ema_leave_one_out=True)
            np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_constant_reward_normalization_fallback(self):
        fn = self.ns["compute_grpo_kae_tri_outcome_advantage"]
        cfg = SimpleNamespace(kae_ih=10.0, kae_std_mode="group",
                              kae_std_fallback=0.5, ema_min_var=0.01)
        ids = np.repeat(["a", "b"], 4)
        fn(torch.zeros(8, 1), torch.ones(8, 1), ids, config=cfg, global_steps=1)
        actual, _ = fn(torch.ones(8, 1), torch.ones(8, 1), ids, config=cfg, global_steps=2)
        baseline = 1 / 1.9
        expected = (1 - baseline) / (1 - 1 / 7.6) / (0.5 + 1e-6)
        torch.testing.assert_close(actual, torch.full_like(actual, expected))

    def test_kalman_and_js_finite(self):
        rewards = [[0.0], [1.0], [0.0], [1.0], [1.0], [1.0], [1.0], [1.0]]
        for name in ("compute_grpo_kalman_outcome_advantage", "compute_grpo_shrinkage_outcome_advantage"):
            self.assertTrue(torch.isfinite(self.call(name, rewards)).all())


class SamplerTest(unittest.TestCase):
    def test_exact_repeated_batches(self):
        cls = sampler_class()
        sampler = cls(BatchSampler(SequentialSampler(range(10)), 3, drop_last=True), 4)
        self.assertEqual(len(sampler), 12)
        self.assertEqual(list(sampler), [[0, 1, 2]] * 4 + [[3, 4, 5]] * 4 + [[6, 7, 8]] * 4)
        with self.assertRaises(ValueError):
            cls(BatchSampler(SequentialSampler(range(10)), 3, True), 0)

    def test_stateful_loader_resume_inside_repeat(self):
        try:
            from torchdata.stateful_dataloader import StatefulDataLoader
        except ImportError:
            self.skipTest("torchdata is needed for DataLoader resume coverage")
        def create():
            return StatefulDataLoader(list(range(12)), batch_sampler=sampler_class()(
                BatchSampler(SequentialSampler(range(12)), 3, drop_last=True), 4), num_workers=0)
        loader = create()
        it = iter(loader)
        for _ in range(5):
            next(it)
        state = loader.state_dict()
        expected = [x.tolist() for x in it]
        restored = create()
        restored.load_state_dict(state)
        self.assertEqual([x.tolist() for x in restored], expected)


class ConfigurationTest(unittest.TestCase):
    def options(self, dataset, method):
        return make_overrides(parse_args(["--dataset", dataset, "--method", method, "--train-data", "train.parquet",
                                          "--val-data", "val.parquet", "--output", "results/test"]))

    def test_policy_matrix(self):
        for dataset, steps, group, repeat, length in (
            ("dapo", 500, 4, 8, 3072), ("math", 200, 8, 10, 2048), ("gsm8k", 200, 8, 10, 512)):
            for method in ("grpo", "gpg", "drgrpo", "js", "kalman", "ema", "kae", "kae_norm"):
                c = self.options(dataset, method)
                self.assertEqual(c["trainer.total_training_steps"], steps)
                self.assertEqual(c["actor_rollout_ref.rollout.n"], group)
                self.assertEqual(c["+data.fixed_repeat.repeat_count"], repeat)
                self.assertEqual(c["data.max_response_length"], length)
                if method.startswith("kae"):
                    self.assertEqual(c["+algorithm.kae_ih"], 10.0)
                    self.assertIsNone(c["+algorithm.kae_ih_after"])
                    self.assertEqual(c["+algorithm.kae_ih_switch_step"], 0)
                    self.assertFalse(any("linear_alpha" in key for key in c))
                    self.assertEqual(c["algorithm.adv_estimator"], "grpo_kae_tri")
                elif method == "ema":
                    self.assertEqual(c["algorithm.adv_estimator"], "grpo_kae_exp")
            self.assertEqual(self.options(dataset, "js")["actor_rollout_ref.actor.loss_agg_mode"], "seq-mean-token-mean")
            self.assertEqual(self.options(dataset, "drgrpo")["actor_rollout_ref.actor.loss_agg_mode"], "seq-mean-token-sum-norm")
            self.assertTrue(self.options(dataset, "ema")["+algorithm.ema_leave_one_out"])

    def test_bandwidth_cli_override(self):
        for method in ("kae", "kae_norm"):
            args = parse_args(["--dataset", "dapo", "--method", method,
                               "--kernel-ih", "5", "--output", "results/test"])
            self.assertEqual(make_overrides(args)["+algorithm.kae_ih"], 5.0)

    def test_optional_switch_cli(self):
        args = parse_args(["--dataset", "dapo", "--method", "kae", "--kernel-ih", "5",
                           "--kernel-ih-after", "10", "--kernel-switch-step", "30",
                           "--output", "results/test"])
        cfg = make_overrides(args)
        self.assertEqual(cfg["+algorithm.kae_ih"], 5.0)
        self.assertEqual(cfg["+algorithm.kae_ih_after"], 10.0)
        self.assertEqual(cfg["+algorithm.kae_ih_switch_step"], 30)

    def test_bandwidth_cli_rejects_invalid_or_unused_values(self):
        for method, width in (("kae", "0"), ("kae", "-1"), ("kae", "nan"),
                              ("kae", "inf"), ("grpo", "10"), ("ema", "10")):
            with self.subTest(method=method, width=width), redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    parse_args(["--dataset", "dapo", "--method", method,
                                "--kernel-ih", width, "--output", "results/test"])


if __name__ == "__main__":
    unittest.main()
