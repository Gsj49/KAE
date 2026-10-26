"""One-shot/single-stream semantics and code-evaluation I/O; no candidate execution."""

import ast
import importlib.util
import json
from pathlib import Path
import sys
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
import pandas as pd
import torch

from test_paper_components import ROOT, estimator_namespace
from illustrations import options, parse_args
from prepare_oneshot import prepare, validate_oneshot


def load_code(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / f"test_scripts/code/{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


common = load_code("common")
score = load_code("score")
worker = load_code("worker")


class IllustrationTest(unittest.TestCase):
    def config(self, experiment, method, dataset="gsm8k"):
        argv = ["--method", method, "--train-data", "train.parquet", "--val-data", "val.parquet", "--output", "results/demo"]
        if experiment == "single_stream":
            argv += ["--dataset", dataset]
        return options(experiment, parse_args(experiment, argv))

    def test_one_shot_presets(self):
        for method, estimator in (("grpo", "grpo"), ("pool_oracle", "grpo_pool"), ("kae", "grpo_kae_tri")):
            cfg = self.config("one_shot", method)
            self.assertEqual(cfg["algorithm.adv_estimator"], estimator)
            self.assertEqual(cfg["data.train_batch_size"] * cfg["actor_rollout_ref.rollout.n"], 1024)
            self.assertEqual(cfg["trainer.total_training_steps"], 100)
            self.assertFalse(cfg["algorithm.norm_adv_by_std_in_grpo"])
            self.assertFalse(cfg["+data.fixed_repeat.enabled"])
            self.assertEqual(cfg["actor_rollout_ref.actor.loss_agg_mode"], "seq-mean-token-mean")
            if method == "kae":
                self.assertEqual(cfg["+algorithm.kae_ih"], 5.0)

    def test_single_stream_presets(self):
        for dataset, gpus, tokens in (("gsm8k", 1, 512), ("math", 4, 2048)):
            for method in ("reinforce", "reinforce_schedule", "kae"):
                cfg = self.config("single_stream", method, dataset)
                self.assertEqual(cfg["actor_rollout_ref.rollout.n"], 1)
                self.assertEqual(cfg["data.train_batch_size"], 1024)
                self.assertEqual(cfg["trainer.total_training_steps"], 400)
                self.assertEqual(cfg["trainer.n_gpus_per_node"], gpus)
                self.assertEqual(cfg["data.max_response_length"], tokens)
                self.assertIn("1.5B", cfg["actor_rollout_ref.model.path"])
                self.assertEqual(cfg["+data.fixed_repeat.enabled"], method != "reinforce")
                if method == "kae":
                    self.assertEqual(cfg["+algorithm.kae_ih"], 5.0)

    def test_illustration_bandwidth_override(self):
        for experiment in ("one_shot", "single_stream"):
            argv = ["--method", "kae", "--kernel-ih", "10", "--output", "results/demo"]
            if experiment == "single_stream":
                argv.extend(["--dataset", "gsm8k"])
            cfg = options(experiment, parse_args(experiment, argv))
            self.assertEqual(cfg["+algorithm.kae_ih"], 10.0)

    def test_copy_and_pool_ids_are_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "source.parquet", root / "copies.parquet"
            pd.DataFrame([dict(prompt=[{"role": "user", "content": "1+1?"}], reward_model={"ground_truth": "2"},
                               data_source="math", extra_info={"index": 10})]).to_parquet(source)
            prepare(source, output, row=0)
            frame = pd.read_parquet(output)
            validate_oneshot(output)
            self.assertEqual(len({row["index"] for row in frame.extra_info}), 128)
            self.assertEqual(len({row["pool_id"] for row in frame.extra_info}), 1)
            with self.assertRaises(FileExistsError):
                prepare(source, output, row=0)
            frame.at[0, "prompt"] = [{"role": "user", "content": "different question"}]
            frame.to_parquet(output)
            with self.assertRaises(ValueError):
                validate_oneshot(output)

    def test_group_mean_is_not_pool_mean(self):
        ns = estimator_namespace()
        fn = ns["compute_grpo_outcome_advantage"]
        reward = torch.tensor([[0.], [0.], [1.], [1.]])
        mask = torch.ones_like(reward)
        local, _ = fn(reward, mask, np.array(["a", "a", "b", "b"]), norm_adv_by_std_in_grpo=False)
        pooled, _ = fn(reward, mask, np.array(["pool"] * 4), norm_adv_by_std_in_grpo=False)
        torch.testing.assert_close(local, torch.zeros_like(local))
        torch.testing.assert_close(pooled[:, 0], torch.tensor([-.5, -.5, .5, .5]))

    def test_history_and_pool_selectors(self):
        tree = ast.parse((ROOT / "verl/trainer/ppo/kae_state.py").read_text())
        nodes = [n for n in tree.body if getattr(n, "name", "") in ("pooled_indices", "historical_indices")]
        namespace = {"np": np}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "identities", "exec"), namespace)
        batch = {"extra_info": [{"index": "copy0", "pool_id": "q"}, {"index": "copy1", "pool_id": "q"}]}
        self.assertEqual(namespace["historical_indices"](batch).tolist(), ["copy0", "copy1"])
        self.assertEqual(namespace["pooled_indices"](batch).tolist(), ["q", "q"])
        with self.assertRaises(ValueError):
            namespace["pooled_indices"]({"extra_info": [{"index": "copy0"}]})

    def test_single_response_behavior_preserved(self):
        ns = estimator_namespace()
        ids, mask = np.array(["a", "b"]), torch.ones(2, 1)
        reward = torch.tensor([[0.], [1.]])
        grpo, _ = ns["compute_grpo_outcome_advantage"](reward, mask, ids, norm_adv_by_std_in_grpo=False)
        torch.testing.assert_close(grpo, reward)
        fn = ns["compute_grpo_kae_tri_outcome_advantage"]
        first, _ = fn(torch.zeros(2, 1), mask, ids, config=SimpleNamespace(kae_ih=5.0),
                      norm_adv_by_std_in_grpo=False, global_steps=1)
        second, _ = fn(torch.ones(2, 1), mask, ids, config=SimpleNamespace(kae_ih=5.0),
                       norm_adv_by_std_in_grpo=False, global_steps=2)
        torch.testing.assert_close(first, torch.zeros_like(first))
        torch.testing.assert_close(second, torch.full_like(second, 1 - 1 / 1.8))


class CodeEvaluationTest(unittest.TestCase):
    def test_metrics_are_problem_based(self):
        values = [[1] + [0] * 15, [0] * 16]
        self.assertEqual(common.metrics(values), {"pass@16": .5, "avg@16": 1 / 32})
        self.assertEqual(common.metrics([[1], [0]]), {"pass@1": .5, "avg@1": .5})
        for invalid in ([[float("nan")]], [[.2]], [[]]):
            with self.assertRaises(ValueError):
                common.metrics(invalid)

    def test_extraction_and_gold_validation(self):
        self.assertEqual(worker.extract_code("```python\none\n```\n```python\ntwo\n```"), "two\n")
        self.assertEqual(worker.extract_code("raw text"), "raw text")
        with self.assertRaises(ValueError):
            common.gold_record({"reward_model": {"ground_truth": "{}"}}, "humaneval_plus")

    def test_isolation_command(self):
        args = SimpleNamespace(backend="docker", image="sha256:" + "a" * 64, workers=64, cpus=72, smoke=False)
        command = score.sandbox_command(args, Path("/tmp/isolated"))
        for flag in ("--network=none", "--read-only", "--cap-drop=ALL", "--pids-limit=512", "--user=65534:65534"):
            self.assertIn(flag, command)
        self.assertNotIn(str(ROOT), " ".join(command))
        args.image = "mutable:latest"
        with self.assertRaises(ValueError):
            score.sandbox_command(args, Path("/tmp/isolated"))

    def test_three_run_aggregation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = []
            for seed, passing in ((1, 0), (42, 8), (420, 16)):
                run = root / str(seed)
                run.mkdir()
                outputs, measured = {}, {}
                for bench, count in common.COUNTS.items():
                    path = run / f"{bench}.parquet"
                    rewards = [[1] * passing + [0] * (16 - passing)] * count
                    frame = pd.DataFrame(dict(reward_model=[{"ground_truth": json.dumps({"entry_point": "f", "test": "assert True"})}] * count,
                                              responses=[["never executed"] * 16] * count, rewards=rewards))
                    frame.to_parquet(path, index=False)
                    outputs[bench] = dict(sha256=common.sha256(path))
                    measured[bench] = common.metrics(rewards)
                summary = dict(status="complete", smoke=False, timeout=15,
                               generation=dict(method="kae", train_seed=seed, protocol={"n": 16}, input_sha256={"synthetic": True}),
                               outputs=outputs, metrics=measured, worker_sha256="synthetic", common_sha256="synthetic")
                path = run / "summary.json"
                path.write_text(json.dumps(summary))
                paths.append(str(path))
            command = [sys.executable, str(ROOT / "test_scripts/code/aggregate.py"), *paths, "--output", str(root / "aggregate")]
            subprocess.run(command, check=True, capture_output=True, text=True)
            frame = pd.read_csv(root / "aggregate/mean_sd_se.csv")
            rows = frame[frame.metric == "avg@16"]
            np.testing.assert_allclose(rows["mean"], .5)
            np.testing.assert_allclose(rows["std"], .5)
            self.assertEqual(set(rows["benchmark"]), {"humaneval_plus", "mbpp_plus", "Avg"})


if __name__ == "__main__":
    unittest.main()
