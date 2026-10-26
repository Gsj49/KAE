"""Check portable model manifests and the exact six persisted draw plans."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/gradient_mse"))
from global_plan import read_plan, legacy_draws
from verify_checkpoint import verify
from report_global_loo import validate as validate_result


class ReproductionAssetsTest(unittest.TestCase):
    def test_persisted_plans_match_rng_and_pool_manifests(self):
        assets = json.loads((ROOT / "data/rollout_assets.json").read_text())["assets"]
        lookup = {(a["dataset"], a["step"]): a for a in assets}
        for dataset in ("gsm8k", "math"):
            for step in (10, 90, 170):
                meta, data = read_plan(ROOT / "experiments/gradient_mse/draws" / dataset / f"step_{step}")
                self.assertEqual((meta["dataset"], meta["step"]), (dataset, step))
                order, indices = legacy_draws(sorted(data["uids"].tolist()), np.full((256,5),128))
                np.testing.assert_array_equal(order, data["uids"])
                np.testing.assert_array_equal(indices, data["indices"])
                for item in meta["inputs"]:
                    self.assertEqual(item["sha256"], lookup[dataset,item["step"]]["sha256"])

    def test_six_models_and_seventy_two_files(self):
        manifest = json.loads((ROOT / "experiments/gradient_mse/checkpoint_assets.json").read_text())
        self.assertEqual(manifest["repo_id"], "Gsj49/verl_reasoning_checkpoints")
        self.assertEqual(len(manifest["revision"]), 40)
        self.assertEqual({(r["dataset"],r["step"]) for r in manifest["checkpoints"]},
                         {(d,s) for d in ("gsm8k","math") for s in (10,90,170)})
        self.assertEqual(sum(len(r["files"]) for r in manifest["checkpoints"]), 72)
        for record in manifest["checkpoints"]:
            self.assertEqual(record["bytes"], sum(f["bytes"] for f in record["files"]))
            self.assertEqual(len(record["files"]), record["file_count"])
            for f in record["files"]:
                self.assertEqual(len(f["sha256"]), 64)

    def test_checkpoint_verification_is_portable_and_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "downloaded_model"
            model.mkdir()
            (model / "config.json").write_bytes(b"{}")
            manifest = dict(repo_id="test/model", revision="0"*40, checkpoints=[
                dict(dataset="math", step=10, files=[dict(name="config.json",bytes=2,
                     sha256=hashlib.sha256(b"{}").hexdigest())])])
            path = root / "manifest.json"
            path.write_text(json.dumps(manifest))
            receipt = verify(model, "math", 10, path)
            self.assertEqual(receipt["checkpoints"][0]["path"], str(model.resolve()))
            self.assertEqual(receipt["status"], "complete")
            (model / "config.json").write_bytes(b"[]")
            with self.assertRaises(ValueError):
                verify(model, "math", 10, path)

    def test_global_report_checks_scope_draws_and_all_methods(self):
        meta, plan = read_plan(ROOT / "experiments/gradient_mse/draws/math/step_10")
        methods = ['reinforcepp','grpo_no_norm','overt_no_norm']
        cfg = dict(dataset='math',step=10,prompts=256,global_batch_prompts=256,
                   global_batch_completions=2048,k_eval=8,m_repeats=50,seed=0,dtype='bf16',
                   baseline='global_batch_completion_leave_one_out',
                   grad_scope='last5+model_norm+lm_head_no_embeddings',reference_pool_size=128,
                   sampling_with_replacement=True,score_reduction='sum',cov_device='cuda',
                   linear_alpha=0.2,variance_floor=0.01,history_steps=[6,7,8,9],
                   reference_microbatch=8,repeat_microbatch=0,methods=methods,
                   plan_sha256=meta['plan_sha256'])
        frame = pd.DataFrame(dict(uid=plan['uids'],global_index=np.arange(256),repeats=50,
                                  reference_pool_size=128,plan_sha256=meta['plan_sha256']))
        for method in methods:
            frame[f'trace_ref_error_{method}'] = 1.0
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            frame.to_parquet(root/'prompt_details.parquet',index=False)
            summary=dict(config=cfg,status='complete',num_prompts_succeeded=256,num_prompts_failed=0,
                         scope=dict(parameter_count=1710289920,selected_layer_ids=[23,24,25,26,27],
                                    include_model_norm=True,include_embed_tokens=False,include_lm_head=True),
                         detail_sha256=hashlib.sha256((root/'prompt_details.parquet').read_bytes()).hexdigest(),
                         avg_trace_ref_error={m:1.0 for m in methods})
            def save():
                (root/'summary.json').write_text(json.dumps(summary))
            save()
            validate_result(root,'math',10)
            cfg['plan_sha256']='0'*64
            save()
            with self.assertRaises(ValueError):
                validate_result(root,'math',10)
            cfg['plan_sha256']=meta['plan_sha256']
            summary['scope']['parameter_count']=778055168
            save()
            with self.assertRaises(ValueError):
                validate_result(root,'math',10)


if __name__ == "__main__":
    unittest.main()
