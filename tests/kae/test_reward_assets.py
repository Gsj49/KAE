"""Lossless reward projection must preserve IDs/order and reject bad pools."""

import importlib.util
from pathlib import Path
import tempfile
import unittest

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "prepare_reward_assets", ROOT / "experiments/value_mse/prepare_reward_assets.py")
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class RewardProjectionTest(unittest.TestCase):
    def test_lossless_order_and_text_removed(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "full.parquet"
            target = Path(temporary) / "compact.parquet"
            pq.write_table(pa.Table.from_pylist([
                dict(extra_info=dict(index=9, question="q", answer="a"),
                     responses=["long text"] * 3, rewards=[1.0, 0.25, 0.0]),
                dict(extra_info=dict(index=3, question="q2", answer="a2"),
                     responses=["text"] * 3, rewards=[0.0, 0.0, 1.0]),
            ]), source)
            compact, digest, uids = MODULE.reward_table(source, 2, 3)
            self.assertEqual(compact.column_names, ["extra_info", "rewards"])
            self.assertEqual(compact["extra_info"].to_pylist(), [{"index": 9}, {"index": 3}])
            self.assertEqual(uids, {"9", "3"})
            pq.write_table(compact, target, compression="zstd")
            restored, repeated_digest, _ = MODULE.reward_table(target, 2, 3)
            self.assertTrue(compact.equals(restored))
            self.assertEqual(digest, repeated_digest)

    def test_bad_ids_and_rewards_rejected(self):
        cases = [
            [dict(extra_info=dict(index=1), rewards=[0.0, 1.0])] * 2,
            [dict(extra_info=dict(index=1), rewards=[0.0, 1.0]),
             dict(extra_info=dict(index=2), rewards=[float("nan"), 1.0])],
            [dict(extra_info=dict(index=1), rewards=[0.0, 1.0]),
             dict(extra_info=dict(index=2), rewards=[-1.0, 1.0])],
        ]
        with tempfile.TemporaryDirectory() as temporary:
            for i, rows in enumerate(cases):
                path = Path(temporary) / f"invalid_{i}.parquet"
                pq.write_table(pa.Table.from_pylist(rows), path)
                with self.assertRaises(ValueError):
                    MODULE.reward_table(path, 2, 2)


if __name__ == "__main__":
    unittest.main()
