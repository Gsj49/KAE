"""Math-Verify reward scoring with the paper's answer-extraction settings.

Parser timeouts return timeout_score; worker failures propagate to training.
"""

import math

import ray


@ray.remote(num_gpus=0)
class MathVerifyWorker:
    def __init__(self):
        from math_verify.metric import math_metric
        from math_verify.parser import ExprExtractionConfig, LatexExtractionConfig

        self.metric = math_metric(
            gold_extraction_target=(LatexExtractionConfig(),),
            pred_extraction_target=(ExprExtractionConfig(), LatexExtractionConfig()),
        )

    def score(self, solution, ground_truth, timeout_score):
        from math_verify.errors import TimeoutException

        try:
            score, _ = self.metric(["\\boxed{" + str(ground_truth) + "}"], [solution])
        except TimeoutException:
            score = timeout_score
        score = float(score)
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(f"Invalid verifier score: {score}")
        return score


_worker = None


def compute_score(data_source, solution_str, ground_truth, extra_info=None, timeout_score=0.0):
    del data_source, extra_info
    global _worker
    if _worker is None:
        _worker = MathVerifyWorker.remote()
    # Ray timeout, actor death and unexpected verifier errors intentionally fail
    # the job. They must not silently produce a trainable all-zero reward batch.
    return float(ray.get(_worker.score.remote(solution_str, ground_truth, timeout_score), timeout=60.0))
