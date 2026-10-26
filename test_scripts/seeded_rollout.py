"""Explicit evaluation-engine seed without changing the training rollout config."""

from dataclasses import dataclass
from verl.workers.config import RolloutConfig


@dataclass
class SeededRolloutConfig(RolloutConfig):
    seed: int = 0
