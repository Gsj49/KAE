# Copyright 2025 Amazon.com Inc and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from abc import abstractmethod
from collections.abc import Iterator, Sized

from omegaconf import DictConfig
from torch.utils.data import BatchSampler, Sampler

from verl import DataProto


class AbstractSampler(Sampler[int]):
    """Abstract interface for custom samplers."""

    @abstractmethod
    def __init__(
        self,
        data_source: Sized,
        data_config: DictConfig,
    ):
        pass


class AbstractCurriculumSampler(AbstractSampler):
    """Experimental interface for curriculum learning samplers."""

    @abstractmethod
    def update(self, batch: DataProto) -> None:
        pass


class FixedRepeatBatchSampler(Sampler[list[int]]):
    """Yield each prompt batch exactly ``k`` consecutive times.

    Each yield triggers new rollout generation. This repeats prompt indices, not
    responses, and is independent of rollout group size ``G``.
    """

    def __init__(self, base_batch_sampler: BatchSampler, k: int):
        if int(k) != k or k < 1:
            raise ValueError("repeat count must be a positive integer")
        self.base_batch_sampler = base_batch_sampler
        self.k = int(k)

    def __iter__(self) -> Iterator[list[int]]:
        for batch_indices in self.base_batch_sampler:
            for _ in range(self.k):
                yield list(batch_indices)

    def __len__(self) -> int:
        return len(self.base_batch_sampler) * self.k
