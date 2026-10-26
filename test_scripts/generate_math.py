# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""
Generate responses given a dataset of prompts
"""

import os

import hydra
import numpy as np
import ray

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from pprint import pprint

import pandas as pd
from omegaconf import OmegaConf

from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils import hf_tokenizer
from verl.utils.fs import copy_to_local
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

import re



@hydra.main(config_path="../verl/trainer/config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        default_runtime_env = {"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}}
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})
        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    local_path = copy_to_local(config.model.path)
    trust_remote_code = config.data.get("trust_remote_code", False)
    tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)

    if config.rollout.temperature == 0.0:
        assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."
    assert config.data.n_samples >= 1, "n_samples should always >= 1"

    # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
    dataset = pd.read_parquet(config.data.path)
    chat_lst = dataset[config.data.prompt_key].tolist()

    chat_lst = [chat.tolist() if hasattr(chat, "tolist") else chat for chat in chat_lst]

    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
    resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
    wg = RayWorkerGroup(
        resource_pool=resource_pool,
        ray_cls_with_init=ray_cls_with_init,
        device_name=config.trainer.device,
    )
    wg.init_model()

    total_samples = len(dataset)
    config_batch_size = config.data.batch_size
    apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
    num_batch = -(-total_samples // config_batch_size)
    output_lst = [[] for _ in range(config.data.n_samples)]

    for batch_idx in range(num_batch):
        print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
        batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
        inputs = tokenizer.apply_chat_template(
            batch_chat_lst,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=config.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
            **apply_chat_template_kwargs,
        )
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]
        position_ids = compute_position_id_with_mask(attention_mask)
        batch_dict = {"input_ids": input_ids, "attention_mask": attention_mask, "position_ids": position_ids}

        data = DataProto.from_dict(batch_dict)
        n = config.data.n_samples
        bs = len(batch_chat_lst)

        # data_repeated = data.repeat(repeat_times=n, interleave=False)
        data_repeated = data.repeat(repeat_times=n, interleave=True)

        data_padded, pad_size = pad_dataproto_to_divisor(data_repeated, wg.world_size)

        print(f"[{batch_idx + 1}/{num_batch}] Start to generate (n={n} in one pass).")
        output_padded = wg.generate_sequences(data_padded)
        output = unpad_dataproto(output_padded, pad_size=pad_size)

        decoded_all = []
        for i in range(len(output)):
            data_item = output[i]
            prompt_length = data_item.batch["prompts"].shape[-1]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = data_item.batch["responses"][:valid_response_length]
            decoded_all.append(tokenizer.decode(valid_response_ids, skip_special_tokens=True))

        assert len(decoded_all) == n * bs

        # 原始 flat 列表 decoded_all 的结构是:
        # [P1_S1, P1_S2, ..., P1_Sn, P2_S1, P2_S2, ..., P2_Sn, ...] (P=Prompt, S=Sample)
        # 1. reshape(bs, n) -> 得到 (Batch, Samples) 的矩阵
        # 2. transpose(1, 0) -> 转置为 (Samples, Batch) 的矩阵，即 (n, bs)
        decoded_mat = np.array(decoded_all, dtype=object).reshape(bs, n).transpose(1, 0)
        assert decoded_mat.shape == (n, bs)

        # decoded_mat = np.array(decoded_all, dtype=object).reshape(n, bs)

        # 直接写回到 output_lst：第 s 轮采样，追加该轮的整个 batch
        for s in range(n):
            output_lst[s].extend(decoded_mat[s].tolist())

    # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
    output_lst = np.array(output_lst, dtype=object)
    output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

    # add to the data frame
    dataset["responses"] = output_lst

    # Save generated samples before reward computation so generation work is not lost
    # if reward scoring (e.g. math-verify) hangs or fails.
    output_root, output_ext = os.path.splitext(config.data.output_path)
    pre_reward_output_path = f"{output_root}_pre_reward{output_ext or '.parquet'}"
    output_dir = os.path.dirname(config.data.output_path)
    makedirs(output_dir, exist_ok=True)
    print(f"Saving pre-reward samples to: {pre_reward_output_path}")
    dataset.to_parquet(pre_reward_output_path)

    # compute reward score
    rewards = []
    for i in range(len(dataset)):
        # 取每条样本的 ground_truth
        rm = dataset.iloc[i]["reward_model"]
        gt = None
        if isinstance(rm, dict) and "ground_truth" in rm:
            gt = rm["ground_truth"]

        # 对该样本的 n 个 responses 打分（没有 gt 就写 None）
        resp_list = dataset.iloc[i]["responses"]
        if gt is None:
            raise ValueError(f"Missing ground truth at row {i}")
        else:
            from verl.utils.reward_score.kae_math_verify import compute_score
            scores_i = [float(compute_score(dataset.iloc[i]["data_source"], resp, gt)) for resp in resp_list]
            rewards.append(scores_i)
    dataset["rewards"] = rewards


    # write to a new parquet
    makedirs(output_dir, exist_ok=True)
    dataset.to_parquet(config.data.output_path)


if __name__ == "__main__":
    main()
