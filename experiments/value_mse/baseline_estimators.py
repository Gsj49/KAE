"""NumPy implementations of the baseline estimators."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np


def leave_one_out_mean(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.size < 2:
        raise ValueError("Leave-one-out baselines require at least two completions")
    return (values.sum() - values) / (values.size - 1)


def global_batch_leave_one_out(groups: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Exclude the current completion from the reward mean of the full batch.

    A computational shard is not a statistical batch: pass all prompt groups
    before selecting the prompts whose gradients will be evaluated locally.
    """
    arrays = {key: np.asarray(values, dtype=np.float64) for key, values in groups.items()}
    if any(values.ndim != 1 or not np.isfinite(values).all() for values in arrays.values()):
        raise ValueError("Reward groups must be finite one-dimensional arrays")
    count = sum(values.size for values in arrays.values())
    if count < 2:
        raise ValueError("Global leave-one-out requires at least two completions")
    total = sum(float(values.sum()) for values in arrays.values())
    return {key: (total - values) / (count - 1) for key, values in arrays.items()}


def kalman_posterior_baselines(
    rewards: np.ndarray,
    process_variance: float = 1e-5,
    measurement_variance: float = 1e-2,
    initial_mean: float = 0.0,
    initial_variance: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """KRPO baseline: reset per prompt and update sequentially within the group."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if process_variance < 0 or measurement_variance <= 0 or initial_variance <= 0:
        raise ValueError("Kalman variances must satisfy Q>=0, R>0, P0>0")
    mean = float(initial_mean)
    variance = float(initial_variance)
    baselines = np.empty_like(rewards)
    posterior_variances = np.empty_like(rewards)
    for position, reward in enumerate(rewards):
        predicted_variance = variance + process_variance
        gain = predicted_variance / (predicted_variance + measurement_variance)
        mean = mean + gain * (float(reward) - mean)
        variance = (1.0 - gain) * predicted_variance
        baselines[position] = mean
        posterior_variances[position] = variance
    return baselines, posterior_variances


def shrinkage_baselines(
    groups: Mapping[str, np.ndarray],
    epsilon: float = 1e-12,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Two-level leave-one-out shrinkage baseline from arXiv:2511.03710."""
    keys = list(groups)
    if len(keys) < 2:
        raise ValueError("Shrinkage baseline requires at least two prompts")
    arrays = {key: np.asarray(groups[key], dtype=np.float64) for key in keys}
    group_sizes = {array.size for array in arrays.values()}
    if len(group_sizes) != 1 or next(iter(group_sizes)) < 2:
        raise ValueError("Shrinkage baseline requires an equal group size m>=2 for every prompt")

    m = next(iter(group_sizes))
    n = len(keys)
    means = {key: float(arrays[key].mean()) for key in keys}
    # Estimated variance of the sample mean for each prompt.
    prompt_mean_variances = {key: float(arrays[key].var(ddof=1) / m) for key in keys}
    outputs: dict[str, np.ndarray] = {}
    lambdas: dict[str, float] = {}
    for key in keys:
        other_keys = [other for other in keys if other != key]
        global_loo_mean = float(np.mean([means[other] for other in other_keys]))
        noise_variance = float(np.mean([prompt_mean_variances[other] for other in other_keys]))
        signal_variance = float(np.mean([(means[other] - global_loo_mean) ** 2 for other in other_keys]))
        denominator = noise_variance + signal_variance
        shrinkage = 0.0 if denominator <= epsilon else ((n - 1.0) / n) * noise_variance / denominator
        shrinkage = float(np.clip(shrinkage, 0.0, 1.0))
        local_loo = leave_one_out_mean(arrays[key])
        outputs[key] = (1.0 - shrinkage) * local_loo + shrinkage * global_loo_mean
        lambdas[key] = shrinkage
    return outputs, lambdas


def triangular_kae_baselines(
    rewards: np.ndarray,
    history_means: list[float],
    alpha: float,
) -> np.ndarray:
    """Existing KAE linear-kernel estimator with completion-level LOO correction."""
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.size < 2:
        raise ValueError("KAE LOO correction requires at least two current completions")
    means = np.asarray(history_means + [float(rewards.mean())], dtype=np.float64)
    distances = (means.size - 1) - np.arange(means.size)
    weights = np.maximum(1.0 - float(alpha) * distances, 0.0)
    valid = np.isfinite(means) & (weights > 0)
    if not valid.any():
        return np.full_like(rewards, np.nan)
    weights = weights[valid]
    weights /= weights.sum()
    clean_mean = float(np.dot(means[valid], weights))
    current_weight = float(weights[-1])
    denominator = 1.0 - current_weight / rewards.size
    if denominator <= 1e-12:
        return np.full_like(rewards, np.nan)
    return (clean_mean - (current_weight / rewards.size) * rewards) / denominator
