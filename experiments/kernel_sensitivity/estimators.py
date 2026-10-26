"""Kernel weights and leave-one-out operations for the sensitivity analysis."""

import numpy as np

ALPHAS = [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]
BETAS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]


def linear_weighted_mean_with_current_weight(values, alpha):
    vals = np.array(values, dtype=np.float32)
    if vals.size == 0:
        return np.nan, np.nan
    dist = (vals.size - 1) - np.arange(vals.size, dtype=np.float32)
    raw_w = np.maximum(1.0 - alpha * dist, 0.0)
    valid = (~np.isnan(vals)) & (raw_w > 0.0)
    if not valid.any():
        return np.nan, np.nan
    weights = raw_w[valid]
    weights = weights / float(weights.sum())
    return float((vals[valid] * weights).sum()), float(weights[-1])


def ema_weighted_mean_with_current_weight(values, beta):
    vals = np.array(values, dtype=np.float32)
    vals = vals[~np.isnan(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    if vals.size == 1:
        return float(vals[0]), 1.0
    weights = np.empty(vals.size, dtype=np.float32)
    weights[0] = float((1.0 - beta) ** (vals.size - 1))
    exponents = np.arange(vals.size - 2, -1, -1, dtype=np.float32)
    weights[1:] = float(beta) * np.power(float(1.0 - beta), exponents, dtype=np.float32)
    weights /= float(weights.sum())
    return float(np.dot(vals, weights)), float(weights[-1])


def loo_baselines_from_clean_mean(rewards, clean_mean, current_step_weight):
    if rewards.size < 2 or not np.isfinite([clean_mean, current_step_weight]).all():
        raise ValueError("Leave-one-out requires finite inputs and at least two rewards")
    denom = 1.0 - float(current_step_weight) / float(rewards.size)
    if denom <= 1e-8:
        raise ValueError("Undefined leave-one-out denominator")
    baselines = (float(clean_mean) - (float(current_step_weight) / float(rewards.size)) * rewards) / denom
    return baselines.astype(np.float32, copy=False)


def bandwidth(family, parameter, step):
    """Return H=i*h_i for K(u)=(1-u)+ or exp(-u), and the paper's h_i."""
    if family == "triangular":
        width = 1.0 / parameter
    elif family == "exponential":
        width = 0.0 if parameter == 1.0 else -1.0 / np.log1p(-parameter)
    else:
        raise ValueError(family)
    return float(width), float(width / step)
