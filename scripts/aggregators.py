"""Aggregators. Every aggregator exposes exactly:
aggregate(param_vectors: list[np.ndarray], client_sizes: list[int]) -> np.ndarray
"""
import numpy as np


def fedavg(param_vectors, client_sizes) -> np.ndarray:
    """Weighted mean of client parameter vectors, weighted by client size."""
    weights = np.array(client_sizes, dtype=float)
    weights /= weights.sum()
    stacked = np.stack(param_vectors, axis=0)
    return np.average(stacked, axis=0, weights=weights)


def circular_mean(param_vectors, client_sizes) -> np.ndarray:
    """Arm 5 ablation. Only meaningful for angle-valued parameters (the
    VQC's rotation angles) -- a plain weighted mean of two angles near the
    wraparound point (e.g. 0.05 and 2*pi-0.05) is wrong (gives ~pi, not
    ~0). Circular mean averages on the unit circle instead:
    atan2(sum(w*sin(theta)), sum(w*cos(theta))) per parameter. Not used
    for the classical models (LR/MLP weights have no wraparound)."""
    weights = np.array(client_sizes, dtype=float)
    weights /= weights.sum()
    stacked = np.stack(param_vectors, axis=0)  # (n_clients, P)
    sin_sum = np.sum(weights[:, None] * np.sin(stacked), axis=0)
    cos_sum = np.sum(weights[:, None] * np.cos(stacked), axis=0)
    return np.arctan2(sin_sum, cos_sum)
