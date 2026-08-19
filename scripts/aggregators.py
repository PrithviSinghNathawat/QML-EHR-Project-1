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
