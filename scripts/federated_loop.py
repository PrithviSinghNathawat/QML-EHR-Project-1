"""The one shared federated training loop (CLAUDE.md architecture: one loop,
not five scripts). Only ever calls the four Model interface methods and the
aggregator's aggregate() -- never branches on what kind of model or
aggregator it's holding. If a quantum model and a classical model are both
passed in here later, this file should not need to change.
"""
import numpy as np


def run_centralized(model_factory, X_train, y_train, epochs: int):
    """Arm 1: no federation, one model trained on the pooled training set."""
    model = model_factory()
    model.fit(X_train, y_train, epochs=epochs)
    return model


def _mean_pairwise_l2(param_vectors) -> float:
    n = len(param_vectors)
    if n < 2:
        return 0.0
    dists = [
        np.linalg.norm(param_vectors[i] - param_vectors[j])
        for i in range(n)
        for j in range(i + 1, n)
    ]
    return float(np.mean(dists))


def run_federated(
    model_factory, aggregator, client_data, rounds: int, local_epochs: int, track_divergence: bool = False
):
    """client_data: list of (X_client, y_client) arrays, one per client.
    aggregator: a function matching aggregate(param_vectors, client_sizes).
    Returns the final global model, holding the aggregated parameters.

    track_divergence (interface amendment, added for the 2026-08-18
    diagnostic session -- see docs/decisions.md and docs/INTERFACE.md):
    when True, returns (model, divergence_per_round) instead of just
    model, where divergence_per_round[r] is the mean pairwise L2 distance
    between client parameter vectors after local training but before
    aggregation, in round r. Default False preserves the original return
    contract for existing callers (scripts/run_grid.py)."""
    global_model = model_factory()
    global_params = global_model.get_params()
    divergence_per_round = []

    for _ in range(rounds):
        client_params = []
        client_sizes = []
        for X_c, y_c in client_data:
            local_model = model_factory()
            local_model.set_params(global_params.copy())
            local_model.fit(X_c, y_c, epochs=local_epochs)
            client_params.append(local_model.get_params())
            client_sizes.append(len(X_c))
        if track_divergence:
            divergence_per_round.append(_mean_pairwise_l2(client_params))
        global_params = aggregator(client_params, client_sizes)

    global_model.set_params(global_params)
    if track_divergence:
        return global_model, divergence_per_round
    return global_model
