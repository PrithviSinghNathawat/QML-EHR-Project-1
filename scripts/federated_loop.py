"""The one shared federated training loop (CLAUDE.md architecture: one loop,
not five scripts). Only ever calls the four Model interface methods and the
aggregator's aggregate() -- never branches on what kind of model or
aggregator it's holding. If a quantum model and a classical model are both
passed in here later, this file should not need to change.
"""


def run_centralized(model_factory, X_train, y_train, epochs: int):
    """Arm 1: no federation, one model trained on the pooled training set."""
    model = model_factory()
    model.fit(X_train, y_train, epochs=epochs)
    return model


def run_federated(model_factory, aggregator, client_data, rounds: int, local_epochs: int):
    """client_data: list of (X_client, y_client) arrays, one per client.
    aggregator: a function matching aggregate(param_vectors, client_sizes).
    Returns the final global model, holding the aggregated parameters."""
    global_model = model_factory()
    global_params = global_model.get_params()

    for _ in range(rounds):
        client_params = []
        client_sizes = []
        for X_c, y_c in client_data:
            local_model = model_factory()
            local_model.set_params(global_params.copy())
            local_model.fit(X_c, y_c, epochs=local_epochs)
            client_params.append(local_model.get_params())
            client_sizes.append(len(X_c))
        global_params = aggregator(client_params, client_sizes)

    global_model.set_params(global_params)
    return global_model
