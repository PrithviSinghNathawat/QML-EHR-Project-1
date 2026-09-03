"""Presentation demo: a short federated MLP training run with visibly
descending loss, small enough to run live in under a minute. Built entirely
from already-tested pipeline pieces (aggregators.fedavg, federated_loop.
run_federated, models_mlp.MLPModel, partitioner.dirichlet_partition,
preprocessing.get_preprocessed) -- not a new training path, just a thin,
printable wrapper around the real one for a live audience.

MLPModel/LogisticRegressionModel have no built-in .loss() (unlike VQCModel,
scripts/sanity_check_vqc.py) -- binary cross-entropy is computed here
directly from predict_proba, the standard definition, nothing bespoke.
"""
import sys
import time

import numpy as np

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from federated_loop import run_federated  # noqa: E402
from models_mlp import MLPModel  # noqa: E402
from partitioner import dirichlet_partition  # noqa: E402
from preprocessing import get_preprocessed  # noqa: E402

SEED = 0
ALPHA = 100
N_CLIENTS = 4
N_ROUNDS = 20
LOCAL_EPOCHS = 5


def bce(model, X, y) -> float:
    p = np.clip(model.predict_proba(X), 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


if __name__ == "__main__":
    df_train, df_test, X_train, y_train, X_test, y_test = get_preprocessed(SEED)
    client_idx, n_attempts = dirichlet_partition(df_train, ALPHA, n_clients=N_CLIENTS, seed=SEED)
    pos = df_train.index.get_indexer
    client_data = [(X_train[pos(idx)], y_train[pos(idx)]) for idx in client_idx]
    print(f"client sizes: {[len(y) for _, y in client_data]} (guard attempts: {n_attempts})")

    global_params = MLPModel(6, seed=SEED).get_params()
    losses = []
    start = time.perf_counter()
    for r in range(N_ROUNDS):
        def factory(gp=global_params):
            m = MLPModel(6, seed=SEED)
            m.set_params(gp)
            return m

        model = run_federated(factory, fedavg, client_data, rounds=1, local_epochs=LOCAL_EPOCHS)
        global_params = model.get_params()
        loss = bce(model, X_train, y_train)
        losses.append(loss)
        print(f"round {r + 1:2d}/{N_ROUNDS}  loss={loss:.4f}  ({time.perf_counter() - start:.1f}s elapsed)")

    total = time.perf_counter() - start
    print(f"\ntotal: {total:.1f}s")
    print(f"loss: round 1 = {losses[0]:.4f}, round {N_ROUNDS} = {losses[-1]:.4f}, "
          f"decrease = {losses[0] - losses[-1]:.4f}")
