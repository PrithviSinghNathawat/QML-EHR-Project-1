"""Composition-vs-training decomposition for Arm 3 (FedProx MLP), same
method as scripts/composition_decomposition.py: train once per (seed,
fold) at alpha=100 for each mu, evaluate that fixed model against every
condition's test-slice composition. The gap between this and the observed
(per-condition-trained) curve is the residual training effect -- the
quantity that answers whether FedProx recovers MLP's genuine drift damage.
"""
import sys

import numpy as np
from sklearn.metrics import accuracy_score

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from federated_loop import run_federated  # noqa: E402
from models_mlp import FedProxMLPModel  # noqa: E402

CONDITIONS = [100, 1.0, 0.5, 0.1, "natural"]
SEEDS = list(range(10))
MUS = [0.01, 0.05, 0.1]
FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 5


def evaluate(model, X, y) -> float:
    if len(y) == 0:
        return float("nan")
    pred = (model.predict_proba(X) >= 0.5).astype(int)
    return accuracy_score(y, pred)


if __name__ == "__main__":
    pool = load_pool()

    for mu in MUS:
        results = {cond: [] for cond in CONDITIONS}
        for seed in SEEDS:
            assignments = {cond: client_assignment(pool, cond, seed) for cond in CONDITIONS}
            assign_100 = assignments[100]
            for fold_id, df_train_fold, df_test_fold in cv_folds(pool, seed):
                X_train, y_train, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)
                client_train = split_by_client(X_train, y_train, assign_100, df_train_fold)
                client_data = [client_train[c] for c in sorted(client_train)]

                model = run_federated(
                    lambda: FedProxMLPModel(6, mu=mu, seed=seed), fedavg, client_data,
                    rounds=FEDERATED_ROUNDS, local_epochs=LOCAL_EPOCHS,
                )

                for cond in CONDITIONS:
                    client_test = split_by_client(X_test, y_test, assignments[cond], df_test_fold)
                    accs = [evaluate(model, Xc, yc) for c, (Xc, yc) in client_test.items() if len(yc) > 0]
                    if accs:
                        results[cond].append(min(accs))

        print(f"=== mu={mu}: composition-only worst-client (alpha=100-trained model) ===")
        for cond in CONDITIONS:
            v = results[cond]
            print(f"  {cond}: mean={np.mean(v):.4f} std={np.std(v):.4f} n={len(v)}")
