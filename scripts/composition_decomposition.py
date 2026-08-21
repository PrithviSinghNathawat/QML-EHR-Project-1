"""Decomposes worst-client movement into a composition effect (test-slice
skew changing with alpha) and a training effect (the model itself
changing with alpha), per Prithvi's 2026-08-20 follow-up.

Method: train once per (seed, fold) at alpha=100 (fixed model). Evaluate
that SAME fixed model against the per-client test slices defined by every
condition's client assignment (100, 1.0, 0.5, 0.1, natural), applied to
the same fold's held-out test set. Any worst-client movement across
conditions here is pure composition, since the model never changes.
Compare to the observed curve (the real per-condition-trained models,
already in results/*_diagnostic_results.csv) to get the residual training
effect.
"""
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, "scripts")
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from federated_loop import run_centralized  # noqa: E402

CONDITIONS = [100, 1.0, 0.5, 0.1, "natural"]
SEEDS = list(range(10))


def evaluate(model, X, y) -> float:
    if len(y) == 0:
        return float("nan")
    pred = (model.predict_proba(X) >= 0.5).astype(int)
    return accuracy_score(y, pred)


def composition_only_curve(model_factory, epochs: int, arm_two_train_fn=None):
    """model_factory(seed) -> Model. Trains a model at alpha=100 for
    every (seed, fold) using arm_two_train_fn (defaults to centralized
    training on the pooled alpha=100 training set -- for the classical
    federated arms this is a reasonable fixed-model stand-in since we
    only need ONE trained model per (seed,fold), not a federated one, to
    test the composition question; for VQC the caller passes a federated
    trainer instead). Returns {(model_name, condition): [worst_client_acc, ...]}."""
    pool = load_pool()
    results = {cond: [] for cond in CONDITIONS}

    for seed in SEEDS:
        assignments = {cond: client_assignment(pool, cond, seed) for cond in CONDITIONS}
        for fold_id, df_train_fold, df_test_fold in cv_folds(pool, seed):
            X_train, y_train, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)

            if arm_two_train_fn is None:
                model = run_centralized(lambda: model_factory(seed), X_train, y_train, epochs=epochs)
            else:
                model = arm_two_train_fn(model_factory, seed, X_train, y_train, df_train_fold, assignments[100])

            # evaluate this ONE alpha=100-trained model against every condition's test slicing
            for cond in CONDITIONS:
                client_test = split_by_client(X_test, y_test, assignments[cond], df_test_fold)
                accs = [evaluate(model, Xc, yc) for c, (Xc, yc) in client_test.items() if len(yc) > 0]
                if accs:
                    results[cond].append(min(accs))

    return {cond: (np.mean(v), np.std(v), len(v)) for cond, v in results.items()}


if __name__ == "__main__":
    import argparse

    from aggregators import fedavg
    from federated_loop import run_federated
    from models import LogisticRegressionModel
    from models_mlp import MLPModel

    parser = argparse.ArgumentParser()
    parser.add_argument("model", choices=["LR", "MLP"])
    args = parser.parse_args()

    def fed_train_100(model_factory, seed, X_train, y_train, df_train_fold, assign_100):
        client_train = split_by_client(X_train, y_train, assign_100, df_train_fold)
        client_data = [client_train[c] for c in sorted(client_train)]
        return run_federated(lambda: model_factory(seed), fedavg, client_data, rounds=20, local_epochs=5)

    if args.model == "LR":
        factory = lambda seed: LogisticRegressionModel(6, seed=seed)
    else:
        factory = lambda seed: MLPModel(6, seed=seed)

    result = composition_only_curve(factory, epochs=20, arm_two_train_fn=fed_train_100)
    print(f"=== {args.model}: composition-only worst-client (alpha=100-trained model, varying test-slice condition) ===")
    for cond, (mean, std, n) in result.items():
        print(f"  {cond}: mean={mean:.4f} std={std:.4f} n={n}")
