"""Apples-to-apples validation follow-up: worst-GROUP accuracy on the
shared (alpha-independent) held-out test set, so the statistic (a minimum
over fixed groups) matches the decomposition residual's statistic (a
minimum over clients), while the test set itself stays alpha-independent
(so any decline is training-driven, not composition).

Group partition: for each (seed, fold), a fixed seeded random assignment
of that fold's 184 test rows into 4 groups -- constructed once per
(seed, fold) and reused identically for both the alpha=100 and alpha=0.1
trained models. Not the alpha-dependent Dirichlet client assignment.

LR, MLP: full 10-seed x 5-fold reproduction (deterministic, cheap --
reproducing the exact already-existing trained models, not new training).
VQC: only 2 replicates available (seed 0 and 5, fold 0) -- the earlier
angle-capture work saved trained parameters for a small sample, not the
full 50-replicate sweep. Retraining VQC to n=50 would cost ~14 hours and
was not authorized here -- explicitly using the smaller sample instead,
flagged as lower-powered in the report.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from composition_controlled_eval import cce_fixed_partition, cce_worst_group_accuracy  # noqa: E402
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from federated_loop import run_federated  # noqa: E402
from models import LogisticRegressionModel  # noqa: E402
from models_mlp import MLPModel  # noqa: E402
from models_vqc import VQCModel  # noqa: E402

SEEDS = list(range(10))
FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 5
N_GROUPS = 4
GROUP_BASE_SEED = 100_000  # this script's original fixed_group_assignment base -- preserved exactly


def fixed_group_assignment(n_rows: int, seed: int, fold: int) -> np.ndarray:
    """Deterministic, alpha-independent group assignment -- same seed
    formula used for both the alpha=100 and alpha=0.1 evaluation of a
    given (seed, fold), so the partition never varies with alpha. Now a
    thin wrapper over the CCE module (composition_controlled_eval.py),
    preserving this script's original base_seed exactly."""
    return cce_fixed_partition(n_rows, seed, fold, n_groups=N_GROUPS, base_seed=GROUP_BASE_SEED)


def worst_group_accuracy(model, X, y, groups) -> float:
    return cce_worst_group_accuracy(model, X, y, groups, n_groups=N_GROUPS, metric="accuracy")


def lr_mlp_worst_group_decline(model_factory):
    pool = load_pool()
    per_replicate = {100: [], 0.1: []}
    for seed in SEEDS:
        assign_100 = client_assignment(pool, 100, seed)
        assign_01 = client_assignment(pool, 0.1, seed)
        for fold_id, df_train_fold, df_test_fold in cv_folds(pool, seed):
            X_train, y_train, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)
            groups = fixed_group_assignment(len(y_test), seed, fold_id)

            for cond, assign in [(100, assign_100), (0.1, assign_01)]:
                client_train = split_by_client(X_train, y_train, assign, df_train_fold)
                client_data = [client_train[c] for c in sorted(client_train)]
                model = run_federated(
                    lambda: model_factory(seed), fedavg, client_data,
                    rounds=FEDERATED_ROUNDS, local_epochs=LOCAL_EPOCHS,
                )
                wg = worst_group_accuracy(model, X_test, y_test, groups)
                per_replicate[cond].append((seed, fold_id, wg))
    return per_replicate


def vqc_worst_group_decline():
    pool = load_pool()
    results = {100: [], 0.1: []}
    for seed in [0, 5]:
        fold_id = 0
        folds = list(cv_folds(pool, seed))
        _, df_train_fold, df_test_fold = folds[fold_id]
        _, _, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)
        groups = fixed_group_assignment(len(y_test), seed, fold_id)

        for cond_raw, cond in [("100", 100), ("0.1", 0.1)]:
            data = np.load(f"results/angle_capture/arm4_{cond_raw}_{seed}_{fold_id}.npz")
            model = VQCModel(seed=seed)
            model.set_params(data["global_params"][-1])
            wg = worst_group_accuracy(model, X_test, y_test, groups)
            results[cond].append((seed, fold_id, wg))
    return results


if __name__ == "__main__":
    import time

    start = time.perf_counter()
    print("=== LR ===")
    lr = lr_mlp_worst_group_decline(lambda seed: LogisticRegressionModel(6, seed=seed))
    a100 = np.mean([v for _, _, v in lr[100]])
    a01 = np.mean([v for _, _, v in lr[0.1]])
    print(f"worst-group a100={a100:.4f} a0.1={a01:.4f} decline={100*(a100-a01):.2f}pp ({time.perf_counter()-start:.1f}s)")

    print("=== MLP ===")
    mlp = lr_mlp_worst_group_decline(lambda seed: MLPModel(6, seed=seed))
    a100 = np.mean([v for _, _, v in mlp[100]])
    a01 = np.mean([v for _, _, v in mlp[0.1]])
    print(f"worst-group a100={a100:.4f} a0.1={a01:.4f} decline={100*(a100-a01):.2f}pp ({time.perf_counter()-start:.1f}s)")

    print("=== VQC (n=2 only, see note) ===")
    vqc = vqc_worst_group_decline()
    a100 = np.mean([v for _, _, v in vqc[100]])
    a01 = np.mean([v for _, _, v in vqc[0.1]])
    print(f"worst-group a100={a100:.4f} a0.1={a01:.4f} decline={100*(a100-a01):.2f}pp, n={len(vqc[100])} ({time.perf_counter()-start:.1f}s)")
    print("per-replicate VQC:", vqc)

    # save raw per-replicate numbers for paired stats
    import json
    with open("results/shared_test_worst_group.json", "w") as f:
        json.dump({
            "LR": {str(k): v for k, v in lr.items()},
            "MLP": {str(k): v for k, v in mlp.items()},
            "VQC": {str(k): v for k, v in vqc.items()},
        }, f, indent=2)
    print("\nwrote results/shared_test_worst_group.json")
