"""Re-runs a small representative sample of Arm 4 (fedavg) and Arm 5
(circular_mean) replicates, this time saving the raw trained rotation
angles (every client's local parameter vector, every round, before
aggregation) -- needed to verify the D-036 wraparound explanation, which
the original sweep did not save (only evaluation metrics + scalar
divergence were persisted).

Sample: all 5 conditions x seeds {0, 5} x fold 0 = 10 replicates per arm,
20 replicates total. Not the full 250 -- a representative sample chosen
to span the full alpha range (where wraparound, if it happens, would be
most likely at the extreme end) and two different seeds.

Usage: capture_angles_worker.py <arm> <condition> <seed> <fold>
  arm: "4" (fedavg) or "5" (circular_mean)
"""
import os
import sys

import numpy as np

sys.path.insert(0, "scripts")
from aggregators import circular_mean, fedavg  # noqa: E402
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from models_vqc import VQCModel  # noqa: E402

FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 5
OUT_DIR = "results/angle_capture"


def main():
    arm, condition_raw, seed, fold = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    condition = condition_raw if condition_raw == "natural" else float(condition_raw)
    aggregator = fedavg if arm == "4" else circular_mean
    out_path = f"{OUT_DIR}/arm{arm}_{condition_raw}_{seed}_{fold}.npz"

    if os.path.exists(out_path):
        print(f"skip arm{arm} {condition_raw} seed={seed} fold={fold} -- already done")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    pool = load_pool()
    assign = client_assignment(pool, condition, seed)
    folds = list(cv_folds(pool, seed))
    fold_id, df_train_fold, df_test_fold = folds[fold]
    X_train, y_train, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)
    client_train = split_by_client(X_train, y_train, assign, df_train_fold)
    client_ids = sorted(client_train.keys())
    client_data = [client_train[c] for c in client_ids]

    # Reimplements the run_federated loop body (not calling it) purely to
    # capture per-round, per-client local parameter vectors -- deliberately
    # NOT amending federated_loop.py's frozen interface a second time for
    # a one-off diagnostic need.
    global_model = VQCModel(seed=seed)
    global_params = global_model.get_params()
    all_client_params = []  # (round, client) -> 18-vector
    all_global_params = [global_params.copy()]

    for r in range(FEDERATED_ROUNDS):
        round_client_params = []
        client_sizes = []
        for X_c, y_c in client_data:
            local_model = VQCModel(seed=seed)
            local_model.set_params(global_params.copy())
            local_model.fit(X_c, y_c, epochs=LOCAL_EPOCHS)
            round_client_params.append(local_model.get_params())
            client_sizes.append(len(X_c))
        all_client_params.append(np.stack(round_client_params))
        global_params = aggregator(round_client_params, client_sizes)
        all_global_params.append(global_params.copy())

    client_params_arr = np.stack(all_client_params)  # (rounds, n_clients, 18)
    global_params_arr = np.stack(all_global_params)  # (rounds+1, 18)

    np.savez(
        out_path,
        client_params=client_params_arr,
        global_params=global_params_arr,
        condition=condition_raw, seed=seed, fold=fold, arm=arm,
    )
    print(
        f"done arm{arm} {condition_raw} seed={seed} fold={fold}: "
        f"client_params range [{client_params_arr.min():.4f}, {client_params_arr.max():.4f}], "
        f"global_params range [{global_params_arr.min():.4f}, {global_params_arr.max():.4f}]"
    )


if __name__ == "__main__":
    main()
