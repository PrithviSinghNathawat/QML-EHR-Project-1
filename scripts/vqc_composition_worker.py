"""One (seed, fold) replicate of the VQC composition-only decomposition:
train ONCE at alpha=100 (FedAvg, same protocol as Arm 4), then evaluate
that fixed model against the per-client test slices defined by every
condition (100, 1.0, 0.5, 0.1, natural). Any worst-client movement here is
pure evaluation composition, since the model never changes.

50 replicates total (10 seeds x 5 folds) -- much cheaper than the full
250-replicate Arm 4 sweep, since only alpha=100 needs training.

Usage: vqc_composition_worker.py <seed> <fold>
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from federated_loop import run_federated  # noqa: E402
from models_vqc import VQCModel  # noqa: E402

FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 5
CONDITIONS = ["100", "1.0", "0.5", "0.1", "natural"]
OUT_DIR = "results/vqc_composition_partial"


def evaluate(model, X, y) -> dict:
    if len(y) == 0:
        return {"n": 0, "accuracy": float("nan")}
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    return {"n": len(y), "accuracy": round(accuracy_score(y, pred), 4)}


def main():
    seed, fold = int(sys.argv[1]), int(sys.argv[2])
    out_path = f"{OUT_DIR}/{seed}_{fold}.json"
    if os.path.exists(out_path):
        print(f"skip seed={seed} fold={fold} -- already done")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    pool = load_pool()
    assign_100 = client_assignment(pool, 100, seed)
    folds = list(cv_folds(pool, seed))
    fold_id, df_train_fold, df_test_fold = folds[fold]
    X_train, y_train, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)

    client_train = split_by_client(X_train, y_train, assign_100, df_train_fold)
    client_data = [client_train[c] for c in sorted(client_train)]

    start = time.perf_counter()
    model = run_federated(
        lambda: VQCModel(seed=seed), fedavg, client_data,
        rounds=FEDERATED_ROUNDS, local_epochs=LOCAL_EPOCHS,
    )
    wall = time.perf_counter() - start

    per_condition = {}
    for cond_raw in CONDITIONS:
        cond = cond_raw if cond_raw == "natural" else float(cond_raw)
        assign = client_assignment(pool, cond, seed)
        client_test = split_by_client(X_test, y_test, assign, df_test_fold)
        per_client = {str(c): evaluate(model, *client_test[c]) for c in client_test}
        per_condition[cond_raw] = per_client

    result = {"seed": seed, "fold": fold, "wall_clock_sec": round(wall, 2), "per_condition": per_condition}
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"done seed={seed} fold={fold} ({wall:.1f}s)")


if __name__ == "__main__":
    main()
