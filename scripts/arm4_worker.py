"""One Arm 4 (VQC + FedAvg) replicate: a single (condition, seed, fold).
Invoked as a subprocess so multiple replicates can run concurrently
(validated 4-way parallelism, D-020) without corrupting a shared CSV --
each worker writes its own small output file under
results/arm4_partial/, merged afterward by arm4_orchestrator.py.

Resumable: if this replicate's output file already exists, does nothing
and exits immediately. This is the actual resume mechanism -- the
orchestrator just checks file existence before launching a worker.

Usage: arm4_worker.py <condition> <seed> <fold>
  condition: "100", "1.0", "0.5", "0.1", or "natural"
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
OUT_DIR = "results/arm4_partial"


def evaluate(model, X, y) -> dict:
    if len(y) == 0:
        return {"n": 0, "accuracy": float("nan"), "f1": float("nan"), "auroc": float("nan")}
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    auroc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    return {"n": len(y), "accuracy": round(acc, 4), "f1": round(f1, 4), "auroc": round(auroc, 4) if not np.isnan(auroc) else float("nan")}


def main():
    condition_raw, seed, fold = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
    condition = condition_raw if condition_raw == "natural" else float(condition_raw)
    out_path = f"{OUT_DIR}/{condition_raw}_{seed}_{fold}.json"

    if os.path.exists(out_path):
        print(f"skip {condition_raw} seed={seed} fold={fold} -- already done")
        return

    os.makedirs(OUT_DIR, exist_ok=True)
    pool = load_pool()
    assign = client_assignment(pool, condition, seed)

    folds = list(cv_folds(pool, seed))
    fold_id, df_train_fold, df_test_fold = folds[fold]
    X_train, y_train, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)

    client_train = split_by_client(X_train, y_train, assign, df_train_fold)
    client_test = split_by_client(X_test, y_test, assign, df_test_fold)
    client_ids = sorted(client_train.keys())
    client_data = [client_train[c] for c in client_ids]

    start = time.perf_counter()
    model, divergence = run_federated(
        lambda: VQCModel(seed=seed), fedavg, client_data,
        rounds=FEDERATED_ROUNDS, local_epochs=LOCAL_EPOCHS, track_divergence=True,
    )
    wall = time.perf_counter() - start

    global_metrics = evaluate(model, X_test, y_test)
    per_client = {str(c): evaluate(model, *client_test[c]) for c in client_test}

    result = {
        "condition": condition_raw, "seed": seed, "fold": fold,
        "wall_clock_sec": round(wall, 2),
        "global": global_metrics,
        "per_client": per_client,
        "divergence": [round(d, 6) for d in divergence],
    }
    with open(out_path, "w") as f:
        json.dump(result, f)
    print(f"done {condition_raw} seed={seed} fold={fold}: global={global_metrics} ({wall:.1f}s)")


if __name__ == "__main__":
    main()
