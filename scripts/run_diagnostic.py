"""Diagnostic session driver (docs/decisions.md, 2026-08-18). Classical
arms only -- no Arm 4/5. Does NOT tune anything toward a decline; reports
whatever the sweep produces.

For every seed x fold x model x condition:
  - Arm 1 (centralized) trained once per (seed, fold, model) -- condition-
    independent -- then evaluated globally and per-client (per-client
    slicing IS condition-dependent, since client assignment depends on
    condition).
  - Arm 2 (FedAvg) trained once per (seed, fold, model, condition),
    evaluated globally and per-client, with per-round client-parameter
    divergence tracked.

Writes:
  results/diagnostic_results.csv   -- long format, one row per
                                       (seed, fold, model, arm, condition, client)
  results/diagnostic_divergence.csv -- one row per
                                        (seed, fold, model, condition, round)
"""
import csv
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from federated_loop import run_centralized, run_federated  # noqa: E402
from models import LogisticRegressionModel  # noqa: E402
from models_mlp import MLPModel  # noqa: E402

SEEDS = list(range(10))
CONDITIONS = [100, 1.0, 0.5, 0.1, "natural"]
CENTRALIZED_EPOCHS = 200
FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 5

MODEL_FACTORIES = {
    "LR": lambda seed: LogisticRegressionModel(6, seed=seed),
    "MLP": lambda seed: MLPModel(6, seed=seed),
}

RESULTS_CSV = "results/diagnostic_results.csv"
DIVERGENCE_CSV = "results/diagnostic_divergence.csv"
RESULTS_FIELDS = [
    "seed", "fold", "model", "arm", "condition", "client",
    "n", "accuracy", "f1", "auroc",
]
DIVERGENCE_FIELDS = ["seed", "fold", "model", "condition", "round", "mean_pairwise_l2"]


def evaluate(model, X, y) -> dict:
    if len(y) == 0:
        return {"n": 0, "accuracy": float("nan"), "f1": float("nan"), "auroc": float("nan")}
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    auroc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    return {"n": len(y), "accuracy": round(acc, 4), "f1": round(f1, 4), "auroc": round(auroc, 4) if not np.isnan(auroc) else float("nan")}


def open_writers():
    os.makedirs("results", exist_ok=True)
    rf = open(RESULTS_CSV, "w", newline="")
    rw = csv.DictWriter(rf, fieldnames=RESULTS_FIELDS)
    rw.writeheader()
    df_ = open(DIVERGENCE_CSV, "w", newline="")
    dw = csv.DictWriter(df_, fieldnames=DIVERGENCE_FIELDS)
    dw.writeheader()
    return rf, rw, df_, dw


def record(writer, **kwargs):
    writer.writerow({k: kwargs.get(k, "") for k in RESULTS_FIELDS})


if __name__ == "__main__":
    pool = load_pool()
    rf, rw, dvf, dvw = open_writers()
    start = time.perf_counter()
    n_arm1_trainings = 0
    n_arm2_trainings = 0

    for seed in SEEDS:
        assignments = {cond: client_assignment(pool, cond, seed) for cond in CONDITIONS}
        for fold_id, df_train_fold, df_test_fold in cv_folds(pool, seed):
            X_train, y_train, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)

            for model_name, factory in MODEL_FACTORIES.items():
                # Arm 1: centralized, condition-independent training
                arm1_model = run_centralized(lambda: factory(seed), X_train, y_train, epochs=CENTRALIZED_EPOCHS)
                n_arm1_trainings += 1
                g = evaluate(arm1_model, X_test, y_test)
                record(rw, seed=seed, fold=fold_id, model=model_name, arm="arm1", condition="", client="global", **g)
                for cond in CONDITIONS:
                    client_test = split_by_client(X_test, y_test, assignments[cond], df_test_fold)
                    for c, (Xc, yc) in client_test.items():
                        m = evaluate(arm1_model, Xc, yc)
                        record(rw, seed=seed, fold=fold_id, model=model_name, arm="arm1", condition=cond, client=c, **m)

                # Arm 2: federated, condition-dependent training
                for cond in CONDITIONS:
                    assign = assignments[cond]
                    client_train = split_by_client(X_train, y_train, assign, df_train_fold)
                    client_test = split_by_client(X_test, y_test, assign, df_test_fold)
                    client_ids = sorted(client_train.keys())
                    client_data = [client_train[c] for c in client_ids]

                    model, divergence = run_federated(
                        lambda: factory(seed), fedavg, client_data,
                        rounds=FEDERATED_ROUNDS, local_epochs=LOCAL_EPOCHS, track_divergence=True,
                    )
                    n_arm2_trainings += 1
                    g = evaluate(model, X_test, y_test)
                    record(rw, seed=seed, fold=fold_id, model=model_name, arm="arm2", condition=cond, client="global", **g)
                    for c, (Xc, yc) in client_test.items():
                        m = evaluate(model, Xc, yc)
                        record(rw, seed=seed, fold=fold_id, model=model_name, arm="arm2", condition=cond, client=c, **m)
                    for r, d in enumerate(divergence):
                        dvw.writerow({"seed": seed, "fold": fold_id, "model": model_name, "condition": cond, "round": r, "mean_pairwise_l2": round(d, 6)})
        print(f"seed {seed} done ({time.perf_counter() - start:.1f}s elapsed)")

    rf.close()
    dvf.close()
    total = time.perf_counter() - start
    print(f"\ndone in {total:.1f}s -- {n_arm1_trainings} Arm1 trainings, {n_arm2_trainings} Arm2 trainings")
    print(f"wrote {RESULTS_CSV}, {DIVERGENCE_CSV}")
