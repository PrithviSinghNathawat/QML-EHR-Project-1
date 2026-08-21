"""Arm 3 (FedProx), MLP only. Same protocol as every other arm: 20 rounds,
E=5, 5-fold CV, 10 seeds, identical partitions and seeds. mu in
{0.01, 0.05, 0.1}. Classical (MLP), fast -- no parallelism needed, matches
scripts/run_diagnostic.py's pattern.

Scope: MLP only. LR and VQC showed essentially no genuine training-effect
residual to recover (D-050, D-051/D-049) -- FedProx's proximal term has
nothing to act on for either. Only MLP (+13.47pp residual, D-047) has
real drift damage a proximal term could plausibly restrain.
"""
import csv
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from federated_loop import run_federated  # noqa: E402
from models_mlp import FedProxMLPModel  # noqa: E402

SEEDS = list(range(10))
CONDITIONS = [100, 1.0, 0.5, 0.1, "natural"]
MUS = [0.01, 0.05, 0.1]
FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 5

RESULTS_CSV = "results/arm3_diagnostic_results.csv"
DIVERGENCE_CSV = "results/arm3_diagnostic_divergence.csv"
RUNS_ARM3_CSV = "results/runs_arm3.csv"

RESULTS_FIELDS = ["seed", "fold", "model", "arm", "condition", "client", "n", "accuracy", "f1", "auroc"]
DIVERGENCE_FIELDS = ["seed", "fold", "model", "condition", "round", "mean_pairwise_l2"]
RUNS_FIELDS = ["arm", "alpha", "seed", "f1", "auroc", "accuracy", "rounds_to_converge", "wall_clock_sec", "n_qubits", "circuit_depth"]


def evaluate(model, X, y) -> dict:
    if len(y) == 0:
        return {"n": 0, "accuracy": float("nan"), "f1": float("nan"), "auroc": float("nan")}
    proba = model.predict_proba(X)
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, zero_division=0)
    auroc = roc_auc_score(y, proba) if len(np.unique(y)) > 1 else float("nan")
    return {"n": len(y), "accuracy": round(acc, 4), "f1": round(f1, 4), "auroc": round(auroc, 4) if not np.isnan(auroc) else float("nan")}


if __name__ == "__main__":
    import time

    pool = load_pool()
    rf = open(RESULTS_CSV, "w", newline="")
    rw = csv.DictWriter(rf, fieldnames=RESULTS_FIELDS)
    rw.writeheader()
    dvf = open(DIVERGENCE_CSV, "w", newline="")
    dvw = csv.DictWriter(dvf, fieldnames=DIVERGENCE_FIELDS)
    dvw.writeheader()
    runsf = open(RUNS_ARM3_CSV, "w", newline="")
    runsw = csv.DictWriter(runsf, fieldnames=RUNS_FIELDS)
    runsw.writeheader()

    start = time.perf_counter()
    for mu in MUS:
        model_name = f"MLP-FedProx-mu{mu}"
        for seed in SEEDS:
            assignments = {cond: client_assignment(pool, cond, seed) for cond in CONDITIONS}
            for fold_id, df_train_fold, df_test_fold in cv_folds(pool, seed):
                X_train, y_train, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)
                for cond in CONDITIONS:
                    assign = assignments[cond]
                    client_train = split_by_client(X_train, y_train, assign, df_train_fold)
                    client_test = split_by_client(X_test, y_test, assign, df_test_fold)
                    client_ids = sorted(client_train.keys())
                    client_data = [client_train[c] for c in client_ids]

                    t0 = time.perf_counter()
                    model, divergence = run_federated(
                        lambda: FedProxMLPModel(6, mu=mu, seed=seed), fedavg, client_data,
                        rounds=FEDERATED_ROUNDS, local_epochs=LOCAL_EPOCHS, track_divergence=True,
                    )
                    wall = time.perf_counter() - t0

                    g = evaluate(model, X_test, y_test)
                    rw.writerow({"seed": seed, "fold": fold_id, "model": model_name, "arm": "arm3",
                                 "condition": cond, "client": "global", **g})
                    for c, (Xc, yc) in client_test.items():
                        m = evaluate(model, Xc, yc)
                        rw.writerow({"seed": seed, "fold": fold_id, "model": model_name, "arm": "arm3",
                                     "condition": cond, "client": c, **m})
                    for r, d in enumerate(divergence):
                        dvw.writerow({"seed": seed, "fold": fold_id, "model": model_name,
                                      "condition": cond, "round": r, "mean_pairwise_l2": round(d, 6)})
                    runsw.writerow({
                        "arm": "arm3", "alpha": cond if cond != "natural" else "", "seed": seed,
                        "f1": g["f1"], "auroc": g["auroc"], "accuracy": g["accuracy"],
                        "rounds_to_converge": FEDERATED_ROUNDS, "wall_clock_sec": round(wall, 4),
                        "n_qubits": "", "circuit_depth": "",
                    })
        print(f"mu={mu} done ({time.perf_counter()-start:.1f}s elapsed)")

    rf.close()
    dvf.close()
    runsf.close()
    print(f"\ntotal: {time.perf_counter()-start:.1f}s")
    print(f"wrote {RESULTS_CSV}, {DIVERGENCE_CSV}, {RUNS_ARM3_CSV}")
