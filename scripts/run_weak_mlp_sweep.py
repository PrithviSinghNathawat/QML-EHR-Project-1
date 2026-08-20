"""Capacity control for Arm 4 (Prithvi's follow-up, 2026-08-20): a
deliberately weakened classical MLP whose alpha=100 worst-client accuracy
matches the VQC's as closely as achievable, so the VQC-vs-MLP comparison
in docs/arm4_report.md isn't confounded by the VQC simply being a weaker
learner (weaker learners are closer to constant predictors and degrade
less by construction).

Calibration (done once, against alpha=100 only, seed=0, reported in
docs/decisions.md -- NOT re-tuned after seeing the full sweep):
hidden=1 (9 params, down from 17), FEDERATED_ROUNDS=4 (down from 20),
LOCAL_EPOCHS=1 (down from 5). alpha=100 worst-client, seed=0 only: 0.6483
vs VQC's 0.6488 (full 50-replicate mean) -- 0.05pp gap. This script then
runs the identical full protocol (5-fold CV x 10 seeds x 5 conditions) and
reports whatever comes out, unmodified from here.
"""
import sys

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from federated_loop import run_federated  # noqa: E402
from models_mlp import MLPModel  # noqa: E402

SEEDS = list(range(10))
CONDITIONS = [100, 1.0, 0.5, 0.1, "natural"]
HIDDEN = 1
FEDERATED_ROUNDS = 4
LOCAL_EPOCHS = 1
RESULTS_CSV = "results/weak_mlp_diagnostic_results.csv"
DIVERGENCE_CSV = "results/weak_mlp_diagnostic_divergence.csv"


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
    import csv

    pool = load_pool()
    rf = open(RESULTS_CSV, "w", newline="")
    rw = csv.DictWriter(rf, fieldnames=["seed", "fold", "model", "arm", "condition", "client", "n", "accuracy", "f1", "auroc"])
    rw.writeheader()
    dvf = open(DIVERGENCE_CSV, "w", newline="")
    dvw = csv.DictWriter(dvf, fieldnames=["seed", "fold", "model", "condition", "round", "mean_pairwise_l2"])
    dvw.writeheader()

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

                model, divergence = run_federated(
                    lambda: MLPModel(6, hidden=HIDDEN, seed=seed), fedavg, client_data,
                    rounds=FEDERATED_ROUNDS, local_epochs=LOCAL_EPOCHS, track_divergence=True,
                )
                g = evaluate(model, X_test, y_test)
                rw.writerow({"seed": seed, "fold": fold_id, "model": "weak-MLP", "arm": "arm2",
                             "condition": cond, "client": "global", **g})
                for c, (Xc, yc) in client_test.items():
                    m = evaluate(model, Xc, yc)
                    rw.writerow({"seed": seed, "fold": fold_id, "model": "weak-MLP", "arm": "arm2",
                                 "condition": cond, "client": c, **m})
                for r, d in enumerate(divergence):
                    dvw.writerow({"seed": seed, "fold": fold_id, "model": "weak-MLP",
                                  "condition": cond, "round": r, "mean_pairwise_l2": round(d, 6)})
        print(f"seed {seed} done")

    rf.close()
    dvf.close()
    print(f"wrote {RESULTS_CSV}, {DIVERGENCE_CSV}")
