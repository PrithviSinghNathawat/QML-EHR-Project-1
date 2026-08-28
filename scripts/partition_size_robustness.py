"""P-019: minimum-partition-size robustness check on the composition-share
headline. The K=4 worst case has n=2 test samples (dataset 2, alpha=0.1),
where worst-client accuracy can only take 3 values (0, 0.5, 1) -- a
near-degenerate statistic that could be inflating or deflating the
composition share by chance. This re-analyses ALREADY-COMPUTED per-client
accuracy/size data (no retraining) wherever it exists, and only re-runs the
deterministic training pipeline where per-client detail was not persisted
the first time (dataset 1 LR/MLP composition-only curve; dataset 2 LR/MLP
both curves) -- same seeds, same protocol, reproducing the exact
already-published aggregate numbers as a correctness check before applying
anything new.

No new training regime, no hyperparameter change: this is instrumentation
added to an unchanged, already-frozen recipe, consistent with this
project's established "deterministic reproduction is not retraining"
precedent (P-006/P-007).

Filter: excludes a (seed, fold, condition, client) cell from the
worst-client minimum if its test-partition size n < threshold. A
replicate with zero surviving clients contributes no value (NaN, dropped
from the mean) rather than a fabricated one.
"""
import json
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from dataset2_cv_protocol import client_assignment as d2_client_assignment  # noqa: E402
from dataset2_cv_protocol import cv_folds as d2_cv_folds  # noqa: E402
from dataset2_cv_protocol import fit_transform_fold as d2_fit_transform_fold  # noqa: E402
from dataset2_cv_protocol import load_pool as d2_load_pool  # noqa: E402
from dataset2_cv_protocol import split_by_client as d2_split_by_client  # noqa: E402
from dataset2_decomposition import ALPHAS_BY_K, LOCAL_EPOCHS as D2_LOCAL_EPOCHS, ROUNDS as D2_ROUNDS, _bal_acc  # noqa: E402
from federated_loop import run_federated  # noqa: E402
from models import LogisticRegressionModel  # noqa: E402
from models_mlp import MLPModel  # noqa: E402
from models_weighted import WeightedLogisticRegressionModel, WeightedMLPModel  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

THRESHOLDS = [0, 5, 10, 15, 20]
D1_SEEDS = list(range(10))
D1_CONDITIONS_HEADLINE = ("100", "0.1")  # dataset 1's headline pair, matches SS V-A


def filtered_min(records, threshold):
    """records: list of (n, acc). Returns min acc among n>=threshold, or NaN."""
    kept = [acc for n, acc in records if n >= threshold]
    return min(kept) if kept else float("nan")


# ---------------------------------------------------------------------------
# Dataset 1: LR/MLP observed curve -- from results/diagnostic_results.csv,
# arm2 (FedAvg), already has per-client n+accuracy. No retraining.
# ---------------------------------------------------------------------------
def d1_observed_records(model_name):
    df = pd.read_csv("results/diagnostic_results.csv")
    sub = df[(df["model"] == model_name) & (df["arm"] == "arm2") & (df["client"] != "global")]
    out = {}
    for (seed, fold, cond), g in sub.groupby(["seed", "fold", "condition"]):
        out.setdefault(str(cond), {})[(seed, fold)] = list(zip(g["n"], g["accuracy"]))
    return out


# ---------------------------------------------------------------------------
# Dataset 1: VQC observed -- results/arm4_diagnostic_results.csv. No retraining.
# Dataset 1: VQC composition-only -- results/vqc_composition_partial/*.json. No retraining.
# ---------------------------------------------------------------------------
def d1_vqc_observed_records():
    df = pd.read_csv("results/arm4_diagnostic_results.csv")
    sub = df[(df["arm"] == "arm4") & (df["client"] != "global")]
    out = {}
    for (seed, fold, cond), g in sub.groupby(["seed", "fold", "condition"]):
        out.setdefault(str(cond), {})[(seed, fold)] = list(zip(g["n"], g["accuracy"]))
    return out


def d1_vqc_composition_records():
    out = {}
    for seed in range(10):
        for fold in range(5):
            with open(f"results/vqc_composition_partial/{seed}_{fold}.json") as f:
                data = json.load(f)
            for cond, clients in data["per_condition"].items():
                recs = [(c["n"], c["accuracy"]) for c in clients.values()]
                out.setdefault(cond, {})[(seed, fold)] = recs
    return out


# ---------------------------------------------------------------------------
# Dataset 1: LR/MLP composition-only curve -- NOT persisted per-client the
# first time (composition_decomposition.py discarded it after taking min).
# Deterministic reproduction: same seeds, same fed_train_100 recipe.
# ---------------------------------------------------------------------------
def d1_lr_mlp_composition_records(model_factory):
    pool = load_pool()
    conditions = [100, 1.0, 0.5, 0.1, "natural"]
    out = {str(c): {} for c in conditions}
    for seed in D1_SEEDS:
        assignments = {cond: client_assignment(pool, cond, seed) for cond in conditions}
        for fold_id, df_train_fold, df_test_fold in cv_folds(pool, seed):
            X_train, y_train, X_test, y_test = fit_transform_fold(df_train_fold, df_test_fold)
            client_train = split_by_client(X_train, y_train, assignments[100], df_train_fold)
            client_data = [client_train[c] for c in sorted(client_train)]
            model = run_federated(lambda: model_factory(seed), fedavg, client_data, rounds=20, local_epochs=5)

            for cond in conditions:
                client_test = split_by_client(X_test, y_test, assignments[cond], df_test_fold)
                recs = []
                for c, (Xc, yc) in client_test.items():
                    if len(yc) == 0:
                        continue
                    pred = (model.predict_proba(Xc) >= 0.5).astype(int)
                    recs.append((len(yc), accuracy_score(yc, pred)))
                out[str(cond)][(seed, fold_id)] = recs
    return out


# ---------------------------------------------------------------------------
# Dataset 2: LR/MLP, K=4 and K=130 -- NOT persisted per-client the first
# time either. Deterministic reproduction of P-014's exact recipe
# (WeightedLogisticRegressionModel/WeightedMLPModel, same seeds, same
# rounds/epochs), capturing per-client (n, balanced-accuracy) instead of
# collapsing straight to the minimum.
# ---------------------------------------------------------------------------
def d2_records(model_name, K, factory_fn):
    alphas = ALPHAS_BY_K[K]
    df, numeric_cols, has_race = d2_load_pool()
    observed = {str(a): {} for a in alphas}
    comp_only = {str(a): {} for a in alphas}

    for seed in range(10):
        assignments = {}
        for a in alphas:
            assign, _ = d2_client_assignment(df, a, seed=seed, n_clients=K)
            assignments[a] = assign

        for fold_id, df_train, df_test in d2_cv_folds(df, seed):
            X_train, y_train, X_test, y_test = d2_fit_transform_fold(df_train, df_test, numeric_cols, has_race)
            n_features = X_train.shape[1]

            trained = {}
            for a in alphas:
                client_train = d2_split_by_client(X_train, y_train, assignments[a], df_train, n_clients=K)
                client_data = [client_train[c] for c in sorted(client_train)]
                model = run_federated(
                    lambda: factory_fn(n_features, seed), fedavg, client_data,
                    rounds=D2_ROUNDS, local_epochs=D2_LOCAL_EPOCHS,
                )
                trained[a] = model
                client_test = d2_split_by_client(X_test, y_test, assignments[a], df_test, n_clients=K)
                recs = []
                for c, (Xc, yc) in client_test.items():
                    if len(yc) == 0:
                        continue
                    pred = (model.predict_proba(Xc) >= 0.5).astype(int)
                    recs.append((len(yc), _bal_acc(yc, pred)))
                observed[str(a)][(seed, fold_id)] = recs

            frozen = trained[100]
            for a in alphas:
                client_test = d2_split_by_client(X_test, y_test, assignments[a], df_test, n_clients=K)
                recs = []
                for c, (Xc, yc) in client_test.items():
                    if len(yc) == 0:
                        continue
                    pred = (frozen.predict_proba(Xc) >= 0.5).astype(int)
                    recs.append((len(yc), _bal_acc(yc, pred)))
                comp_only[str(a)][(seed, fold_id)] = recs
        print(f"  [{model_name} K={K}] seed {seed} done", flush=True)

    return observed, comp_only


# ---------------------------------------------------------------------------
def decline_and_share(observed_records, comp_records, cond_hi, cond_lo, threshold):
    obs_hi = [filtered_min(observed_records[cond_hi][k], threshold) for k in observed_records[cond_hi]]
    obs_lo = [filtered_min(observed_records[cond_lo][k], threshold) for k in observed_records[cond_lo]]
    comp_hi = [filtered_min(comp_records[cond_hi][k], threshold) for k in comp_records[cond_hi]]
    comp_lo = [filtered_min(comp_records[cond_lo][k], threshold) for k in comp_records[cond_lo]]

    obs_decline = 100 * (np.nanmean(obs_hi) - np.nanmean(obs_lo))
    comp_decline = 100 * (np.nanmean(comp_hi) - np.nanmean(comp_lo))
    share = 100 * comp_decline / obs_decline if obs_decline != 0 else float("nan")
    n_valid_hi = int(np.sum(~np.isnan(obs_hi)))
    n_valid_lo = int(np.sum(~np.isnan(obs_lo)))
    return {
        "observed_decline_pp": round(float(obs_decline), 3),
        "composition_only_decline_pp": round(float(comp_decline), 3),
        "composition_share_pct": round(float(share), 1) if not np.isnan(share) else None,
        "n_valid_hi": n_valid_hi, "n_total_hi": len(obs_hi),
        "n_valid_lo": n_valid_lo, "n_total_lo": len(obs_lo),
    }


if __name__ == "__main__":
    results = {}

    print("=== Dataset 1: LR ===", flush=True)
    lr_obs = d1_observed_records("LR")
    lr_comp = d1_lr_mlp_composition_records(lambda seed: LogisticRegressionModel(6, seed=seed))
    results["D1_LR"] = {str(t): decline_and_share(lr_obs, lr_comp, "100", "0.1", t) for t in THRESHOLDS}

    print("=== Dataset 1: MLP ===", flush=True)
    mlp_obs = d1_observed_records("MLP")
    mlp_comp = d1_lr_mlp_composition_records(lambda seed: MLPModel(6, seed=seed))
    results["D1_MLP"] = {str(t): decline_and_share(mlp_obs, mlp_comp, "100", "0.1", t) for t in THRESHOLDS}

    print("=== Dataset 1: VQC ===", flush=True)
    vqc_obs = d1_vqc_observed_records()
    vqc_comp = d1_vqc_composition_records()
    results["D1_VQC"] = {str(t): decline_and_share(vqc_obs, vqc_comp, "100", "0.1", t) for t in THRESHOLDS}

    print("=== Dataset 2: LR K=4 ===", flush=True)
    lr4_obs, lr4_comp = d2_records("LR", 4, lambda n, seed: WeightedLogisticRegressionModel(n, seed=seed))
    results["D2_LR_K4"] = {str(t): decline_and_share(lr4_obs, lr4_comp, "100", "0.1", t) for t in THRESHOLDS}

    print("=== Dataset 2: MLP K=4 ===", flush=True)
    mlp4_obs, mlp4_comp = d2_records("MLP", 4, lambda n, seed: WeightedMLPModel(n, seed=seed))
    results["D2_MLP_K4"] = {str(t): decline_and_share(mlp4_obs, mlp4_comp, "100", "0.1", t) for t in THRESHOLDS}

    print("=== Dataset 2: LR K=130 ===", flush=True)
    lr130_obs, lr130_comp = d2_records("LR", 130, lambda n, seed: WeightedLogisticRegressionModel(n, seed=seed))
    results["D2_LR_K130"] = {str(t): decline_and_share(lr130_obs, lr130_comp, "100", "0.5", t) for t in THRESHOLDS}

    print("=== Dataset 2: MLP K=130 ===", flush=True)
    mlp130_obs, mlp130_comp = d2_records("MLP", 130, lambda n, seed: WeightedMLPModel(n, seed=seed))
    results["D2_MLP_K130"] = {str(t): decline_and_share(mlp130_obs, mlp130_comp, "100", "0.5", t) for t in THRESHOLDS}

    with open("results/partition_size_robustness.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n=== SUMMARY (composition share %, by threshold) ===")
    for key, by_thresh in results.items():
        row = " ".join(f"t={t}:{by_thresh[str(t)]['composition_share_pct']}%" for t in THRESHOLDS)
        print(f"{key}: {row}")
    print("\nwrote results/partition_size_robustness.json")
