"""Dataset 2 (Diabetes 130-US Hospitals), minimal scope per P-010: LR and
MLP only, FedAvg only, Dirichlet only, same alpha grid and protocol as
dataset 1 (20 rounds, 5 local epochs, 5-fold stratified CV x 10 seeds),
run at K=4 and K=130 to isolate client count as the variable, with the
composition-vs-training decomposition (D-044/D-047 method) AND the
shared-test training-effect estimate (P-006/P-007 method) both applied, so
the two can be compared per configuration exactly as in P-008.

K=130, alpha=0.1 is EXCLUDED, not attempted: verified empirically
(dataset2_cv_protocol.py's InfeasiblePartition check, 1000+ independent
draws) that every draw produces at least one zero-row client -- not a rare
event fixable with more attempts, a structural property of Dirichlet(0.1)
spread over 130 bins on this dataset. Reported as a finding (P-010), not
worked around by loosening the floor.

Because alpha=0.1 is unavailable at K=130, the primary client-count
comparison (isolating K as the only variable) uses alpha=0.5 -- the most
extreme condition available at BOTH K=4 and K=130 -- as the observed-decline
endpoint, in addition to reporting each K's own full feasible grid.
"""
import json
import sys
import time

import numpy as np
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from composition_controlled_eval import cce_fixed_partition, cce_pooled_accuracy, cce_worst_group_accuracy  # noqa: E402
from dataset2_cv_protocol import (  # noqa: E402
    InfeasiblePartition,
    client_assignment,
    cv_folds,
    fit_transform_fold,
    load_pool,
    split_by_client,
)
from federated_loop import run_federated  # noqa: E402
from models_weighted import WeightedLogisticRegressionModel, WeightedMLPModel  # noqa: E402

SEEDS = list(range(10))
ROUNDS = 20
LOCAL_EPOCHS = 5
N_GROUPS = 4  # fixed shared-test worst-group split, same convention as P-007
GROUP_BASE_SEED = 200_000  # this script's original fixed_group_assignment base -- preserved exactly

ALPHAS_BY_K = {
    4: [100, 1.0, 0.5, 0.1],
    130: [100, 1.0, 0.5],  # 0.1 excluded -- see module docstring
}


def _bal_acc(y_true, y_pred) -> float:
    """balanced_accuracy_score requires both classes present; falls back to
    plain accuracy for a degenerate single-class slice (P-011). Used by
    worst_client_acc below, which scores over this study's real
    (heterogeneity-dependent) Dirichlet clients -- a different quantity
    from the CCE module's fixed-partition evaluation, so it stays local
    rather than moving into composition_controlled_eval.py."""
    if len(np.unique(y_true)) < 2:
        return float((y_true == y_pred).mean())
    return balanced_accuracy_score(y_true, y_pred)


def worst_client_acc(model, X, y, groups: dict) -> float:
    accs = []
    for c, (Xc, yc) in groups.items():
        if len(yc) == 0:
            continue
        pred = (model.predict_proba(Xc) >= 0.5).astype(int)
        accs.append(_bal_acc(yc, pred))
    return min(accs) if accs else float("nan")


def pooled_acc(model, X, y) -> float:
    """Now a thin wrapper over the CCE module (composition_controlled_eval.py)."""
    return cce_pooled_accuracy(model, X, y, metric="balanced")


def fixed_group_assignment(n_rows: int, seed: int, fold: int) -> np.ndarray:
    """Now a thin wrapper over the CCE module, preserving this script's
    original base_seed (200_000) exactly."""
    return cce_fixed_partition(n_rows, seed, fold, n_groups=N_GROUPS, base_seed=GROUP_BASE_SEED)


def worst_group_acc(model, X, y, groups: np.ndarray) -> float:
    return cce_worst_group_accuracy(model, X, y, groups, n_groups=N_GROUPS, metric="balanced")


def run_config(model_name: str, K: int, factory_fn):
    alphas = ALPHAS_BY_K[K]
    df, numeric_cols, has_race = load_pool()

    per_replicate = {
        "observed_worst_client": {a: [] for a in alphas},
        "composition_only_worst_client": {a: [] for a in alphas},
        "shared_pooled": {a: [] for a in alphas},
        "shared_worst_group": {a: [] for a in alphas},
    }
    partition_attempts = {a: [] for a in alphas}

    t_start = time.perf_counter()
    for seed in SEEDS:
        assignments = {}
        for a in alphas:
            assign, n_attempts = client_assignment(df, a, seed=seed, n_clients=K)
            assignments[a] = assign
            partition_attempts[a].append(n_attempts)

        for fold_id, df_train, df_test in cv_folds(df, seed):
            X_train, y_train, X_test, y_test = fit_transform_fold(df_train, df_test, numeric_cols, has_race)
            n_features = X_train.shape[1]
            fixed_groups = fixed_group_assignment(len(y_test), seed, fold_id)

            trained = {}
            for a in alphas:
                client_train = split_by_client(X_train, y_train, assignments[a], df_train, n_clients=K)
                client_data = [client_train[c] for c in sorted(client_train)]
                model = run_federated(
                    lambda: factory_fn(n_features, seed), fedavg, client_data,
                    rounds=ROUNDS, local_epochs=LOCAL_EPOCHS,
                )
                trained[a] = model

                client_test = split_by_client(X_test, y_test, assignments[a], df_test, n_clients=K)
                per_replicate["observed_worst_client"][a].append(worst_client_acc(model, X_test, y_test, client_test))
                per_replicate["shared_pooled"][a].append(pooled_acc(model, X_test, y_test))
                per_replicate["shared_worst_group"][a].append(worst_group_acc(model, X_test, y_test, fixed_groups))

            # composition-only: freeze the alpha=100 model, vary only the test-slice composition
            frozen = trained[100]
            for a in alphas:
                client_test = split_by_client(X_test, y_test, assignments[a], df_test, n_clients=K)
                per_replicate["composition_only_worst_client"][a].append(
                    worst_client_acc(frozen, X_test, y_test, client_test)
                )

        elapsed = time.perf_counter() - t_start
        print(f"  [{model_name} K={K}] seed {seed} done, {elapsed:.1f}s elapsed", flush=True)

    return per_replicate, partition_attempts


def summarize(per_replicate: dict, alphas: list) -> dict:
    def mean_pp(series):
        return 100 * np.mean(series)

    def se_pp(series):
        return 100 * np.std(series, ddof=1) / np.sqrt(len(series))

    out = {}
    for a in alphas:
        if a == 100:
            continue
        obs = np.array(per_replicate["observed_worst_client"][100]) - np.array(per_replicate["observed_worst_client"][a])
        comp = np.array(per_replicate["composition_only_worst_client"][100]) - np.array(per_replicate["composition_only_worst_client"][a])
        shared_pooled = np.array(per_replicate["shared_pooled"][100]) - np.array(per_replicate["shared_pooled"][a])
        shared_wg = np.array(per_replicate["shared_worst_group"][100]) - np.array(per_replicate["shared_worst_group"][a])

        observed_pp = 100 * obs.mean()
        comp_pp = 100 * comp.mean()
        residual_pp = observed_pp - comp_pp
        comp_share = 100 * comp_pp / observed_pp if observed_pp != 0 else float("nan")

        shared_pooled_pp = 100 * shared_pooled.mean()
        shared_pooled_se = se_pp(shared_pooled)
        shared_wg_pp = 100 * shared_wg.mean()
        shared_wg_se = se_pp(shared_wg)

        implied_comp_lo = observed_pp - max(shared_pooled_pp, shared_wg_pp)
        implied_comp_hi = observed_pp - min(shared_pooled_pp, shared_wg_pp)
        interaction_lo = residual_pp - max(shared_pooled_pp, shared_wg_pp)
        interaction_hi = residual_pp - min(shared_pooled_pp, shared_wg_pp)

        out[f"100_to_{a}"] = {
            "observed_decline_pp": round(observed_pp, 2),
            "composition_only_decline_pp": round(comp_pp, 2),
            "composition_share_pct": round(comp_share, 1),
            "decomposition_residual_pp": round(residual_pp, 2),
            "shared_test_pooled_decline_pp": round(shared_pooled_pp, 2),
            "shared_test_pooled_se_pp": round(shared_pooled_se, 2),
            "shared_test_worst_group_decline_pp": round(shared_wg_pp, 2),
            "shared_test_worst_group_se_pp": round(shared_wg_se, 2),
            "implied_composition_range_pp": [round(implied_comp_lo, 2), round(implied_comp_hi, 2)],
            "implied_composition_share_range_pct": [
                round(100 * implied_comp_lo / observed_pp, 1) if observed_pp else None,
                round(100 * implied_comp_hi / observed_pp, 1) if observed_pp else None,
            ],
            "interaction_range_pp": [round(interaction_lo, 2), round(interaction_hi, 2)],
            "n": len(obs),
        }
    return out


if __name__ == "__main__":
    all_results = {}
    for model_name, factory_fn in [
        ("LR", lambda n, seed: WeightedLogisticRegressionModel(n, seed=seed)),
        ("MLP", lambda n, seed: WeightedMLPModel(n, seed=seed)),
    ]:
        for K in [4, 130]:
            print(f"=== {model_name} K={K} ===", flush=True)
            per_replicate, partition_attempts = run_config(model_name, K, factory_fn)
            summary = summarize(per_replicate, ALPHAS_BY_K[K])
            all_results[f"{model_name}_K{K}"] = {
                "summary": summary,
                "partition_attempts": partition_attempts,
                "alphas_run": ALPHAS_BY_K[K],
            }
            print(json.dumps(summary, indent=2))

            with open("results/dataset2_decomposition_weighted_partial.json", "w") as f:
                json.dump(all_results, f, indent=2)

    with open("results/dataset2_decomposition_weighted.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nwrote results/dataset2_decomposition_weighted.json")
