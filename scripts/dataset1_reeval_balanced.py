"""Re-evaluate dataset 1's LR/MLP/VQC composition decomposition and
shared-test estimate under balanced accuracy, alongside plain accuracy, per
P-011. Condition attached to the metric-choice decision: dataset 1 must be
re-evaluated under the same metric as dataset 2 for the cross-dataset
comparison (P-014) to compare like quantities.

LR/MLP: full 10-seed x 5-fold deterministic reproduction (real federated
training via FedAvg, exactly matching Arm 2/D-047's protocol) -- both
metrics computed in the same pass, one training run per replicate, so the
plain-accuracy numbers here can be checked directly against D-047/D-049
before trusting the balanced-accuracy numbers next to them.

VQC: no full-50 raw predictions or parameters are saved (results/arm4_partial
and results/vqc_composition_partial store only accuracy/F1/AUROC summary
scalars) -- reusing the same n=2 saved-parameter sample already used in
P-007 (results/angle_capture/arm4_{100,0.1}_{0,5}_0.npz) rather than an
unauthorized ~14-hour retrain. Flagged as lower-powered, same as P-007.
"""
import sys

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from cv_protocol import client_assignment, cv_folds, fit_transform_fold, load_pool, split_by_client  # noqa: E402
from federated_loop import run_federated  # noqa: E402
from models import LogisticRegressionModel  # noqa: E402
from models_mlp import MLPModel  # noqa: E402
from models_vqc import VQCModel  # noqa: E402

SEEDS = list(range(10))
ALPHAS = [100, 1.0, 0.5, 0.1]
ROUNDS = 20
LOCAL_EPOCHS = 5
N_GROUPS = 4


def worst_client_both(model, X, y, groups: dict):
    acc_list, bal_list = [], []
    for c, (Xc, yc) in groups.items():
        if len(yc) == 0:
            continue
        pred = (model.predict_proba(Xc) >= 0.5).astype(int)
        acc_list.append(accuracy_score(yc, pred))
        bal_list.append(balanced_accuracy_score(yc, pred) if len(np.unique(yc)) > 1 else accuracy_score(yc, pred))
    return min(acc_list), min(bal_list)


def pooled_both(model, X, y):
    pred = (model.predict_proba(X) >= 0.5).astype(int)
    return accuracy_score(y, pred), balanced_accuracy_score(y, pred)


def fixed_group_assignment(n_rows, seed, fold):
    rng = np.random.default_rng(300_000 + seed * 100 + fold)
    return rng.integers(0, N_GROUPS, size=n_rows)


def worst_group_both(model, X, y, groups):
    acc_list, bal_list = [], []
    for g in range(N_GROUPS):
        mask = groups == g
        if mask.sum() > 0:
            pred = (model.predict_proba(X[mask]) >= 0.5).astype(int)
            yg = y[mask]
            acc_list.append(accuracy_score(yg, pred))
            bal_list.append(balanced_accuracy_score(yg, pred) if len(np.unique(yg)) > 1 else accuracy_score(yg, pred))
    return min(acc_list), min(bal_list)


def run_lr_mlp(factory_fn):
    pool = load_pool()
    keys = ["observed_acc", "observed_bal", "comp_acc", "comp_bal",
            "pooled_acc", "pooled_bal", "wg_acc", "wg_bal"]
    per_replicate = {a: {k: [] for k in keys} for a in ALPHAS}

    for seed in SEEDS:
        assignments = {a: client_assignment(pool, a, seed) for a in ALPHAS}
        for fold_id, df_train, df_test in cv_folds(pool, seed):
            X_train, y_train, X_test, y_test = fit_transform_fold(df_train, df_test)
            fixed_groups = fixed_group_assignment(len(y_test), seed, fold_id)

            trained = {}
            for a in ALPHAS:
                client_train = split_by_client(X_train, y_train, assignments[a], df_train)
                client_data = [client_train[c] for c in sorted(client_train)]
                model = run_federated(lambda: factory_fn(seed), fedavg, client_data, rounds=ROUNDS, local_epochs=LOCAL_EPOCHS)
                trained[a] = model

                client_test = split_by_client(X_test, y_test, assignments[a], df_test)
                wc_acc, wc_bal = worst_client_both(model, X_test, y_test, client_test)
                per_replicate[a]["observed_acc"].append(wc_acc)
                per_replicate[a]["observed_bal"].append(wc_bal)

                p_acc, p_bal = pooled_both(model, X_test, y_test)
                per_replicate[a]["pooled_acc"].append(p_acc)
                per_replicate[a]["pooled_bal"].append(p_bal)

                wg_acc, wg_bal = worst_group_both(model, X_test, y_test, fixed_groups)
                per_replicate[a]["wg_acc"].append(wg_acc)
                per_replicate[a]["wg_bal"].append(wg_bal)

            frozen = trained[100]
            for a in ALPHAS:
                client_test = split_by_client(X_test, y_test, assignments[a], df_test)
                c_acc, c_bal = worst_client_both(frozen, X_test, y_test, client_test)
                per_replicate[a]["comp_acc"].append(c_acc)
                per_replicate[a]["comp_bal"].append(c_bal)

    return per_replicate


def summarize(per_replicate):
    def decline(metric_key):
        return 100 * (np.mean(per_replicate[100][metric_key]) - np.mean(per_replicate[0.1][metric_key]))

    observed_acc = decline("observed_acc")
    comp_acc = decline("comp_acc")
    observed_bal = decline("observed_bal")
    comp_bal = decline("comp_bal")
    pooled_acc_decl = decline("pooled_acc")
    pooled_bal_decl = decline("pooled_bal")
    wg_acc_decl = decline("wg_acc")
    wg_bal_decl = decline("wg_bal")

    return {
        "plain_accuracy": {
            "observed_decline_pp": round(observed_acc, 2),
            "composition_only_decline_pp": round(comp_acc, 2),
            "composition_share_pct": round(100 * comp_acc / observed_acc, 1) if observed_acc else None,
            "residual_pp": round(observed_acc - comp_acc, 2),
            "shared_test_pooled_decline_pp": round(pooled_acc_decl, 2),
            "shared_test_worst_group_decline_pp": round(wg_acc_decl, 2),
        },
        "balanced_accuracy": {
            "observed_decline_pp": round(observed_bal, 2),
            "composition_only_decline_pp": round(comp_bal, 2),
            "composition_share_pct": round(100 * comp_bal / observed_bal, 1) if observed_bal else None,
            "residual_pp": round(observed_bal - comp_bal, 2),
            "shared_test_pooled_decline_pp": round(pooled_bal_decl, 2),
            "shared_test_worst_group_decline_pp": round(wg_bal_decl, 2),
        },
    }


def run_vqc_n2():
    pool = load_pool()
    results = {a: {"observed_acc": [], "observed_bal": [], "comp_acc": [], "comp_bal": []} for a in [100, 0.1]}
    for seed in [0, 5]:
        fold_id = 0
        folds = list(cv_folds(pool, seed))
        _, df_train, df_test = folds[fold_id]
        _, _, X_test, y_test = fit_transform_fold(df_train, df_test)
        assignments = {a: client_assignment(pool, a, seed) for a in [100, 0.1]}

        models = {}
        for a in [100, 0.1]:
            cond_raw = "100" if a == 100 else "0.1"
            data = np.load(f"results/angle_capture/arm4_{cond_raw}_{seed}_{fold_id}.npz")
            model = VQCModel(seed=seed)
            model.set_params(data["global_params"][-1])
            models[a] = model
            client_test = split_by_client(X_test, y_test, assignments[a], df_test)
            wc_acc, wc_bal = worst_client_both(model, X_test, y_test, client_test)
            results[a]["observed_acc"].append(wc_acc)
            results[a]["observed_bal"].append(wc_bal)

        frozen = models[100]
        for a in [100, 0.1]:
            client_test = split_by_client(X_test, y_test, assignments[a], df_test)
            c_acc, c_bal = worst_client_both(frozen, X_test, y_test, client_test)
            results[a]["comp_acc"].append(c_acc)
            results[a]["comp_bal"].append(c_bal)

    observed_acc = 100 * (np.mean(results[100]["observed_acc"]) - np.mean(results[0.1]["observed_acc"]))
    comp_acc = 100 * (np.mean(results[100]["comp_acc"]) - np.mean(results[0.1]["comp_acc"]))
    observed_bal = 100 * (np.mean(results[100]["observed_bal"]) - np.mean(results[0.1]["observed_bal"]))
    comp_bal = 100 * (np.mean(results[100]["comp_bal"]) - np.mean(results[0.1]["comp_bal"]))
    return {
        "n": 2,
        "plain_accuracy": {
            "observed_decline_pp": round(observed_acc, 2),
            "composition_only_decline_pp": round(comp_acc, 2),
            "composition_share_pct": round(100 * comp_acc / observed_acc, 1) if observed_acc else None,
            "residual_pp": round(observed_acc - comp_acc, 2),
        },
        "balanced_accuracy": {
            "observed_decline_pp": round(observed_bal, 2),
            "composition_only_decline_pp": round(comp_bal, 2),
            "composition_share_pct": round(100 * comp_bal / observed_bal, 1) if observed_bal else None,
            "residual_pp": round(observed_bal - comp_bal, 2),
        },
    }


if __name__ == "__main__":
    import json

    out = {}
    print("=== LR ===", flush=True)
    lr_pr = run_lr_mlp(lambda seed: LogisticRegressionModel(6, seed=seed))
    out["LR"] = summarize(lr_pr)
    print(json.dumps(out["LR"], indent=2))

    print("=== MLP ===", flush=True)
    mlp_pr = run_lr_mlp(lambda seed: MLPModel(6, seed=seed))
    out["MLP"] = summarize(mlp_pr)
    print(json.dumps(out["MLP"], indent=2))

    print("=== VQC (n=2) ===", flush=True)
    out["VQC"] = run_vqc_n2()
    print(json.dumps(out["VQC"], indent=2))

    with open("results/dataset1_reeval_balanced.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/dataset1_reeval_balanced.json")
