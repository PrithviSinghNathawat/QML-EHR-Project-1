"""P-017: per-client TEST partition sizes for dataset 2, K=4 and K=130, at
every alpha the actual decomposition grid (P-014) used -- same seeds (0-9),
same client_assignment call, same cv_folds split, so these numbers describe
exactly the partitions that produced P-014's results, not a fresh draw.

K=130/alpha=0.1 is not computed -- P-014 already established it is
structurally infeasible (no valid draw exists under the min-client-size
floor), so there is no partition to size.
"""
import json
import sys

import numpy as np

sys.path.insert(0, "scripts")
from dataset2_cv_protocol import InfeasiblePartition, client_assignment, cv_folds, load_pool  # noqa: E402

SEEDS = list(range(10))
CONFIGS = [
    (4, [100, 1.0, 0.5, 0.1]),
    (130, [100, 1.0, 0.5]),  # 0.1 infeasible at K=130, see P-014
]


def partition_sizes(df, alpha, n_clients):
    sizes = []
    for seed in SEEDS:
        assign, _ = client_assignment(df, alpha, seed=seed, n_clients=n_clients)
        for fold_id, df_train_fold, df_test_fold in cv_folds(df, seed):
            fold_assign = assign.loc[df_test_fold.index]
            for c in range(n_clients):
                n = int((fold_assign == c).sum())
                if n > 0 or True:  # a client can legitimately have 0 test rows in a given fold
                    sizes.append(n)
    return np.array(sizes)


if __name__ == "__main__":
    df, _numeric_cols, _has_race = load_pool()
    output = {}
    for n_clients, alphas in CONFIGS:
        output[str(n_clients)] = {}
        for alpha in alphas:
            sizes = partition_sizes(df, alpha, n_clients)
            stats = {
                "mean": float(sizes.mean()),
                "median": float(np.median(sizes)),
                "min": int(sizes.min()),
                "max": int(sizes.max()),
                "n_cells": len(sizes),
                "n_zero": int((sizes == 0).sum()),
            }
            output[str(n_clients)][str(alpha)] = stats
            print(f"K={n_clients} alpha={alpha}: mean={stats['mean']:.1f} median={stats['median']:.1f} "
                  f"min={stats['min']} max={stats['max']} n_zero_cells={stats['n_zero']} (n={stats['n_cells']})")

    with open("results/dataset2_partition_sizes.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nwrote results/dataset2_partition_sizes.json")
