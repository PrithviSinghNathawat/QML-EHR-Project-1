"""Dirichlet(alpha) label-skew partitioner.

Pools all sites together (site identity is not used) and splits records into
N_CLIENTS simulated clients. For each class independently, a per-client
proportion vector is drawn from Dirichlet(alpha, ..., alpha) and used to
assign that class's rows to clients. Low alpha -> highly skewed clients
(each mostly one class). High alpha -> close to IID.

Minimum-client-size guard: if any client's total row count would fall below
MIN_CLIENT_SIZE, the draw is rejected and re-drawn (different sub-seed) up
to MAX_ATTEMPTS times. Without this, a low-alpha draw can occasionally hand
a client only 1-2 rows -- not a heterogeneity effect, just a degenerate
sample that would make a local gradient step meaningless.

Also plots the natural 4-site split (by real hospital, not Dirichlet) for
comparison, since site identity alone is already non-IID (see labbook.md).

This script demonstrates partitioning on the full pooled dataset (920 rows)
for the review presentation / paper figures. In the actual federated loop,
partitioning is applied per-seed to the training split only (see
scripts/preprocessing.py) -- the test set is never partitioned.
"""
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from data_loader import get_feature_frame, load_all_sites, SITES  # noqa: E402

N_CLIENTS = 4  # matches the 4 natural sites, so client count isn't a confound
                # when comparing natural vs. Dirichlet partitioning (D-017)
ALPHAS = [100, 1.0, 0.5, 0.1]
SEED = 0

MIN_CLIENT_SIZE = 15  # floor: enough rows for LOCAL_STEPS=5 local gradient
                       # steps to mean something, not a single unstable batch
MAX_ATTEMPTS = 500


def _draw_once(df: pd.DataFrame, alpha: float, n_clients: int, seed: int):
    rng = np.random.default_rng(seed)
    client_indices = [[] for _ in range(n_clients)]
    for cls in sorted(df["target"].unique()):
        cls_idx = df.index[df["target"] == cls].to_numpy().copy()
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(alpha * np.ones(n_clients))
        split_points = (np.cumsum(proportions) * len(cls_idx)).astype(int)[:-1]
        for c, idx in enumerate(np.split(cls_idx, split_points)):
            client_indices[c].extend(idx.tolist())
    return [np.array(idx) for idx in client_indices]


def dirichlet_partition(
    df: pd.DataFrame,
    alpha: float,
    n_clients: int = N_CLIENTS,
    seed: int = SEED,
    min_client_size: int = MIN_CLIENT_SIZE,
):
    """Return a list of n_clients arrays of row-labels (df.index values).
    Guarded: retries with a different sub-seed if any client falls below
    min_client_size. Returns (client_indices, n_attempts)."""
    for attempt in range(MAX_ATTEMPTS):
        client_indices = _draw_once(df, alpha, n_clients, seed=seed * 100_000 + attempt)
        sizes = [len(idx) for idx in client_indices]
        if min(sizes) >= min_client_size:
            return client_indices, attempt + 1
    raise RuntimeError(
        f"could not satisfy min_client_size={min_client_size} for alpha={alpha} "
        f"after {MAX_ATTEMPTS} attempts -- alpha may be too extreme for this "
        f"dataset size / client count."
    )


def summarize(df: pd.DataFrame, group_indices, group_labels) -> pd.DataFrame:
    rows = []
    for label, idx in zip(group_labels, group_indices):
        sub = df.loc[idx, "target"]
        n = len(sub)
        n1 = int((sub == 1).sum())
        rows.append(
            {
                "group": label,
                "n": n,
                "n_class_0": n - n1,
                "n_class_1": n1,
                "pct_class_1": round(100 * n1 / n, 1) if n else float("nan"),
            }
        )
    return pd.DataFrame(rows).set_index("group")


def stacked_bar(ax, summary: pd.DataFrame, title: str):
    ax.bar(summary.index.astype(str), summary["n_class_0"], label="class 0")
    ax.bar(
        summary.index.astype(str),
        summary["n_class_1"],
        bottom=summary["n_class_0"],
        label="class 1",
    )
    ax.set_title(title)
    ax.set_xlabel("client")
    ax.set_ylabel("n records")


if __name__ == "__main__":
    df, _ = load_all_sites()
    df = get_feature_frame(df)

    print(f"minimum-client-size guard: floor={MIN_CLIENT_SIZE} rows, max {MAX_ATTEMPTS} redraw attempts\n")

    # natural (real-hospital) partition, for comparison
    natural_summary = summarize(df, [df.index[df["site"] == s] for s in SITES], SITES)
    print("=== natural 4-site partition ===")
    print(natural_summary.to_string())

    alpha_summaries = {}
    for alpha in ALPHAS:
        client_idx, n_attempts = dirichlet_partition(df, alpha)
        summary = summarize(df, client_idx, list(range(N_CLIENTS)))
        alpha_summaries[alpha] = summary
        guard_note = f"(guard fired -- needed {n_attempts} draws)" if n_attempts > 1 else "(guard did not need to fire -- first draw satisfied the floor)"
        print(f"\n=== alpha={alpha} {guard_note} ===")
        print(summary.to_string())

        fig, ax = plt.subplots(figsize=(5, 4))
        stacked_bar(ax, summary, f"Dirichlet partition, alpha={alpha}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"results/figs/partition_alpha_{alpha}.png", dpi=150)
        plt.close(fig)

    # comparison figure: natural site split alongside all 4 alphas
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=False)
    stacked_bar(axes[0], natural_summary, "Natural (real site)")
    axes[0].set_xlabel("site")
    for ax, alpha in zip(axes[1:], ALPHAS):
        stacked_bar(ax, alpha_summaries[alpha], f"Dirichlet alpha={alpha}")
    axes[0].legend()
    fig.suptitle("Natural site skew vs. Dirichlet-controlled synthetic skew")
    fig.tight_layout()
    fig.savefig("results/figs/partition_natural_vs_dirichlet.png", dpi=150)
    plt.close(fig)

    print("\nsaved: results/figs/partition_alpha_{100,1.0,0.5,0.1}.png")
    print("saved: results/figs/partition_natural_vs_dirichlet.png")
