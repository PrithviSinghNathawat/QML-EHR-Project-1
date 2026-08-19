"""Feature count vs. missingness retention threshold -- goes in the paper.

Shows how many features would survive at any given "minimum % present at
every site" cutoff, so the final 6-feature selection (age, sex, cp, restecg,
thalach, exang -- the 6 highest worst-site-availability columns) is shown as
a data-driven choice sitting on a real cliff, not an arbitrary cutoff.
"""
import sys

import matplotlib.pyplot as plt

sys.path.insert(0, "scripts")
from data_loader import COLUMNS, FEATURE_COLUMNS, load_all_sites, SITES  # noqa: E402

BENCHMARK_THRESHOLDS = [85, 70, 50]  # % present required, as asked

if __name__ == "__main__":
    df, _ = load_all_sites()
    candidate_cols = [c for c in COLUMNS if c != "num"]

    # worst-site missingness per column -- this alone determines, for any
    # "min % present at every site" threshold, whether the column survives.
    worst_site_missing = {}
    for col in candidate_cols:
        per_site = {s: 100 * df.loc[df["site"] == s, col].isna().mean() for s in SITES}
        worst_site_missing[col] = max(per_site.values())

    ranked = sorted(worst_site_missing.items(), key=lambda kv: kv[1])
    print("columns ranked by worst-site missingness (the site with the least data for that column):")
    for col, pct in ranked:
        print(f"  {col:<10} worst-site missing = {pct:5.1f}%")

    # sweep threshold = min % present required at every site, 50..100
    thresholds_present = list(range(50, 101))
    survivor_counts = []
    for t in thresholds_present:
        max_allowed_missing = 100 - t
        n_survive = sum(1 for pct in worst_site_missing.values() if pct <= max_allowed_missing)
        survivor_counts.append(n_survive)

    print("\nbenchmark thresholds:")
    for t in BENCHMARK_THRESHOLDS:
        max_allowed_missing = 100 - t
        survivors = [c for c, pct in worst_site_missing.items() if pct <= max_allowed_missing]
        print(f"  >={t}% present at every site -> {len(survivors)} features: {survivors}")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.step(thresholds_present, survivor_counts, where="post")
    for t in BENCHMARK_THRESHOLDS:
        max_allowed_missing = 100 - t
        n = sum(1 for pct in worst_site_missing.values() if pct <= max_allowed_missing)
        ax.scatter([t], [n], zorder=5)
        ax.annotate(f"{t}% -> {n}", (t, n), textcoords="offset points", xytext=(6, 6))
    ax.axhline(
        len(FEATURE_COLUMNS),
        color="gray",
        linestyle="--",
        linewidth=1,
        label=f"selected feature count = {len(FEATURE_COLUMNS)} ({', '.join(FEATURE_COLUMNS)})",
    )
    ax.set_xlabel("retention threshold: min % present required at every site")
    ax.set_ylabel("number of surviving features")
    ax.set_title("Feature count vs. missingness retention threshold")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig("results/figs/feature_retention_curve.png", dpi=150)
    fig.savefig("results/figs/feature_threshold_curve.png", dpi=220)
    plt.close(fig)

    print("\nsaved: results/figs/feature_retention_curve.png")
    print("saved: results/figs/feature_threshold_curve.png (dpi=220, paper copy)")
