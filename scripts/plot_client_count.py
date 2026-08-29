"""Client-count figure for the Review-2 deck: shared-test-implied composition
share vs. client count (K=4 vs K=130), dataset 2 (Diabetes 130-US Hospitals).

Separate from scripts/plots.py deliberately -- this is a one-off figure for
the presentation, not a pipeline primary; plots.py is Prithvi's actively-
evolving module and this avoids touching it for a single deck asset.

Reads results/dataset2_decomposition_weighted.json (the same source as the
paper draft's SS V.F table, P-014) -- no new computation, just visualization
of already-verified numbers.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

SRC = "results/dataset2_decomposition_weighted.json"
OUT = "results/figs/client_count_composition_share.png"

MODELS = ["LR", "MLP"]
ALPHA_PAIRS = ["100_to_1.0", "100_to_0.5"]
ALPHA_LABELS = {"100_to_1.0": "α: 100→1.0", "100_to_0.5": "α: 100→0.5"}
K_VALUES = [4, 130]


def load_ranges(data):
    """Returns {(model, alpha_pair): {K: (lo, hi)}}"""
    out = {}
    for model in MODELS:
        for pair in ALPHA_PAIRS:
            out[(model, pair)] = {}
            for k in K_VALUES:
                entry = data[f"{model}_K{k}"]["summary"].get(pair)
                if entry:
                    out[(model, pair)][k] = tuple(entry["implied_composition_share_range_pct"])
    return out


def make_figure(ranges):
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    colors = {"LR": "#1f77b4", "MLP": "#ff7f0e"}

    for ax, pair in zip(axes, ALPHA_PAIRS):
        x = np.arange(len(K_VALUES))
        width = 0.35
        for i, model in enumerate(MODELS):
            los = [ranges[(model, pair)][k][0] for k in K_VALUES]
            his = [ranges[(model, pair)][k][1] for k in K_VALUES]
            mids = [(lo + hi) / 2 for lo, hi in zip(los, his)]
            errs = [(hi - lo) / 2 for lo, hi in zip(los, his)]
            ax.bar(
                x + (i - 0.5) * width, mids, width, yerr=errs, capsize=4,
                label=model, color=colors[model],
            )
        ax.set_xticks(x)
        ax.set_xticklabels([f"K={k}" for k in K_VALUES])
        ax.set_title(ALPHA_LABELS[pair])
        ax.set_ylim(0, 105)
        ax.axhline(100, linestyle="--", color="gray", alpha=0.4, linewidth=1)
        ax.grid(alpha=0.2, axis="y")

    axes[0].set_ylabel("shared-test-implied composition share (%)")
    axes[0].legend(loc="lower right", fontsize=9)
    fig.suptitle("Composition share grows with client count (Dataset 2)")
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=200)
    plt.close(fig)
    print(f"saved: {OUT}")


if __name__ == "__main__":
    with open(SRC) as f:
        data = json.load(f)
    ranges = load_ranges(data)
    for (model, pair), by_k in ranges.items():
        print(model, pair, by_k)
    make_figure(ranges)
