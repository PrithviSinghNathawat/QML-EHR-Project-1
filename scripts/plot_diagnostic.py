"""Figures for the diagnostic report (docs/diagnostic_report.md).
Reads results/diagnostic_results.csv and results/diagnostic_divergence.csv,
produced by scripts/run_diagnostic.py.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ALPHA_ORDER = ["100", "1.0", "0.5", "0.1"]  # matches string form written by csv.DictWriter
MODELS = ["LR", "MLP"]
COLORS = {"LR": "tab:blue", "MLP": "tab:orange"}


def load():
    df = pd.read_csv("results/diagnostic_results.csv")
    dv = pd.read_csv("results/diagnostic_divergence.csv")
    return df, dv


def plot_divergence(dv, path="results/figs/client_divergence.png"):
    final_round = dv.groupby(["model", "condition", "seed", "fold"])["round"].transform("max")
    final_dv = dv[dv["round"] == final_round]
    summary = final_dv.groupby(["model", "condition"])["mean_pairwise_l2"].agg(["mean", "std"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for model in MODELS:
        y = [summary.loc[(model, a), "mean"] for a in ALPHA_ORDER]
        yerr = [summary.loc[(model, a), "std"] for a in ALPHA_ORDER]
        ax.errorbar(range(len(ALPHA_ORDER)), y, yerr=yerr, marker="o", label=model, color=COLORS[model], capsize=3)
        nat = summary.loc[(model, "natural"), "mean"]
        nat_err = summary.loc[(model, "natural"), "std"]
        ax.errorbar([len(ALPHA_ORDER)], [nat], yerr=[nat_err], marker="s", color=COLORS[model], capsize=3)

    ax.set_xticks(range(len(ALPHA_ORDER) + 1))
    ax.set_xticklabels([str(a) for a in ALPHA_ORDER] + ["natural"])
    ax.set_xlabel("condition (Dirichlet alpha, decreasing = more skewed; natural = real 4-site split)")
    ax.set_ylabel("mean pairwise L2 distance between client params, final round")
    ax.set_title("Client parameter divergence vs. heterogeneity condition")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved: {path}")


def plot_worst_client(df, path="results/figs/worst_client_accuracy.png"):
    per_client = df[(df["arm"] == "arm2") & (df["client"] != "global") & (df["n"] > 0)]
    worst = per_client.groupby(["model", "condition", "seed", "fold"])["accuracy"].min().reset_index()
    worst_summary = worst.groupby(["model", "condition"])["accuracy"].agg(["mean", "std"])

    global_df = df[(df["arm"] == "arm2") & (df["client"] == "global")]
    global_summary = global_df.groupby(["model", "condition"])["accuracy"].agg(["mean", "std"])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, model in zip(axes, MODELS):
        gy = [global_summary.loc[(model, a), "mean"] for a in ALPHA_ORDER]
        gyerr = [global_summary.loc[(model, a), "std"] for a in ALPHA_ORDER]
        wy = [worst_summary.loc[(model, a), "mean"] for a in ALPHA_ORDER]
        wyerr = [worst_summary.loc[(model, a), "std"] for a in ALPHA_ORDER]
        x = range(len(ALPHA_ORDER))
        ax.errorbar(x, gy, yerr=gyerr, marker="o", label="global accuracy", capsize=3)
        ax.errorbar(x, wy, yerr=wyerr, marker="s", label="worst-client accuracy", capsize=3)

        gnat = global_summary.loc[(model, "natural"), "mean"]
        wnat = worst_summary.loc[(model, "natural"), "mean"]
        ax.scatter([len(ALPHA_ORDER)], [gnat], marker="o", color="tab:blue", zorder=5)
        ax.scatter([len(ALPHA_ORDER)], [wnat], marker="s", color="tab:orange", zorder=5)

        ax.set_xticks(range(len(ALPHA_ORDER) + 1))
        ax.set_xticklabels([str(a) for a in ALPHA_ORDER] + ["natural"])
        ax.set_xlabel("condition")
        ax.set_title(model)
        ax.legend()
    axes[0].set_ylabel("accuracy")
    fig.suptitle("Global vs. worst-client accuracy across heterogeneity conditions")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved: {path}")


if __name__ == "__main__":
    df, dv = load()
    plot_divergence(dv)
    plot_worst_client(df)
