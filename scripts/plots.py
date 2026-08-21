"""Figure generation for the primary results: worst-client accuracy, global
accuracy, client parameter divergence, and the composition-vs-training
decomposition, all vs heterogeneity (alpha).

Reads long-format results CSVs (see RESULTS_SOURCES / DIVERGENCE_SOURCES
below) and produces dpi>=200 PNGs in results/figs/. A missing source file is
skipped with a printed warning, not a crash -- this was true when Arm 4/5
didn't exist yet, and stays true for any future arm.

To add a new arm once its results CSV exists: add its path to
RESULTS_SOURCES (and DIVERGENCE_SOURCES if it tracks divergence). No other
code change is needed, as long as the new CSV reuses the same long-format
columns already used here: seed, fold, model, arm, condition, client, n,
accuracy, f1, auroc (results) / seed, fold, model, condition, round,
mean_pairwise_l2 (divergence). Confirmed working end-to-end for Arms 3, 4,
and 5, all of which reuse this exact schema without modification.

Arm 1 (centralized) has no client axis and no alpha axis by construction --
it is not part of the worst-client/global/divergence figures, which are all
heterogeneity-vs-something plots. That is a consequence of what Arm 1 is,
not an oversight.

The fourth figure (composition-vs-training decomposition, D-044-D-052) is
the primary result as of 2026-08-22: it shows that most of the observed
worst-client decline for LR and the VQC is an evaluation-composition
artifact, not a genuine training-heterogeneity effect, while MLP's is
mostly real. Reads results/composition_decomposition_summary.csv, produced
by scripts/composition_summary.py (run once; not regenerated on every
plots.py call, since it involves live federated training for LR/MLP).
"""
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd

RESULTS_DIR = "results"
FIGS_DIR = os.path.join(RESULTS_DIR, "figs")

RESULTS_SOURCES = [
    os.path.join(RESULTS_DIR, "diagnostic_results.csv"),
    os.path.join(RESULTS_DIR, "arm3_diagnostic_results.csv"),
    os.path.join(RESULTS_DIR, "arm4_diagnostic_results.csv"),
    os.path.join(RESULTS_DIR, "arm5_diagnostic_results.csv"),
]
DIVERGENCE_SOURCES = [
    os.path.join(RESULTS_DIR, "diagnostic_divergence.csv"),
    os.path.join(RESULTS_DIR, "arm3_diagnostic_divergence.csv"),
    os.path.join(RESULTS_DIR, "arm4_diagnostic_divergence.csv"),
    os.path.join(RESULTS_DIR, "arm5_diagnostic_divergence.csv"),
]
COMPOSITION_SOURCE = os.path.join(RESULTS_DIR, "composition_decomposition_summary.csv")

ALPHAS = [100, 1.0, 0.5, 0.1]  # sweep order: low heterogeneity -> high


def _load(paths: list) -> pd.DataFrame:
    frames = []
    for path in paths:
        if not os.path.exists(path):
            warnings.warn(f"plots.py: skipping missing results file: {path}")
            continue
        frames.append(pd.read_csv(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _alpha_subset(df: pd.DataFrame) -> pd.DataFrame:
    sub = df[df["condition"].astype(str).isin([str(a) for a in ALPHAS])].copy()
    sub["alpha"] = sub["condition"].astype(float)
    return sub


def worst_client_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, model, condition, seed, fold): min accuracy over
    that replicate's non-global clients."""
    per_client = df[df["client"] != "global"]
    return (
        per_client.groupby(["arm", "model", "condition", "seed", "fold"])["accuracy"]
        .min()
        .reset_index()
    )


def global_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["client"] == "global"][["arm", "model", "condition", "seed", "fold", "accuracy"]]


def final_round_divergence(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, condition, seed, fold): divergence at the last
    federated round (matches docs/diagnostic_report.md Section 2)."""
    idx = df.groupby(["model", "condition", "seed", "fold"])["round"].idxmax()
    return df.loc[idx, ["model", "condition", "seed", "fold", "mean_pairwise_l2"]]


def _series_colors(df: pd.DataFrame, series_cols: list) -> dict:
    """Fixed color per series, keyed off the full (unfiltered) source data --
    not assigned per-figure -- so a series like Arm 2 keeps the same color in
    every figure it appears in, even when other series (e.g. Arm 1, which has
    no alpha-conditioned global accuracy) drop out of one particular figure."""
    keys = sorted((key for key, _ in df.groupby(series_cols)), key=str)
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {k: cycle[i % len(cycle)] for i, k in enumerate(keys)}


def _plot_vs_alpha(df, value_col, series_cols, title, ylabel, out_path, color_map, ylim=None):
    fig, ax = plt.subplots(figsize=(7.5, 5))
    for key, sub in df.groupby(series_cols):
        alpha_sub = _alpha_subset(sub)
        if alpha_sub.empty:
            # e.g. Arm 1's global accuracy has no alpha-conditioned rows at
            # all (it isn't partitioned) -- skip rather than draw an empty
            # line with a legend entry for data that doesn't exist.
            continue
        label = " / ".join(str(k) for k in (key if isinstance(key, tuple) else (key,)))
        stats = alpha_sub.groupby("alpha")[value_col].agg(["mean", "std"]).reindex(ALPHAS)
        color = color_map[key]
        ax.errorbar(
            stats.index, stats["mean"], yerr=stats["std"].fillna(0),
            marker="o", capsize=3, label=label, color=color,
        )

        natural = sub[sub["condition"] == "natural"]
        if len(natural):
            nat_mean = natural[value_col].mean()
            # In the legend, not as floating text -- with several series close
            # together the annotated-text version overlapped illegibly and
            # ran past the axes edge at alpha=0.1.
            ax.axhline(
                nat_mean, linestyle="--", color=color, alpha=0.5, linewidth=1,
                label=f"{label} natural (~α 0.5-1.0)",
            )

    ax.set_xscale("log")
    ax.invert_xaxis()  # left = IID (alpha=100), right = extreme skew (alpha=0.1)
    ax.set_xlabel("α (Dirichlet concentration -- log scale, lower = more skewed)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(ylim)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(FIGS_DIR, exist_ok=True)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved: {out_path}")


def make_worst_client_figure(df: pd.DataFrame, color_map: dict, ylim=None):
    wc = worst_client_accuracy(df)
    _plot_vs_alpha(
        wc, "accuracy", ["arm", "model"],
        "Worst-client accuracy vs. heterogeneity (observed, pre-decomposition)",
        "worst-client accuracy",
        os.path.join(FIGS_DIR, "worst_client_accuracy_vs_alpha.png"),
        color_map, ylim=ylim,
    )


def make_global_accuracy_figure(df: pd.DataFrame, color_map: dict, ylim=None):
    ga = global_accuracy(df)
    _plot_vs_alpha(
        ga, "accuracy", ["arm", "model"],
        "Global accuracy vs. heterogeneity",
        "global accuracy",
        os.path.join(FIGS_DIR, "global_accuracy_vs_alpha.png"),
        color_map, ylim=ylim,
    )


def make_divergence_figure(div_df: pd.DataFrame):
    fr = final_round_divergence(div_df)
    color_map = _series_colors(div_df, ["model"])
    _plot_vs_alpha(
        fr, "mean_pairwise_l2", ["model"],
        "Client parameter divergence vs. heterogeneity (final round)",
        "mean pairwise L2 distance between client params",
        os.path.join(FIGS_DIR, "client_divergence_vs_alpha.png"),
        color_map,
    )


def make_decomposition_figure(comp_path: str = COMPOSITION_SOURCE):
    """The primary figure (per Task 4, 2026-08-22): observed worst-client
    decline, composition-only decline (a fixed alpha=100-trained model
    scored against increasingly skewed test slices), and the residual
    (training effect) they imply, for each model. See
    scripts/composition_summary.py for how the source CSV is built, and
    docs/decisions.md D-044-D-052 / docs/arm4_report.md for the full
    methodology and reasoning. Skips gracefully if the source CSV doesn't
    exist yet -- same convention as every other figure here."""
    if not os.path.exists(comp_path):
        warnings.warn(f"plots.py: skipping missing results file: {comp_path}")
        return

    df = pd.read_csv(comp_path)
    rows = []
    for model, sub in df.groupby("model"):
        sub = sub.set_index("condition")
        if "100" not in sub.index or "0.1" not in sub.index:
            continue
        observed_decline = sub.loc["100", "observed_worst_client_accuracy"] - sub.loc["0.1", "observed_worst_client_accuracy"]
        composition_decline = sub.loc["100", "composition_only_worst_client_accuracy"] - sub.loc["0.1", "composition_only_worst_client_accuracy"]
        residual = observed_decline - composition_decline
        rows.append({"model": model, "Observed decline": observed_decline, "Composition-only decline": composition_decline, "Residual (training effect)": residual})

    if not rows:
        warnings.warn("plots.py: composition summary CSV has no usable alpha=100/0.1 rows -- skipping decomposition figure")
        return

    plot_df = pd.DataFrame(rows).set_index("model")
    model_order = [m for m in ["LR", "MLP", "VQC"] if m in plot_df.index] + [m for m in plot_df.index if m not in ["LR", "MLP", "VQC"]]
    plot_df = plot_df.loc[model_order]

    fig, ax = plt.subplots(figsize=(7.5, 5))
    x = range(len(plot_df))
    width = 0.25
    metrics = ["Observed decline", "Composition-only decline", "Residual (training effect)"]
    for i, metric in enumerate(metrics):
        offsets = [xi + (i - 1) * width for xi in x]
        ax.bar(offsets, plot_df[metric] * 100, width=width, label=metric)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(plot_df.index)
    ax.set_ylabel("worst-client accuracy, percentage points (α=100 → α=0.1)")
    ax.set_title("Worst-client decline decomposed: composition vs. training effect")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.2, axis="y")
    fig.tight_layout()
    os.makedirs(FIGS_DIR, exist_ok=True)
    out_path = os.path.join(FIGS_DIR, "composition_decomposition.png")
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"saved: {out_path}")


def _shared_accuracy_ylim(results_df: pd.DataFrame):
    wc = _alpha_subset(worst_client_accuracy(results_df))
    ga = _alpha_subset(global_accuracy(results_df))
    lo = min(wc["accuracy"].min(), ga["accuracy"].min())
    hi = max(wc["accuracy"].max(), ga["accuracy"].max())
    pad = 0.03
    return (lo - pad, hi + pad)


if __name__ == "__main__":
    results_df = _load(RESULTS_SOURCES)
    div_df = _load(DIVERGENCE_SOURCES)

    if results_df.empty:
        print("no results data found -- nothing to plot for accuracy figures")
    else:
        ylim = _shared_accuracy_ylim(results_df)
        color_map = _series_colors(results_df, ["arm", "model"])
        make_worst_client_figure(results_df, color_map, ylim=ylim)
        make_global_accuracy_figure(results_df, color_map, ylim=ylim)

    if div_df.empty:
        print("no divergence data found -- nothing to plot for divergence figure")
    else:
        make_divergence_figure(div_df)

    make_decomposition_figure()
