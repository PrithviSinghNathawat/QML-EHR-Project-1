"""Reusable worst-client aggregation methodology, extracted from the
ad-hoc interactive analysis that originally produced the numbers in
docs/arm4_report.md and docs/arm5_report.md (D-039 -- that methodology
was never persisted as code until now, so those numbers were not
reproducible from committed code before this file existed).

Matches the one prior committed-code precedent, scripts/plot_diagnostic.py
lines 48-49 (written for the classical diagnostic sweep): per
(model, condition, seed, fold), take the minimum accuracy across clients;
then average (mean, std) across the (seed, fold) replicates. Not pooled.

Works on any results CSV sharing the long-format schema used by
results/diagnostic_results.csv, results/arm4_diagnostic_results.csv, and
results/arm5_diagnostic_results.csv: columns
seed, fold, model, arm, condition, client, n, accuracy, f1, auroc.
"""
import pandas as pd


def worst_client_per_replicate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, model, condition, seed, fold): the minimum
    per-client accuracy in that replicate (excludes the 'global' row and
    any client with n=0, i.e. absent from that fold's test slice).

    'arm' MUST be a grouping key: a results CSV can contain more than one
    arm sharing the same model label (e.g. results/diagnostic_results.csv
    has both arm1 [centralized, evaluated per-condition] and arm2
    [federated, trained per-condition] rows for 'LR' and 'MLP') --
    grouping by model alone silently pools two different experiments
    together. Caught by exactly this bug during Task 1 verification,
    2026-08-20 (see docs/decisions.md)."""
    per_client = df[(df["client"] != "global") & (df["n"] > 0)]
    return (
        per_client.groupby(["arm", "model", "condition", "seed", "fold"])["accuracy"]
        .min()
        .reset_index()
        .rename(columns={"accuracy": "worst_client_accuracy"})
    )


def worst_client_summary(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (arm, model, condition): mean, std, count of
    worst-client accuracy across the (seed, fold) replicates."""
    per_rep = worst_client_per_replicate(df)
    return per_rep.groupby(["arm", "model", "condition"])["worst_client_accuracy"].agg(["mean", "std", "count"])


def load_and_summarize(results_csv_path: str) -> pd.DataFrame:
    return worst_client_summary(pd.read_csv(results_csv_path))


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "results/arm4_diagnostic_results.csv"
    print(load_and_summarize(path).round(4).to_string())
