"""Consolidates the composition-vs-training decomposition (D-044-D-052,
docs/arm4_report.md) into one results CSV, so plots.py can read it like any
other results source instead of the numbers only existing in a markdown
report and one-off interactive runs.

Three pieces combined here, none re-derived by hand:
- Observed worst-client accuracy per model/condition: reuses
  worst_client.py (already-persisted methodology) against the existing
  diagnostic_results.csv / arm4_diagnostic_results.csv.
- LR/MLP composition-only curve: reuses
  composition_decomposition.composition_only_curve() directly (imported,
  not re-implemented) -- this re-trains a single alpha=100 federated model
  per (seed, fold) and re-evaluates it against every condition's test-slice
  composition. Same protocol Prithvi's D-047 used, cheap (~seconds, same
  order as the Arm 2 grid).
- VQC composition-only curve: aggregates the 50 already-computed
  per-(seed,fold) JSON files in results/vqc_composition_partial/ -- no
  training, just reading files that already exist.

Output: results/composition_decomposition_summary.csv, one row per
(model, condition): observed_mean, composition_mean.
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "scripts")
from composition_decomposition import composition_only_curve  # noqa: E402
from models import LogisticRegressionModel  # noqa: E402
from models_mlp import MLPModel  # noqa: E402
from worst_client import load_and_summarize  # noqa: E402
from aggregators import fedavg  # noqa: E402
from cv_protocol import split_by_client  # noqa: E402
from federated_loop import run_federated  # noqa: E402

CONDITIONS = ["100", "1.0", "0.5", "0.1", "natural"]
OUT_PATH = "results/composition_decomposition_summary.csv"


def _fed_train_100(model_factory, seed, X_train, y_train, df_train_fold, assign_100):
    client_train = split_by_client(X_train, y_train, assign_100, df_train_fold)
    client_data = [client_train[c] for c in sorted(client_train)]
    return run_federated(lambda: model_factory(seed), fedavg, client_data, rounds=20, local_epochs=5)


def observed_curve(model: str) -> dict:
    """model -> {condition: mean_worst_client_accuracy}, arm2 rows only."""
    path = "results/arm4_diagnostic_results.csv" if model == "VQC" else "results/diagnostic_results.csv"
    summary = load_and_summarize(path)
    out = {}
    for cond in CONDITIONS:
        key = (("arm2", model, cond) if model != "VQC" else ("arm4", model, cond))
        if key in summary.index:
            out[cond] = summary.loc[key, "mean"]
    return out


def composition_curve_classical(model: str) -> dict:
    factory = (lambda seed: LogisticRegressionModel(6, seed=seed)) if model == "LR" else (lambda seed: MLPModel(6, seed=seed))
    result = composition_only_curve(factory, epochs=20, arm_two_train_fn=_fed_train_100)
    return {str(cond): mean for cond, (mean, std, n) in result.items()}


def composition_curve_vqc() -> dict:
    files = sorted(glob.glob("results/vqc_composition_partial/*.json"))
    per_condition = {cond: [] for cond in CONDITIONS}
    for path in files:
        with open(path) as f:
            data = json.load(f)
        for cond in CONDITIONS:
            client_accs = [v["accuracy"] for v in data["per_condition"][cond].values() if v["n"] > 0]
            if client_accs:
                per_condition[cond].append(min(client_accs))
    return {cond: float(np.mean(v)) for cond, v in per_condition.items() if v}


if __name__ == "__main__":
    rows = []
    for model in ["LR", "MLP", "VQC"]:
        observed = observed_curve(model)
        composition = composition_curve_vqc() if model == "VQC" else composition_curve_classical(model)
        for cond in CONDITIONS:
            if cond in observed and cond in composition:
                rows.append({
                    "model": model,
                    "condition": cond,
                    "observed_worst_client_accuracy": observed[cond],
                    "composition_only_worst_client_accuracy": composition[cond],
                })
        print(f"{model}: observed={observed}, composition={composition}")

    os.makedirs("results", exist_ok=True)
    pd.DataFrame(rows).to_csv(OUT_PATH, index=False)
    print(f"saved: {OUT_PATH}")
