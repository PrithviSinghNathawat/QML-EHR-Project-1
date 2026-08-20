"""Merge results/arm4_partial/*.json (one per replicate) into the same
long-format CSVs used for the classical diagnostic (results/diagnostic_
results.csv / results/diagnostic_divergence.csv), so Arm 4 can be analyzed
with the same tooling: results/arm4_diagnostic_results.csv and
results/arm4_diagnostic_divergence.csv.
"""
import csv
import glob
import json

RESULTS_FIELDS = ["seed", "fold", "model", "arm", "condition", "client", "n", "accuracy", "f1", "auroc"]
DIVERGENCE_FIELDS = ["seed", "fold", "model", "condition", "round", "mean_pairwise_l2"]

if __name__ == "__main__":
    files = sorted(glob.glob("results/arm4_partial/*.json"))
    print(f"merging {len(files)} replicate files")

    with open("results/arm4_diagnostic_results.csv", "w", newline="") as rf, \
         open("results/arm4_diagnostic_divergence.csv", "w", newline="") as dvf:
        rw = csv.DictWriter(rf, fieldnames=RESULTS_FIELDS)
        rw.writeheader()
        dvw = csv.DictWriter(dvf, fieldnames=DIVERGENCE_FIELDS)
        dvw.writeheader()

        for path in files:
            with open(path) as f:
                r = json.load(f)
            seed, fold, condition = r["seed"], r["fold"], r["condition"]

            g = r["global"]
            rw.writerow({"seed": seed, "fold": fold, "model": "VQC", "arm": "arm4",
                         "condition": condition, "client": "global",
                         "n": g["n"], "accuracy": g["accuracy"], "f1": g["f1"], "auroc": g["auroc"]})

            for c, m in r["per_client"].items():
                rw.writerow({"seed": seed, "fold": fold, "model": "VQC", "arm": "arm4",
                             "condition": condition, "client": c,
                             "n": m["n"], "accuracy": m["accuracy"], "f1": m["f1"], "auroc": m["auroc"]})

            for round_idx, div in enumerate(r["divergence"]):
                dvw.writerow({"seed": seed, "fold": fold, "model": "VQC",
                              "condition": condition, "round": round_idx, "mean_pairwise_l2": div})

    print("wrote results/arm4_diagnostic_results.csv, results/arm4_diagnostic_divergence.csv")
