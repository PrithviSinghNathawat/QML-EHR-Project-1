"""Append-only CSV run logger. Shared by every script that produces a result
worth keeping -- including throwaway/diagnostic runs, per CLAUDE.md.
"""
import csv
import os

RUNS_CSV = os.path.join(os.path.dirname(__file__), "..", "results", "runs.csv")
FIELDS = [
    "arm",
    "alpha",
    "seed",
    "f1",
    "auroc",
    "accuracy",
    "rounds_to_converge",
    "wall_clock_sec",
    "n_qubits",
    "circuit_depth",
]


def append_run(row: dict) -> None:
    write_header = not os.path.exists(RUNS_CSV) or os.path.getsize(RUNS_CSV) == 0
    with open(RUNS_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in FIELDS})
