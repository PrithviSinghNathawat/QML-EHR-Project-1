"""Launches Arm 4 (VQC + FedAvg) replicates as concurrent subprocesses
(validated 4-way parallelism, D-020), resumable by construction (each
replicate's output file in results/arm4_partial/ is the resume marker --
scripts/arm4_worker.py skips a replicate whose output already exists).

Task 3 fix (long runs were unobservable): PYTHONUNBUFFERED=1 set for every
worker subprocess, and results/runs_arm4.csv is appended to immediately
after each replicate completes -- progress is checkable by reading that
CSV at any time, no need to wait for a notification or read stdout.

Usage: arm4_orchestrator.py [--estimate-only N]
  --estimate-only N: run only the first N replicates (for timing), then stop.
  (no args): run the full 250-replicate grid (5 folds x 10 seeds x 5 conditions).
"""
import csv
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, "scripts")
from log_run import FIELDS as RUNS_FIELDS  # noqa: E402

SEEDS = list(range(10))
FOLDS = list(range(5))
CONDITIONS = ["100", "1.0", "0.5", "0.1", "natural"]
N_WORKERS = 4

PARTIAL_DIR = "results/arm4_partial"
RUNS_ARM4_CSV = "results/runs_arm4.csv"


def all_combos():
    return [(c, s, f) for c in CONDITIONS for s in SEEDS for f in FOLDS]


def output_path(condition, seed, fold):
    return f"{PARTIAL_DIR}/{condition}_{seed}_{fold}.json"


def append_runs_arm4_row(result: dict):
    write_header = not os.path.exists(RUNS_ARM4_CSV) or os.path.getsize(RUNS_ARM4_CSV) == 0
    g = result["global"]
    row = {
        "arm": "arm4",
        "alpha": result["condition"] if result["condition"] != "natural" else "",
        "seed": result["seed"],
        "f1": g["f1"],
        "auroc": g["auroc"],
        "accuracy": g["accuracy"],
        "rounds_to_converge": "",
        "wall_clock_sec": result["wall_clock_sec"],
        "n_qubits": 6,
        "circuit_depth": 3,
    }
    with open(RUNS_ARM4_CSV, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUNS_FIELDS)
        if write_header:
            w.writeheader()
        w.writerow(row)


def run_batch(combos, n_workers=N_WORKERS):
    os.makedirs(PARTIAL_DIR, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"  # Task 3 fix

    pending = [c for c in combos if not os.path.exists(output_path(*c))]
    print(f"{len(combos) - len(pending)}/{len(combos)} already done, {len(pending)} to run")

    running = {}  # proc -> combo
    done_count = len(combos) - len(pending)
    start = time.perf_counter()

    def launch(combo):
        condition, seed, fold = combo
        p = subprocess.Popen(
            [sys.executable, "-u", "scripts/arm4_worker.py", condition, str(seed), str(fold)],
            env=env,
        )
        running[p] = combo

    while pending or running:
        while pending and len(running) < n_workers:
            launch(pending.pop(0))

        for p in list(running):
            if p.poll() is not None:
                condition, seed, fold = running.pop(p)
                out = output_path(condition, seed, fold)
                if os.path.exists(out):
                    with open(out) as f:
                        result = json.load(f)
                    append_runs_arm4_row(result)
                    done_count += 1
                    elapsed = time.perf_counter() - start
                    print(f"[{done_count}/{len(combos)}] {condition} seed={seed} fold={fold} done ({elapsed:.0f}s elapsed)")
                else:
                    print(f"WARNING: worker for {condition} seed={seed} fold={fold} exited without output")
        time.sleep(0.2)

    total = time.perf_counter() - start
    print(f"\nbatch complete: {len(combos)} replicates in {total:.1f}s")
    return total


if __name__ == "__main__":
    combos = all_combos()
    if len(sys.argv) > 2 and sys.argv[1] == "--estimate-only":
        n = int(sys.argv[2])
        combos = combos[:n]
        print(f"ESTIMATE MODE: running only the first {n} replicates")
    run_batch(combos)
