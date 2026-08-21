"""Launches the 50-replicate VQC composition-only decomposition
(10 seeds x 5 folds, alpha=100 training only) with 4-way parallelism,
same resumable pattern as arm4_orchestrator.py.
"""
import os
import subprocess
import sys
import time

SEEDS = list(range(10))
FOLDS = list(range(5))
N_WORKERS = 4
OUT_DIR = "results/vqc_composition_partial"


def all_combos():
    return [(s, f) for s in SEEDS for f in FOLDS]


def output_path(seed, fold):
    return f"{OUT_DIR}/{seed}_{fold}.json"


if __name__ == "__main__":
    combos = all_combos()
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    pending = [c for c in combos if not os.path.exists(output_path(*c))]
    print(f"{len(combos) - len(pending)}/{len(combos)} already done, {len(pending)} to run")

    running = {}
    start = time.perf_counter()
    done = len(combos) - len(pending)

    while pending or running:
        while pending and len(running) < N_WORKERS:
            seed, fold = pending.pop(0)
            p = subprocess.Popen(
                [sys.executable, "-u", "scripts/vqc_composition_worker.py", str(seed), str(fold)],
                env=env,
            )
            running[p] = (seed, fold)
        for p in list(running):
            if p.poll() is not None:
                combo = running.pop(p)
                done += 1
                print(f"[{done}/{len(combos)}] seed={combo[0]} fold={combo[1]} done ({time.perf_counter()-start:.0f}s elapsed)")
        time.sleep(0.2)

    print(f"\nVQC composition decomposition complete: {len(combos)} replicates in {time.perf_counter()-start:.1f}s")
