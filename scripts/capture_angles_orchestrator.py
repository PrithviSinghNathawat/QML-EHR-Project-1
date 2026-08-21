"""Runs the 20-replicate angle-capture sample (5 conditions x 2 seeds x
1 fold x 2 arms) with 4-way parallelism, same pattern as
arm4_orchestrator.py.
"""
import os
import subprocess
import sys
import time

CONDITIONS = ["100", "1.0", "0.5", "0.1", "natural"]
SEEDS = [0, 5]
FOLD = 0
ARMS = ["4", "5"]
N_WORKERS = 4
OUT_DIR = "results/angle_capture"


def all_combos():
    return [(arm, c, s, FOLD) for arm in ARMS for c in CONDITIONS for s in SEEDS]


def output_path(arm, condition, seed, fold):
    return f"{OUT_DIR}/arm{arm}_{condition}_{seed}_{fold}.npz"


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
            arm, cond, seed, fold = pending.pop(0)
            p = subprocess.Popen(
                [sys.executable, "-u", "scripts/capture_angles_worker.py", arm, cond, str(seed), str(fold)],
                env=env,
            )
            running[p] = (arm, cond, seed, fold)
        for p in list(running):
            if p.poll() is not None:
                combo = running.pop(p)
                done += 1
                print(f"[{done}/{len(combos)}] {combo} done ({time.perf_counter()-start:.0f}s elapsed)")
        time.sleep(0.2)

    print(f"\nangle capture complete: {len(combos)} replicates in {time.perf_counter()-start:.1f}s")
