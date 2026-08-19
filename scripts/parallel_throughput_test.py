"""Run-level parallelism throughput test: does launching multiple
(arm, alpha, seed) runs as concurrent OS processes actually speed up the
grid, or does lightning.qubit's internal OpenMP threading oversubscribe the
machine's cores and cancel out the gain?

Measures: 1 baseline run, then 4 concurrent runs (default env), then 4
concurrent runs with OMP_NUM_THREADS capped to cpu_count // 4, so each
process doesn't try to grab every core.
"""
import json
import os
import subprocess
import sys
import time

OUT_DIR = "results/parallel_test"
N_WORKERS = 4


def run_batch(n_procs: int, env_overrides: dict, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    env = os.environ.copy()
    env.update(env_overrides)

    start = time.perf_counter()
    procs = [
        subprocess.Popen(
            [sys.executable, "scripts/single_run_worker.py", str(i), out_dir],
            env=env,
        )
        for i in range(n_procs)
    ]
    for p in procs:
        p.wait()
    wall_clock = time.perf_counter() - start

    per_run = []
    for i in range(n_procs):
        with open(f"{out_dir}/{i}.json") as f:
            per_run.append(json.load(f)["wall_clock_sec"])
    return wall_clock, per_run


if __name__ == "__main__":
    cpu_count = os.cpu_count()
    print(f"cpu_count: {cpu_count}\n")

    print("=== baseline: 1 run, no contention ===")
    baseline_wall, baseline_per_run = run_batch(1, {}, f"{OUT_DIR}/baseline")
    baseline_time = baseline_per_run[0]
    print(f"baseline single-run time: {baseline_time:.2f}s\n")

    print(f"=== {N_WORKERS} concurrent runs, default threading env ===")
    par_wall, par_per_run = run_batch(N_WORKERS, {}, f"{OUT_DIR}/parallel_default")
    print(f"batch wall-clock: {par_wall:.2f}s")
    print(f"per-run times under contention: {[round(t, 2) for t in par_per_run]}\n")

    threads_per_proc = max(1, cpu_count // N_WORKERS)
    print(f"=== {N_WORKERS} concurrent runs, OMP_NUM_THREADS={threads_per_proc} per process ===")
    capped_wall, capped_per_run = run_batch(
        N_WORKERS, {"OMP_NUM_THREADS": str(threads_per_proc)}, f"{OUT_DIR}/parallel_capped"
    )
    print(f"batch wall-clock: {capped_wall:.2f}s")
    print(f"per-run times under contention: {[round(t, 2) for t in capped_per_run]}\n")

    sequential_equivalent = baseline_time * N_WORKERS
    print("=== summary ===")
    print(f"sequential-equivalent time for {N_WORKERS} runs (baseline x {N_WORKERS}): {sequential_equivalent:.2f}s")
    print(
        f"actual wall-clock, {N_WORKERS} concurrent, default env: {par_wall:.2f}s "
        f"(speedup {sequential_equivalent / par_wall:.2f}x, efficiency {100 * sequential_equivalent / par_wall / N_WORKERS:.0f}%)"
    )
    print(
        f"actual wall-clock, {N_WORKERS} concurrent, OMP_NUM_THREADS={threads_per_proc}: {capped_wall:.2f}s "
        f"(speedup {sequential_equivalent / capped_wall:.2f}x, efficiency {100 * sequential_equivalent / capped_wall / N_WORKERS:.0f}%)"
    )
