"""Resumable experiment runner for Arms 1 and 2. Appends to results/runs.csv
after every individual run (not batched at the end), and skips any
(arm, alpha, seed) combination already present in that file -- so a
killed/restarted run continues instead of redoing completed work.

Not the quantum grid (D-015's 40-run estimate) -- that's Arms 4/5, not built
yet. This is the classical-only Arm 1 + Arm 2 grid, cheap enough to run
directly (no parallelism needed).
"""
import os
import sys
import time

import pandas as pd

# Only for the kill/resume test: each real run takes ~10-30ms, too fast to
# reliably interrupt mid-grid. Set RUN_GRID_TEST_DELAY_SEC to add an
# artificial per-run delay so the kill test is actually observable. Unset
# (0) in normal use -- does not affect results, only wall-clock.
TEST_DELAY_SEC = float(os.environ.get("RUN_GRID_TEST_DELAY_SEC", "0"))
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

sys.path.insert(0, "scripts")
from aggregators import fedavg  # noqa: E402
from federated_loop import run_centralized, run_federated  # noqa: E402
from log_run import append_run, RUNS_CSV  # noqa: E402
from models import LogisticRegressionModel  # noqa: E402
from partitioner import dirichlet_partition  # noqa: E402
from preprocessing import get_preprocessed  # noqa: E402

N_FEATURES = 6
SEEDS = [0, 1, 2, 3, 4]
ALPHAS = [100, 1.0, 0.5, 0.1]
CENTRALIZED_EPOCHS = 200
FEDERATED_ROUNDS = 20
LOCAL_EPOCHS = 5  # D-005


def _norm(v) -> str:
    """Normalize a key component (arm/alpha/seed) to a comparable string,
    so int/float/blank representations read back from CSV don't cause a
    false 'not done yet' and redo completed work."""
    if v is None or v == "" or (isinstance(v, float) and pd.isna(v)):
        return ""
    try:
        return str(float(v))
    except (TypeError, ValueError):
        return str(v)


def already_done() -> set:
    try:
        df = pd.read_csv(RUNS_CSV)
    except FileNotFoundError:
        return set()
    return {(_norm(r.arm), _norm(r.alpha), _norm(r.seed)) for r in df.itertuples()}


def evaluate(model, X_test, y_test) -> dict:
    proba = model.predict_proba(X_test)
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(accuracy_score(y_test, pred), 4),
        "f1": round(f1_score(y_test, pred), 4),
        "auroc": round(roc_auc_score(y_test, proba), 4),
    }


def run_arm1(seed: int, done: set):
    key = ("arm1", "", _norm(seed))
    if key in done:
        print(f"skip arm1 seed={seed} -- already in runs.csv")
        return
    _, _, X_train, y_train, X_test, y_test = get_preprocessed(seed)
    start = time.perf_counter()
    model = run_centralized(
        lambda: LogisticRegressionModel(N_FEATURES, seed=seed), X_train, y_train, epochs=CENTRALIZED_EPOCHS
    )
    wall = round(time.perf_counter() - start, 3)
    metrics = evaluate(model, X_test, y_test)
    append_run({"arm": "arm1", "alpha": "", "seed": seed, "wall_clock_sec": wall, **metrics})
    print(f"arm1 seed={seed}: {metrics} ({wall}s)")


def run_arm2(alpha: float, seed: int, done: set):
    key = ("arm2", _norm(alpha), _norm(seed))
    if key in done:
        print(f"skip arm2 alpha={alpha} seed={seed} -- already in runs.csv")
        return
    df_train, _, X_train, y_train, X_test, y_test = get_preprocessed(seed)
    client_idx, _ = dirichlet_partition(df_train, alpha, seed=seed)
    pos = df_train.index.get_indexer  # label -> positional row in X_train/y_train
    client_data = [(X_train[pos(idx)], y_train[pos(idx)]) for idx in client_idx]

    start = time.perf_counter()
    model = run_federated(
        lambda: LogisticRegressionModel(N_FEATURES, seed=seed),
        fedavg,
        client_data,
        rounds=FEDERATED_ROUNDS,
        local_epochs=LOCAL_EPOCHS,
    )
    wall = round(time.perf_counter() - start, 3)
    metrics = evaluate(model, X_test, y_test)
    append_run(
        {
            "arm": "arm2",
            "alpha": alpha,
            "seed": seed,
            "wall_clock_sec": wall,
            "rounds_to_converge": FEDERATED_ROUNDS,
            **metrics,
        }
    )
    print(f"arm2 alpha={alpha} seed={seed}: {metrics} ({wall}s)")


if __name__ == "__main__":
    done = already_done()
    for seed in SEEDS:
        run_arm1(seed, done)
        done = already_done()
        if TEST_DELAY_SEC:
            time.sleep(TEST_DELAY_SEC)
    for alpha in ALPHAS:
        for seed in SEEDS:
            run_arm2(alpha, seed, done)
            done = already_done()
            if TEST_DELAY_SEC:
                time.sleep(TEST_DELAY_SEC)
    print("\ngrid complete.")
