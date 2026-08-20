# Running This Project

Exact command, expected output, and failure signature for every component.
Updated as components are built — see `docs/labbook.md` for narrative session
history and `docs/decisions.md` for why things are built this way.

All commands assume the repo root as the working directory and the project
venv active (or invoke `.venv/Scripts/python.exe` directly on Windows, as
below).

---

## Environment check

**Command:**
```
.venv/Scripts/python.exe scripts/verify_env.py
```

**Expected output:** ends with `ENV CHECK: OK`. Device class line must contain
`pennylane_lightning...LightningQubit` — if it says `default.qubit` instead,
the lightning backend isn't actually being used even though the device name
string still says `lightning.qubit`.

**Failure signature:**
- `ModuleNotFoundError: No module named 'pennylane_lightning'` — venv wasn't
  activated, or install step was skipped.
- Assertion on device class failing — silent fallback to `default.qubit`;
  reinstall `pennylane-lightning` for this Python/platform combination.
- Assertion `adjoint gradient is all zero` — something is wrong with the
  circuit definition (e.g. no trainable gates reachable from the output).

---

## Data loader

**Command:**
```
.venv/Scripts/python.exe scripts/data_loader.py
```

**Expected output:** `loaded 920 records from 4 sites`, followed by a
per-site missingness table and a per-site class balance table.

**Failure signature:**
- `loaded` count != 920 — one of the four `data/raw/processed.*.data` files
  is missing or truncated; check `data/raw/`.
- `FileNotFoundError` — not running from the repo root, or `data/raw/` was
  moved.

**Companion diagnostic — feature retention check:**
```
.venv/Scripts/python.exe scripts/feature_retention_check.py
```
Applies a stated missingness threshold to every raw column and reports
keep/drop per column against `data/raw` site-level missingness. Currently
shows only 4 columns survive a strict 15%-missing threshold applied
consistently — see `docs/decisions.md`, 2026-08-18 ("Feature retention rule
consistency check"), open question, not yet resolved.

---

## Dirichlet partitioner

**Command:**
```
.venv/Scripts/python.exe scripts/partitioner.py
```

**Expected output:** natural 4-site partition table, then one table per
alpha in `{100, 1.0, 0.5, 0.1}` (row count + class balance per client),
ending with two `saved:` lines.

**Expected files:**
- `results/figs/partition_alpha_100.png`
- `results/figs/partition_alpha_1.0.png`
- `results/figs/partition_alpha_0.5.png`
- `results/figs/partition_alpha_0.1.png`
- `results/figs/partition_natural_vs_dirichlet.png`

**Failure signature:**
- `ValueError: array is read-only` — numpy/pandas version mismatch; the
  fix in place is `.to_numpy().copy()` before `rng.shuffle()`. If this
  reappears after a dependency upgrade, some other array in the function
  needs the same copy.
- A client with `n=0` or `n=1` at low alpha — not an error, this is the
  Dirichlet draw's expected behavior at low alpha, but worth checking
  against whatever minimum-client-size policy is eventually adopted (open
  question — no guard currently implemented, see `docs/decisions.md`).

---

## Circuit diagram (presentation asset)

**Command:**
```
.venv/Scripts/python.exe scripts/draw_circuit.py
```

**Expected output:** ASCII circuit diagram printed to console (may show `?`
in place of box-drawing characters on a Windows console — this is a console
encoding artifact, not a bug; the saved `.txt` file is correct UTF-8),
followed by two `written to` lines.

**Expected files:**
- `docs/circuit_diagram.txt`
- `docs/circuit_diagram.png` (dpi=220)

**Failure signature:**
- `UnicodeEncodeError` on the `print()` line — Windows console default
  codepage (cp1252) can't display the box-drawing characters PennyLane uses.
  Already handled for file output (`encoding="utf-8"` on the `open()` call);
  if it reappears, check that the fix wasn't reverted.

---

## Timing spike

**Command:**
```
.venv/Scripts/python.exe scripts/timing_spike.py
```

**Expected output:** 50 `round N/50  X.XXXs` lines, then total wall-clock,
avg per-round time, and two extrapolated-grid lines (3 seeds, 5 seeds).
Appends one row to `results/runs.csv` (`arm` column: `timing_spike_E<N>`).

**Current numbers (synthetic data, not the real preprocessed dataset):**
E=1 -> 232.1s/run. E=5 -> 1107.6s/run (~18.5 min). See `docs/labbook.md` for
the full history and grid-size implications.

**Failure signature:**
- `TypeError: grad.__init__() got an unexpected keyword argument 'argnum'` —
  this PennyLane version (0.45.1) uses `argnums` (plural), not `argnum`.
  Already fixed in this script; if writing new scripts against `qml.grad`,
  use `argnums`.
- Per-round time increasing steadily across rounds rather than staying flat
  — possible memory leak in the training loop; not observed yet, but the
  timing spike doesn't currently guard against it.

---

## Feature retention curve

**Command:**
```
.venv/Scripts/python.exe scripts/feature_retention_curve.py
```

**Expected output:** columns ranked by worst-site missingness, then survivor
counts at the 85/70/50% present benchmarks, ending with a `saved:` line.

**Expected files:**
- `results/figs/feature_retention_curve.png`

**Failure signature:** none observed yet. If the survivor counts at 70% and
50% ever differ, the "wide plateau" claim in `docs/decisions.md` (D-019) no
longer holds and needs re-checking before citing it.

---

## Run-level parallelism throughput test

**Command:**
```
.venv/Scripts/python.exe scripts/parallel_throughput_test.py
```

**Expected output:** `cpu_count: N`, then baseline single-run time, then two
batches of 4 concurrent runs (default env, then `OMP_NUM_THREADS`-capped),
each reporting batch wall-clock and per-run times, ending with a summary
block reporting speedup and efficiency for both conditions.

**Expected files:** `results/parallel_test/{baseline,parallel_default,parallel_capped}/*.json`
(one per worker process). Appends 3 rows to `results/runs.csv`
(`arm` column: `parallel_test_*`).

**Current numbers (this dev machine, 16 logical cores):** baseline 235.5s;
4 concurrent (default env) 260.8s wall-clock, 90% efficiency; 4 concurrent
(OMP_NUM_THREADS=4) 259.6s, 91% efficiency. See `docs/decisions.md` (D-020)
for the full writeup and grid-time implications. **Re-run on the actual
execution machine before trusting these numbers there** — cpu_count and
contention behavior are machine-specific.

**Failure signature:**
- A worker's `.json` file missing after the batch completes — that
  subprocess crashed silently; rerun with the subprocess's stdout/stderr
  inspected directly (currently not captured separately per-process, they
  share the parent's stdout).
- `OMP_NUM_THREADS`-capped batch running *slower* than default — would
  suggest lightning.qubit needs more threads than `cpu_count // n_workers`
  at whatever circuit size is being tested; not observed at 6 qubits.

---

## Preprocessing

**Command:**
```
.venv/Scripts/python.exe scripts/preprocessing.py
```

**Expected output:** train/test row counts (736/184), `X_train`/`X_test`
shape and range (should be `[0.000, ~3.14]` for train; test can exceed pi
slightly -- expected, see below), and class balance for both splits.

**Failure signature:**
- `X_train` range not starting at exactly `0.000` or exceeding the
  expected `[0, pi]` band by a lot (not just slightly, on test) — scaler
  fit/transform order is probably wrong, check `fit_transform_train` is
  only ever called on `df_train`.
- Test range noticeably higher than train's (e.g. `> 4` or `< 0`) —
  something in the imputer/scaler is not actually fit-on-train-only
  anymore; this is the leakage check for this component specifically.

---

## Partitioner (with minimum-client-size guard)

**Command:**
```
.venv/Scripts/python.exe scripts/partitioner.py
```

**Expected output:** guard config line, natural 4-site table, then one
table per alpha noting whether the guard fired, ending with two `saved:`
lines. At the locked alpha grid (seed=0), the guard should not need to
fire — all client sizes stay above 15.

**Failure signature:**
- `RuntimeError: could not satisfy min_client_size=...` — alpha is too
  extreme for the client count / dataset size (observed at alpha=0.01,
  which is not in the real grid — if this fires on one of {100, 1.0, 0.5,
  0.1}, that's a real problem, not expected behavior).

---

## Federated loop, models, aggregators (Arm 1 + Arm 2)

**Command:**
```
.venv/Scripts/python.exe scripts/run_grid.py
```

**Expected output:** one line per `(arm, alpha, seed)` combination, either
`skip ... -- already in runs.csv` or the metrics dict + wall-clock,
ending with `grid complete.`. Appends to `results/runs.csv` after every
individual run (not batched) — safe to kill and restart at any point.

**To re-verify the resume behavior** (tested once, see
`docs/decisions.md` D-023): set `RUN_GRID_TEST_DELAY_SEC=1` (or higher),
launch in the background, kill the **actual** `python.exe` process (not
just the shell job — under git-bash on Windows, `$!` can report the wrong
PID; find the real one with `ps aux | grep python`), then rerun without
the env var and confirm it skips everything already logged.

**Failure signature:**
- A `(arm, alpha, seed)` combination appears more than once in
  `results/runs.csv` — the `_norm()` key-matching in
  `scripts/run_grid.py:already_done()` isn't matching types consistently
  (e.g. `100` vs `100.0`); check `_norm()` first.
- `run_arm2` raising a shape mismatch — usually means
  `df_train.index.get_indexer(idx)` returned `-1` for some index value,
  i.e. the partition's row labels don't all exist in `df_train` (a
  partitioner/preprocessing mismatch, should not happen if both are
  called with the same `df_train`).

---

## Diagnostic sweep (5-fold CV, divergence, per-client/worst-client, natural vs. Dirichlet)

**Command:**
```
.venv/Scripts/python.exe scripts/run_diagnostic.py
.venv/Scripts/python.exe scripts/plot_diagnostic.py
```

**Expected output:** `run_diagnostic.py` prints one `seed N done (Xs elapsed)`
line per seed, ending with a summary line (total time, training counts).
Takes ~30s sequentially — do not parallelize this, see
`docs/diagnostic_report.md` Section 8 for why. `plot_diagnostic.py` prints
two `saved:` lines.

**Expected files:**
- `results/diagnostic_results.csv` (long format: one row per
  seed/fold/model/arm/condition/client)
- `results/diagnostic_divergence.csv` (one row per
  seed/fold/model/condition/round)
- `results/figs/client_divergence.png`
- `results/figs/worst_client_accuracy.png`

**Failure signature:**
- `KeyError` on a `(model, alpha)` tuple in `plot_diagnostic.py` — the
  `condition` column in the CSVs is read back as strings (`"100"`,
  `"1.0"`, ...) because it's a mixed int/float/string column including
  `"natural"`. `ALPHA_ORDER` in `plot_diagnostic.py` must be strings, not
  numbers — this broke once already, see the fix in that file's history.
- Any per-client row with `n=0` for a condition other than the extreme
  end of the sweep — would mean a client was dropped from a test fold
  entirely; not observed in the actual run (checked explicitly, see
  `docs/diagnostic_report.md`), would be a real bug if it recurs.

---

## Arm 4 sanity check (run this before trusting any Arm 4 accuracy number)

**Command:**
```
.venv/Scripts/python.exe scripts/sanity_check_vqc.py
```

**Expected output:** client sizes line, 15 `round N/15  loss=X.XXXX` lines
with loss decreasing, ending with a total-time line and a `saved:` line.

**Expected files:** `results/figs/arm4_sanity_loss_curve.png`

**Failure signature:**
- Loss flat or not decreasing — barren plateau or gradients not flowing.
  Per `CLAUDE.md`'s validation gate: do not trust any Arm 4 accuracy
  number until this is fixed. Do not proceed to the full sweep.

---

## Arm 4 / Arm 5 full sweep (VQC + FedAvg / VQC + circular-mean)

**Command:**
```
.venv/Scripts/python.exe -u scripts/arm4_orchestrator.py   # or arm5_orchestrator.py
.venv/Scripts/python.exe scripts/merge_arm4_results.py     # (or a merge_arm5_results.py analog)
```

**Cost:** this is compute-bound, unlike the classical arms. Measured:
~518s mean per replicate, 250 replicates, ~9 hours wall-clock with 4-way
parallelism (near-perfect ~4.0x speedup observed). **Before running the
full sweep on a new machine, run `--estimate-only 4` first** and
extrapolate — do not assume the number above transfers; it didn't
transfer from the classical D-020 measurement either.

**Resumable by construction:** each replicate's output is
`results/arm{4,5}_partial/{condition}_{seed}_{fold}.json`. A worker skips
a replicate whose output file already exists. Killing the orchestrator
(or the machine) and rerunning the same command picks up exactly where
it left off — validated with a real kill test for the classical grid
(D-023) and by construction here (same mechanism, same guarantee).

**Progress while running:** `results/runs_arm{4,5}.csv` grows by one row
immediately after each replicate completes (not batched at the end) —
read it directly at any time, no need to wait for a completion
notification. Every worker subprocess runs with `PYTHONUNBUFFERED=1`.

**Expected files after a full run:**
- `results/runs_arm4.csv` / `results/runs_arm5.csv` (simple per-replicate
  summary rows, same schema as `results/runs.csv`)
- `results/arm{4,5}_partial/*.json` (250 files each, one per replicate,
  the resume markers)
- after running the merge script:
  `results/arm{4,5}_diagnostic_results.csv`,
  `results/arm{4,5}_diagnostic_divergence.csv` (long format, same schema
  as the classical `results/diagnostic_*.csv`, for direct comparison)

**Failure signature:**
- `WARNING: worker for ... exited without output` — a subprocess crashed;
  its `.json` was never written so it'll be retried on the next run,
  but the crash reason isn't captured (workers share the parent's
  stdout/stderr, not logged per-worker). Check by rerunning that single
  combo directly: `python scripts/arm4_worker.py <condition> <seed> <fold>`.

---

## Not yet built

Arm 3 (FedProx) doesn't exist yet — being built by a teammate on a
separate branch; do not edit `federated_loop.py`, `data_loader.py`,
`partitioner.py`, `docs/INTERFACE.md`, or any `*_vqc.py` file, per that
work split.

Also outstanding: `scripts/timing_spike.py` still uses 5 synthetic clients,
not yet updated to match the 4-client decision (D-017) — flagged in
`docs/decisions.md`, not yet fixed. Low priority now that the real Arm 2
grid (which does use 4 clients) has real timing data of its own.
