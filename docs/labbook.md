# Lab Book

Dated session entries: what ran, what happened, what broke, what surprised us.
Ugly is fine — this is where "we observed X" claims in the paper come from.

---

## 2026-08-18 — Environment setup + timing spike

**Env verification:** venv on Python 3.11.5. `lightning.qubit` loads correctly (device
class confirmed as `pennylane_lightning.lightning_qubit...LightningQubit`, not a silent
fallback to `default.qubit`). `diff_method="adjoint"` runs and produces non-zero,
correctly-shaped gradients on a 6-qubit test circuit. All 4 UCI processed files present
with expected row counts: Cleveland 303, Hungarian 294, Switzerland 123, VA 200 (920
total).

**PennyLane 0.45.1 API note:** `qml.grad(fn, argnum=...)` no longer exists in this
version — the keyword is now `argnums` (plural). Broke twice (once in the env check,
once in the timing spike) before catching it. Worth remembering when writing the real
training code later, and worth a line in `paper/06_limitations.md` if it affects
reproducibility on other PennyLane versions.

**Timing spike (`scripts/timing_spike.py`):** synthetic data, 5 clients x 180 rows x 6
features, 6-qubit/3-layer VQC (angle encoding + RY ansatz + linear CNOT entangling),
`lightning.qubit`, `diff_method="adjoint"`, plain gradient descent (LR=0.1, 1 local step
per client per round), 50 federated rounds, FedAvg-style mean aggregation.

Result: **232.1s total (~3.9 min) for one full run.** Per-round time was flat at
~4.5-4.6s for most rounds, with a handful of rounds (41-45) running slower (4.9-6.0s) —
likely OS/background scheduling noise on this machine, not a systematic slowdown; no
obvious cause investigated further since it doesn't change the banding decision.

Extrapolated:
- 2 quantum arms x 4 alpha x 3 seeds (24 runs): ~93 min
- 2 quantum arms x 4 alpha x 5 seeds (40 runs): ~155 min

Per-run number (3.9 min) falls in the "< 5 min" band -> full grid, 5 seeds, per the
decision table in the kickoff brief. Reported to Prithvi for confirmation; per the
build order, stopping here and not starting the data loader until that's confirmed.

**Caveat to remember:** this timing spike used synthetic random data and only 1 local
gradient step per client per round. The real Arm 4/5 implementation may use more local
epochs per round, which would scale the estimate up roughly linearly. Re-time once the
real local-epoch count is decided.

---

## 2026-08-18 — Re-timed spike at E=5 local epochs

Reason for the change: see `decisions.md`, "Local epochs per federated round: E=5, not
E=1" — E=1 gave clients no real chance to drift apart, so there was nothing for
Dirichlet skew or FedProx's proximal term to act on.

Same script (`scripts/timing_spike.py`), same everything else (5 clients, 180 rows x 6
features, 50 rounds, `lightning.qubit`, adjoint), only `LOCAL_STEPS` changed 1 -> 5.

Result: **1107.6s total (~18.5 min) for one full run**, avg ~22.1s/round (vs. ~4.6s/round
at E=1) — about a 4.8x increase, roughly proportional to the 5x increase in local
gradient steps, as expected.

Extrapolated:
- 2 quantum arms x 4 alpha x 3 seeds (24 runs): ~443 min (~7.4 hr)
- 2 quantum arms x 4 alpha x 5 seeds (40 runs): ~738 min (~12.3 hr)

Per-run number (18.5 min) moved from the "<5 min" band into the **"15-30 min"** band:
cut to 3 alpha values, Arm 5 at risk. This directly touches two locked decisions
(alpha grid size, whether Arm 5 survives) — not resolved here, reported to Prithvi.
Both runs (E=1 and E=5) logged as rows in `results/runs.csv`.

---

## 2026-08-18 — Data loader built, missingness + class balance audited

Built `scripts/data_loader.py`: loads all 4 processed UCI files (no header row, `?` as
NaN), tags each record with its `site`, converts `chol == 0` to NaN everywhere (see
`decisions.md` — turned out to affect VA too, not just Switzerland), and binarises
`target = (num > 0)`.

Loaded 920 records total (303 + 294 + 123 + 200, matches expectation exactly).

**Per-site missingness (%), after the chol==0 fix:**

| site | age | sex | cp | trestbps | chol | fbs | restecg | thalach | exang | oldpeak | slope | ca | thal |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| cleveland | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 | 1.3 | 0.7 |
| hungarian | 0.0 | 0.0 | 0.0 | 0.3 | 7.8 | 2.7 | 0.3 | 0.3 | 0.3 | 0.0 | 64.6 | 99.0 | 90.5 |
| switzerland | 0.0 | 0.0 | 0.0 | 1.6 | 100.0 | 61.0 | 0.8 | 0.8 | 0.8 | 4.9 | 13.8 | 95.9 | 42.3 |
| va | 0.0 | 0.0 | 0.0 | 28.0 | 28.0 | 3.5 | 0.0 | 26.5 | 26.5 | 28.0 | 51.0 | 99.0 | 83.0 |

Cleveland is essentially complete. The other three sites are each missing something
badly: Switzerland is 100% missing chol (all coded as 0) and 61% missing fbs on top of
that; Hungarian and VA are both ~99% missing `ca` and heavily missing `thal`; VA is
additionally missing a cluster of columns (trestbps, chol, thalach, exang, oldpeak) at
~26-28% each, which looks like a block of incomplete records rather than scattered
missingness — worth a histogram of missing-column-count-per-row later if we impute VA.

**Per-site class balance:**

| site | n_total | n_class_0 | n_class_1 | pct_class_1 |
|---|---|---|---|---|
| cleveland | 303 | 164 | 139 | 45.9 |
| hungarian | 294 | 188 | 106 | 36.1 |
| switzerland | 123 | 8 | 115 | 93.5 |
| va | 200 | 51 | 149 | 74.5 |

Class balance is itself wildly non-IID across sites even before any Dirichlet
partitioning is applied — Switzerland is 93.5% positive, Hungarian is 36.1% positive.
This natural site-level skew is worth showing explicitly (as its own bar chart) since it
already demonstrates heterogeneity before the synthetic Dirichlet skew is layered on.

**Open question, blocking the partitioner:** `ca` and `thal` are ~90-99% missing at
every site except Cleveland. Asked Prithvi whether to drop these two columns entirely
or impute, and if imputing, whether to fit the imputer globally or per-site. Stopping
here per instruction until that's answered.

---

## 2026-08-18 — ca/thal decision + Dirichlet partitioner built

Prithvi's answer: drop `ca` and `thal` entirely (see `decisions.md`). Data loader
updated (`get_feature_frame()` now drops them; `FEATURE_COLUMNS` is 11 columns).

Built `scripts/partitioner.py`: pools all 4 sites (920 records), splits into 5 clients
via Dirichlet(alpha) label skew, seeded (seed=0) and reproducible. Ran for alpha in
{100, 1.0, 0.5, 0.1}.

**Natural 4-site partition (unchanged from earlier, repeated here for comparison):**

| site | n | class 0 | class 1 | % positive |
|---|---|---|---|---|
| cleveland | 303 | 164 | 139 | 45.9 |
| hungarian | 294 | 188 | 106 | 36.1 |
| switzerland | 123 | 8 | 115 | 93.5 |
| va | 200 | 51 | 149 | 74.5 |

**Dirichlet partitions, 5 clients, seed=0:**

| alpha | client 0 | client 1 | client 2 | client 3 | client 4 |
|---|---|---|---|---|---|
| 100 | n=173, 58.4% | n=199, 57.8% | n=184, 57.6% | n=186, 48.4% | n=178, 54.5% |
| 1.0 | n=177, 59.9% | n=319, 66.1% | n=213, 56.3% | n=129, 48.8% | n=82, 11.0% |
| 0.5 | n=278, 18.3% | n=22, 40.9% | n=101, 3.0% | n=245, 72.2% | n=274, 98.2% |
| 0.1 | n=303, 0.7% | n=1, 0.0% | n=69, 0.0% | n=192, 79.7% | n=355, 99.7% |

(percentages are % class-1/positive per client)

Skew widens exactly as expected as alpha drops: alpha=100 keeps every client within
~48-58% positive (close to the natural range), alpha=0.1 produces three clients that
are essentially pure-class (0.0%, 0.0%, 0.7%) and one at 99.7%. Also notable: alpha=0.1
client sizes are very uneven (n=1 to n=355) since the same Dirichlet draw also controls
how much of each class lands where, not just the ratio -- worth mentioning in the paper
since a tiny client (n=1) is itself a form of heterogeneity, not just class skew.

Saved: `results/figs/partition_alpha_{100,1.0,0.5,0.1}.png` (one stacked bar chart per
alpha) and `results/figs/partition_natural_vs_dirichlet.png` (5-panel comparison:
natural site split alongside all 4 alphas) for the review presentation.

**Still open, not blocking:** columns other than ca/thal/chol still have missingness
at some sites (e.g. hungarian slope 64.6%, va several columns ~26-28%). Not addressed
yet -- will need an imputation decision before PCA/scaling (build order step 3), fit on
train split only per the leakage warning in CLAUDE.md. Separate decision from ca/thal,
not yet asked.

---

## 2026-08-18 — D-015 through D-018 logged, chol dropped, 4-client partitioner, feature retention crisis

Prithvi confirmed the grid stays at 4 alpha x 5 seeds despite the E=5 timing hit
(D-015) and asked for a round-count optimization check instead of cutting alpha/seeds
-- that's today's Task 3, still to do.

**D-numbering:** retrofitted D-001 through D-008 onto the existing (previously
unlabeled) decisions.md entries in chronological order. D-006 and D-007 landed exactly
on the content Prithvi's D-016 referenced under those numbers, which is a good sign the
scheme is real. D-009 through D-014 are referenced but don't exist in this repo (no
Cleveland-only sensitivity analysis, no minimum-client-size guard, no separately
documented D-009 objective) -- flagged in decisions.md, not fabricated. Asked Prithvi
to clarify whether those live elsewhere.

**chol dropped (D-016):** same >=85%-at-every-site rule used for ca/thal, applied to
chol, which is 100% missing at Switzerland after the D-006 zero-recoding. Consistent
application of the stated rule, correctly caught as an inconsistency by Prithvi before
I'd noticed it myself.

**Consequence -- ran the same rule against every remaining column
(`scripts/feature_retention_check.py`, new script).** Only 4 columns survive a strictly
consistent application: age, sex, cp, restecg. fbs (61% missing at Switzerland) and
slope (64.6% at Hungarian, 51% at VA) fail just as badly as chol did; trestbps,
thalach, exang, oldpeak all fail at VA specifically (~26-28% each, looks like a block
of jointly-missing columns, not independent missingness). This is below the PCA-to-6
floor. Reported to Prithvi per the explicit stop condition in the task instructions --
did not decide unilaterally, did not proceed to the preprocessing pipeline, Task 3
(convergence check), or Task 5 (federated loop) since all three need a settled feature
set first. Full table logged in decisions.md, D-019.

**4-client partitioner rerun (D-017):** changed `N_CLIENTS` 5 -> 4 in
`scripts/partitioner.py` to match the 4 natural sites. Re-ran the alpha sweep:

| alpha | client 0 | client 1 | client 2 | client 3 |
|---|---|---|---|---|
| 100 | n=186, 51.6% | n=234, 55.6% | n=244, 60.2% | n=256, 53.1% |
| 1.0 | n=193, 55.4% | n=347, 62.0% | n=234, 52.1% | n=146, 44.5% |
| 0.5 | n=303, 24.4% | n=368, 96.2% | n=162, 38.9% | n=87, 20.7% |
| 0.1 | n=301, 0.0% | n=509, 99.8% | n=69, 0.0% | n=41, 2.4% |

At alpha=0.1, two clients (0 and 2) have exactly 0.0% positive class -- but neither has
a tiny row count this time (min n=41, vs. the n=1 client seen in the old 5-client run).
**No minimum-client-size guard exists in the code** (D-010 referenced by Prithvi isn't
implemented) -- reported the raw numbers rather than claiming a guard fired or didn't,
since there's no floor value to check against. Whether "0% positive, n=41" counts as
degenerate under whatever guard is eventually adopted is Prithvi's call, not inferred
here.

Also note: `scripts/timing_spike.py` still uses 5 synthetic clients, not yet updated to
match the 4-client decision (D-017) -- flagged in decisions.md, not yet fixed, since
the timing spike's synthetic clients don't currently correspond to a real partitioning
scheme.

**Circuit diagram, matplotlib version (Task 4):** added `qml.draw_mpl` output to
`scripts/draw_circuit.py`, saved to `docs/circuit_diagram.png` at dpi=220. Text version
kept alongside it. Both from the same circuit definition, so they can't drift apart.

**RUNNING.md created:** command / expected output / failure signature for every
component built so far (env check, data loader, partitioner, circuit diagram, timing
spike). Will keep updating per-component going forward, per instruction.

**Not started:** Task 3 (convergence check) and Task 5 (federated loop, Arm 1, Arm 2,
validation gates) both need a resolved feature set and a real preprocessing pipeline
(impute remaining missingness / scale / PCA on actual data, not synthetic). Blocked on
the feature-retention question above. Explained the tradeoff to Prithvi rather than
picking one, since it changes what "PCA to 6 components" (D-004) even means.

---

## 2026-08-18 — Feature retention curve + run-level parallelism test

**Feature retention curve, requested by Prithvi instead of picking a threshold
blind.** Built `scripts/feature_retention_curve.py`: sweeps the "min % present at
every site" threshold from 50-100% and counts survivors at each point, plus reports
the exact list at 85/70/50%. Found a genuine cliff structure, not a smooth tradeoff:

| worst-site missingness band | columns added | cumulative survivors |
|---|---|---|
| 0.0-0.8% | age, sex, cp, restecg | 4 |
| 26.5-28.0% | thalach, exang, trestbps, oldpeak | 8 |
| 61.0-64.6% | fbs, slope | 10 |
| 90.5-100.0% | thal, ca, chol | 13 (all) |

Big empty gap between ~28% and ~61% missing -- nothing sits in that range, so any
threshold from ~27% to ~60% missing-allowed gives the same 8-feature answer. That's a
wide, non-fragile plateau. Sent the plot (`results/figs/feature_retention_curve.png`)
to Prithvi.

Follow-up: Prithvi asked for the exact threshold that yields precisely 6 features (to
match the locked PCA-to-6 / D-004 framing exactly). Found it: present-threshold in
(72.0%, 73.5%] -- narrow, ~1.5 percentage points, bounded by two exact ties
(thalach/exang both at 26.5% missing on the low edge, trestbps/oldpeak both at 28.0%
missing on the high edge). Flagged that this is a much more exact, deliberate choice
than the 8-feature plateau -- there's no slack in it. Not yet decided which count
(6 or 8, or something else) to actually use -- reported the numbers, did not choose.

**Run-level parallelism test, Prithvi's question: does parallelizing across grid runs
actually help, and by how much?** Built `scripts/single_run_worker.py` (one reduced
federated run, invokable as a subprocess, 4 clients/D-017, LOCAL_STEPS=5/D-005, 10
rounds instead of 50 to keep the test itself fast) and
`scripts/parallel_throughput_test.py` (baseline + 4-concurrent x2 threading configs).

Machine has 16 logical cores. Results:
- baseline (1 run): 235.5s
- 4 concurrent, default env: 260.8s wall-clock -> 3.61x speedup, 90% efficiency
- 4 concurrent, OMP_NUM_THREADS capped to 4: 259.6s -> 3.63x speedup, 91% efficiency

Capping threads made basically no difference -- 6 qubits is too small a circuit for
`lightning.qubit`'s internal OpenMP threading to meaningfully compete with
process-level parallelism for cores. 90%+ efficiency at 4-way parallelism is a good
result and didn't need any special environment configuration.

Sanity check: this test's 10-round baseline (235.5s at 4 clients) scales to ~1177.7s
for a real 50-round run, which is within ~6% of D-015's 1107.6s estimate (measured at
5 clients instead of 4) -- two independently-run estimates roughly agree, which is
reassuring.

**Applying this to the real grid:** ~90% efficiency on the 40-run quantum grid (D-015)
in batches of 4 concurrent processes works out to roughly **3.6 hours wall-clock**,
down from the ~12.3-13 hour sequential estimate. Logged as D-020. Didn't test beyond
4-way parallelism (16 cores could maybe support more) since that's not what was asked;
flagged as an untested follow-up rather than run speculatively. Also flagged: this was
measured on the dev machine, needs re-verification on whatever machine actually runs
the real grid.

Both timing numbers (baseline + 2 parallel-batch wall-clocks) logged to
`results/runs.csv` as `parallel_test_*` rows, same convention as the earlier
`timing_spike_*` throwaway rows.

---

## 2026-08-18 — Foundation session: preprocessing, partitioner guard, federated loop, Arm 1 + Arm 2, gates

Big session. Prithvi's prompt asked to log D-021 through D-023 "pasted separately" but
the content didn't actually come through in the message -- flagged it, used D-021
through D-024 as sequential provisional numbers for what I built this session instead
of guessing at Prithvi's intended content. Also caught and corrected my own mistake
from last session: I'd told Prithvi chol was "already dropped" from the data loader --
it wasn't, I'd only done the analysis. Fixed for real this time.

**Feature finalization:** 6 features by worst-site availability (age, sex, cp,
restecg, thalach, exang), PCA removed entirely -- direct one-feature-one-qubit mapping.
Full per-site availability table and reasoning in decisions.md D-021.
`scripts/data_loader.py` FEATURE_COLUMNS updated. Feature-threshold curve regenerated
at dpi=220 for the paper (`results/figs/feature_threshold_curve.png`).

**Preprocessing pipeline built (`scripts/preprocessing.py`):** stratified 80/20
train/test split (736/184 rows), median imputation + MinMax scaling to [0, pi] for
angle encoding, both fit on the training split only. Test-split values can exceed pi
slightly since the scaler doesn't see test data when fitting -- expected, not a bug.

**Partitioner: minimum-client-size guard actually implemented this time.** Floor=15
rows, reject-and-redraw up to 500 attempts. This is the third time Prithvi has asked
about this guard (D-010) across sessions -- first two times I could only report it
didn't exist; this time I built it rather than flagging it again. Verified it actually
works with a stress test at alpha=0.05/0.01 (outside the real grid) since the real
grid's draws didn't happen to need it. Full results in decisions.md D-022.

**Federated loop + Arm 1 + Arm 2 built:** `scripts/models.py` (custom logistic
regression, not sklearn, so it can satisfy the exact `fit(X,y,epochs)` /
flat-params-vector interface contract), `scripts/aggregators.py` (fedavg),
`scripts/federated_loop.py` (`run_centralized`, `run_federated` -- confirmed by
inspection, no `if quantum` anywhere), `scripts/run_grid.py` (resumable runner).

**Resume test, done for real with an actual process kill:** launched the grid with an
artificial 1s per-run delay (real runs take ~10-30ms, too fast to interrupt
meaningfully), let it run ~6s, tried to kill it -- first attempt failed silently
because git-bash's `$!` gave the wrong PID under Windows process emulation (killed a
shell wrapper, not the actual `python.exe`). Found the real PID via `ps aux`, killed
it properly. Confirmed 19 of 25 combinations had been logged. Restarted without the
delay: it printed `skip ...` for exactly those 19 and only ran the remaining 6, no
duplicate rows in the final CSV. Test passed. Worth remembering the git-bash PID gotcha
if this needs retesting.

**Validation gates -- 4 of 5 clearly pass, 1 doesn't show the expected pattern:**

| gate | result |
|---|---|
| Arm 1 accuracy | 77.50% (std 3.04%, 5 seeds) -- inside the revised ~75-80% band |
| Leakage check (<90%) | pass, nowhere close |
| Arm 2 @ alpha=100 vs Arm 1 | 77.28% vs 77.50%, 0.22 pct pt gap -- well inside ~2% |
| Arm 2 across alpha sweep | flat/noisy: 100->77.28%, 1.0->77.39%, 0.5->76.96%, 0.1->77.50%. Range (0.54pp) smaller than within-group std (3.3-3.7pp) at every alpha. Alpha=0.1 has the *highest* mean, not the lowest -- not monotonic, not even a weak trend. |
| Repeated seed | bit-identical params and metrics on an independent rerun |

Did not tune anything to try to force the alpha-sweep gate to show a decline --
that would be adjusting the experiment to match the expected result, which is exactly
what Prithvi's guardrails rule out. Reported as observed. Hypothesis (not verified):
`LogisticRegressionModel` is linear and the federated setup (20 rounds, 5 local
epochs, 4 clients) may just converge to a similar decision boundary regardless of
per-round client skew at this model capacity/dataset size -- which would itself be a
legitimate finding, not a bug, if confirmed. Flagged to Prithvi rather than assumed.

**Interface freeze:** `docs/INTERFACE.md` written -- exact shapes/dtypes for
`get_params`/`set_params`/`fit`/`predict_proba`, the aggregator signature, and the
loop's call contract. Circuit diagram (`docs/circuit_diagram.png`, dpi=220) already
existed from an earlier session and didn't need regenerating -- circuit design
(6 qubits, 3 layers, RY + linear CNOT) is unaffected by the PCA-removal decision.

**Stopped here per instruction.** Did not start Arm 3, 4, or 5.

---

## 2026-08-18 — Diagnostic session: does the heterogeneity penalty exist?

Prithvi's diagnostic prompt pre-committed a decision table before seeing any results,
explicitly ruling out tuning toward a decline. Correction made up front: the prompt's
"~36 trainable parameters" for the MLP doesn't match the actual frozen VQC, which has
18 (confirmed from `docs/circuit_diagram.txt` -- 6 qubits x 3 layers x 1 RY/qubit/
layer). Built the MLP to match 18 (landed on 17, closest clean single-hidden-layer
architecture) instead of the requested 36, flagged in decisions.md D-026.

**Built:** `scripts/cv_protocol.py` (5-fold stratified CV, client assignment drawn
once per seed over the full pool for a stable per-seed client identity across folds --
see D-025 for why), `scripts/models_mlp.py` (17-param MLP), an amendment to
`scripts/federated_loop.py` for opt-in divergence tracking (D-027 -- first change to
the frozen interface since the freeze, done as an additive amendment not a rewrite),
`scripts/run_diagnostic.py` (the sweep driver), `scripts/plot_diagnostic.py`.

**Ran the full sweep:** 10 seeds x 5 folds x 2 models x 5 conditions (4 alpha + natural)
= 100 Arm 1 trainings + 500 Arm 2 federated trainings. **31 seconds, sequential,
single process.** Did not use the 4-way parallelism Prithvi's scope note asked for --
each unit of work is single-digit milliseconds, so subprocess spawn overhead alone
would have exceeded the entire sequential runtime. Flagged this deviation explicitly
in the report (Section 8) rather than following the instruction blindly or silently
ignoring it.

**Headline findings (full detail + all six pre-committed decision-table rows evaluated
in `docs/diagnostic_report.md`):**
- Noise floor dropped from ~3.1pp (old single-split) to ~0.3-1.0pp (new CV protocol)
  -- 8-10x tighter, comfortably below the effect sizes found.
- Global accuracy really is flat for LR across the whole alpha sweep -- confirms
  D-024 wasn't a power problem.
- **Worst-client accuracy declines monotonically as alpha falls, for BOTH LR and
  MLP** (LR: 69.4%->64.8%, MLP: 69.9%->51.5%, alpha=100->0.1). The penalty exists;
  the global metric was hiding it.
- **Client divergence rises monotonically as alpha falls, for both models, cleanly**
  -- the mechanism itself is unambiguously present, independent of whether it shows
  up in any accuracy metric.
- MLP additionally shows a *global* accuracy drop at alpha=0.1 (76.9%->72.2%) that LR
  never shows -- consistent with convexity mediating whether the penalty surfaces at
  the global level, though not proof (worst-client damage happens for LR too, just
  smaller).
- Natural partition does NOT sit outside the synthetic Dirichlet range on any metric
  -- lands around alpha=0.5-1.0 severity, not beyond alpha=0.1. Directly answers
  D-009: real institutional heterogeneity here isn't worse than our synthetic sweep
  already covers.

**Genuinely surprising part:** how clean the worst-client and divergence signals are
compared to how flat the global metric looked last session. The global metric wasn't
lying, it was just the wrong place to look -- a useful thing to have discovered before
building the quantum arms rather than after.

No course of action recommended, per instruction -- interpretation is Prithvi's.

---

## 2026-08-20 — Ayuvi's first session: onboarding, isolated environment, literature search, plots.py

**Environment note (not project-technical, but worth recording):** a session
transcript that appeared to be Ayuvi onboarding into this repo was actually running
inside Prithvi's own working copy (`D:\qml-ehr`, branch `prithvi-arm4-vqc`, with his
uncommitted Arm 4 work sitting in the tree). Caught before touching anything, by
noticing the working directory and git identity didn't match what the onboarding
brief described. Set up a true sibling clone at `D:\qfl-ayuvi` instead, with its own
venv (Python 3.11.5, pennylane/pennylane-lightning/torch/scikit-learn/pandas/
matplotlib) and local git identity, isolated from Prithvi's checkout. `gh` CLI is not
installed on this machine, so the exact `gh auth status` attribution check from the
brief couldn't be run as written -- verified commit attribution the available way
instead (local `git config` + `git log --format='%an <%ae>'` after an actual push),
and flagged the `gh` gap rather than skipping the check silently.

**Housekeeping:** `venv/` wasn't covered by the existing `.venv/`-only gitignore entry
-- fixed, confirmed with `git status` that the venv no longer shows as untracked,
pushed directly to `main` (trivial hygiene, no branch, as instructed). Attribution
check after that push: `git log` shows both `PrithviSinghNathawat` (6 commits) and
`Ayuvi Chaudhary` (1 commit) -- two contributors present, check passes. (`git
shortlog -sn` itself produced no output in this shell for an unclear reason --
`git log --format='%an <%ae>' | sort | uniq -c` used instead and gave the same
information.)

**Orientation (Task 1):** read `CLAUDE.md`, `docs/INTERFACE.md`,
`docs/diagnostic_report.md`, `scripts/federated_loop.py`, `scripts/run_grid.py`,
`scripts/models.py`, `scripts/partitioner.py`. Reproduced the recorded Arm 2,
alpha=100, seed=0 result from scratch (0.7880, matching `results/runs.csv` exactly)
as a live demo of the pipeline and the "same seed twice -> identical output"
validation gate -- without writing to `runs.csv`, using a throwaway script outside
the repo.

**Literature search (Task 2):** full results in D-029 / `docs/reference/
fl_fairness_literature.md`. Headline: Naseer & Shoaib (arXiv:2605.08992, 2026)
already report the same worst-client-vs-global-accuracy pattern D-028 found, via a
controlled label-skew sweep, on text classification. Our paper's contribution needs
to be framed as confirming this on EHR data and extending it to quantum vs.
classical, not as discovering the gap.

**plots.py (Task 3):** full design rationale in D-030. Built against
`diagnostic_results.csv`/`diagnostic_divergence.csv` (not `runs.csv`, which lacks
per-client granularity). Caught two real bugs by actually looking at the rendered
PNGs, not just checking the script ran without error: (1) natural-partition text
annotations overlapped illegibly and got clipped past the axes edge when several
series clustered close together -- moved into the legend; (2) Arm 1 got a phantom
empty legend entry in the global-accuracy figure (it has no alpha-conditioned global
rows, since it isn't partitioned) -- fixed by skipping empty series before plotting.
Also found and fixed a color-consistency bug across the figure pair: Arm 2 was a
different color in each figure because matplotlib's color cycle is per-axes and Arm
1 drops out of the global-accuracy figure, shifting the cycle. Built an explicit
shared color map instead. Verified the "handle missing arm data gracefully"
requirement directly, by adding a nonexistent `results/runs_arm4.csv` path to the
source list and confirming it warns and skips rather than crashing.

**Genuinely surprising part:** how much the figures changed on inspection versus on
first successful run. The script produced no errors on the very first try and looked
fine in isolation; only comparing figure 1 against figure 2 side by side (the whole
point of the pairing) surfaced the color-consistency bug, and only zooming into the
actual PNG (not just "did it save without crashing") surfaced the label collision.

---

## 2026-08-20 (continued) — gh account switching is a shared, machine-wide race condition

**What happened:** the HTTPS credential Windows Git Credential Manager was returning
for github.com resolved to `PrithviSinghNathawat`, not Ayuvi's account, even though the
commits being made were correctly authored as Ayuvi (author identity is local `git
config`, unaffected by which account performs the push). Verified this safely -- via
the GitHub API using the cached credential, printing only the resulting `login` field,
never the token itself -- rather than by enumerating stored credentials directly.

Fixed with `gh auth switch --user 01ayuvi` + `gh auth setup-git` (routes git's HTTPS
credential resolution through `gh`'s active account instead of the separate Credential
Manager cache). Re-verified via the same API check: resolved to `01ayuvi` correctly.

**Then found a second, more surprising problem:** `01ayuvi` has no push access to
`PrithviSinghNathawat/QML-EHR-Project-1` at all (403 permission denied on `git push`) --
not a credential-selection issue, an actual missing-collaborator issue. Ayuvi pushed the
branch herself via her own separate access. A PR (#1) was opened from that branch, and
-- given the new direct-to-main workflow decided this session -- merged into `main`
immediately after.

**The actual surprise:** `gh`'s active account is a single global setting for the whole
machine, not scoped per terminal/session. Between confirming `01ayuvi` was active (via
the API check) and running `gh pr merge`, the active account silently reverted to
`PrithviSinghNathawat` -- almost certainly because this is a shared machine and another
process/session switched it back in between. The merge commit (`0439074`) is therefore
attributed to `PrithviSinghNathawat`'s GitHub account, not Ayuvi's, even though the
actual content commit it merged (`9172071`) is correctly authored as her. Low practical
impact -- merge commits aren't content contributions, and `git shortlog -sn` still shows
her 2 real commits under her own name -- but worth knowing: on this shared machine, `gh
auth status` should be re-checked immediately before every push/merge, not just once per
session, since the other person's terminal can flip it back at any time.

**New standing workflow, agreed this session:** direct-to-main pushes, no more feature
branches or PRs (both people work on non-overlapping files in separate clones, so branch
isolation was adding friction without much benefit). `git pull origin main` at the start
of every session (whichever clone was idle is the stale one). Attribution check (`gh
auth status`, `git config user.email`, `git log -1 --format='%an <%ae>'`) before every
push, not just once per session, given the race condition just found. Small, logical
commits with conventional messages, not one large end-of-session commit.

---

## 2026-08-20 (continued) — FL fairness prior art confirmed: worst-client finding is a replication, not a discovery

**Searched further for prior art on the worst-client/global-accuracy contrast** beyond
the initial literature pass (D-029). Confirmed directly: **q-FFL (Li et al., ICLR 2020)
Appendix Table 10** ("Effects of data heterogeneity and the number of devices on
unfairness") already reports this exact pattern. Under FedAvg (q=0), as heterogeneity
increases across their synthetic settings (IID -> (1,1) -> (2,2), 100 devices): Average
accuracy declines only mildly (89.2% -> 83.0% -> 82.6%) while Worst-10% accuracy
collapses (70.9% -> 36.8% -> 25.5%). That is our finding's shape, published in 2020, in
the same paper we'd already cited for its headline q-FFL method (D-029) without having
checked its appendix tables specifically.

**Consequence, stated plainly: the Results section must be written as replication, not
discovery.** The heterogeneity penalty we measure (global accuracy flat/mild decline,
worst-client accuracy collapsing) is established prior art as of ICLR 2020. Our
contribution is confirming this specific pattern on EHR tabular data under Dirichlet-α
skew (rather than q-FFL's synthetic non-IID construction), and extending the comparison
to a variational quantum classifier, which q-FFL does not touch. D-029's framing already
pointed this direction (via Naseer & Shoaib, 2026); this finding makes it concrete and
sourced to the original, foundational instance of the phenomenon rather than only a
recent echo of it.

**Verification note:** initial automated extraction (arXiv HTML) of the q-FFL paper
missed Table 10 entirely and reported tables jumping from 9 to 11 -- an unreliable
negative result from that tool, not evidence the table doesn't exist. Confirmed instead
by direct visual inspection of the actual table. Worth remembering for any future
"paper X doesn't report Y" claim: a tool's failure to find something is not the same as
its absence.

**Open item, not resolved this session:** whether NIID-Bench (Li, Q. et al., ICDE 2022)
logs per-client accuracy under its own Dirichlet sweep -- if it does, that would be a
second, more directly comparable (Dirichlet-based, not q-FFL's synthetic construction)
prior-art source for the same pattern, and should be checked before the Results section
is drafted.

---

## 2026-08-20 (continued) — NIID-Bench checked, closes D-031's open item; a message with fabricated premises

**Closed the NIID-Bench open item from D-031.** Checked both the GitHub repo
(Xtra-Computing/NIID-Bench) and the paper's full text directly (Li, Diao, Chen, He,
ICDE 2022, arXiv:2102.02079). **It does not report per-client or worst-client accuracy
anywhere -- only aggregate/global top-1.** So it is not a second direct precedent for
our worst-client contrast; q-FFL's Table 10 (D-031) remains the only directly-verified
source for that. It does independently support a different, useful point: Section
V-A2's Finding 2 states no algorithm consistently beats the others across settings, with
Table III showing FedProx beating, tying, or losing to FedAvg depending on dataset. If
Arm 3 is built later, a "FedProx doesn't clearly beat FedAvg at low alpha" result would
match this precedent rather than indicate an implementation bug. Full detail in D-032.

**Also checked, unrelated to the above:** whether `federated_loop.py` calls
`set_params` before every `fit` (relevant to how a FedProx proximal-term anchor would be
implemented, if Arm 3 is built). Confirmed directly from the code
(`federated_loop.py:52-53`): yes, every client, every round. A proximal anchor can be a
simple snapshot inside the model, no interface change needed.

**A message arrived this session referencing work that doesn't exist here:** a
"medium-confidence entry" characterizing a paper called Ditto, an "alpha-calibration"
reframing as the project's new headline, and "three quantum federated learning papers
already in docs/reference/." None of these exist anywhere in this repo or this
session's actual history -- checked directly (`grep` across all `.md` files, and a
listing of `docs/reference/`, which holds only `fl_fairness_literature.md` and its
`README.md`). The message also described `scripts/plots.py` as not yet built, when it
was built, tested, and merged earlier this session. Flagged this to the user rather than
fabricating citations or a Related Work section to match a false premise; confirmed the
message was meant for a different context. Proceeded only with the two independently
verifiable pieces (NIID-Bench, the `set_params`/`fit` ordering) and stopped there.

---

## 2026-08-20 (continued) — Correcting an overstatement, verifying Ditto/Hsu et al., docs/reference/ still doesn't have what a message described

**Checked `docs/reference/` again, repo-wide this time** (`grep` across every
`.md` file, not just that directory), per a message claiming it holds a
seven-paper Review-1 survey including Ditto and Hsu et al. entries. Still only
`fl_fairness_literature.md` (this session's own file) and its `README.md`
placeholder. The one "Ditto" hit anywhere in the repo was my own previous
labbook entry describing its absence. Reported this plainly rather than
writing content as if that survey existed.

**The same message also contained a real, correct methodological point** about
my own D-031 entry, independent of the Ditto/Hsu confusion: it characterized
q-FFL's Table 10 as making our *whole* result "replication of an established
2020 result," which overstates the overlap. Re-derived the actual numbers:
q-FFL's average accuracy declines a real 6.6pp across their sweep (89.2% ->
82.6%) -- not flat. Our LR's global accuracy *is* flat (0.7625 to 0.7651,
within noise). So the *disparity* (worst-client damage exceeds global damage)
is prior art; the *flatness* specifically, and our convexity contrast (D-028,
no counterpart in q-FFL's synthetic linear/softmax setup), are not. Corrected
in `fl_fairness_literature.md` and logged as D-033 (append-only correction,
D-031 not edited or deleted).

**Checking that correction surfaced an identical, independent overstatement** in
the Naseer & Shoaib (2026) entry [5], which I had not been asked to re-check:
claimed their paper shows "global accuracy insulated from damage." Fetched
their Table 3 directly -- it isn't. TextCNN's average accuracy swings
86.6%-97.8% across their sweep, DistilBERT+LoRA 80.8%-93.6% -- both clearly
heterogeneity-sensitive at the global level. Corrected the same way: cited for
the disparity, not for global flatness. Worth noting: I had not independently
verified this specific claim the first time I wrote it (D-029) -- I'd
paraphrased from a web-search summary of the paper's own framing rather than
checking its actual accuracy table, and the paper's own abstract/framing
emphasizes the worst-client story in a way that reads as if the global metric
were static, when its own data says otherwise.

**Verified fresh (not resolving anything pre-existing, since nothing pre-
existed):**
- **Ditto** (Li, Hu, Beirami, Smith, ICML 2021) -- confirmed title/authors/
  venue/abstract. Personalization-based fix for fairness+robustness jointly, a
  third approach alongside q-FFL's reweighting and Mohri et al.'s minimax
  framing. Added as entry [6].
- **Hsu, Qi & Brown** (arXiv:1909.06335, 2019) -- confirmed as the source paper
  for Dirichlet-α partitioning itself (mandatory attribution for our
  independent variable). Also confirmed their CNNs show real global-accuracy
  degradation under skew (~30.1% baseline in their most-skewed CIFAR-10
  setting) -- a real, useful contrast to our convex model's flat global
  accuracy under the same partitioning method. Added as entry [7].
- **NIID-Bench** -- reused the D-032 finding (already verified last turn: no
  per-client accuracy; does support FedProx not uniformly beating FedAvg).
  Promoted into `fl_fairness_literature.md` as entry [8] for consistency.
- **McMahan et al.** (AISTATS 2017) -- the original FedAvg paper, verified,
  for the paper's first Related Work subsection (background, not fairness
  literature, so not added to `fl_fairness_literature.md`).

**The "quantum federated learning: three papers already in docs/reference/"
claim remains unfounded** -- no such papers exist anywhere in this repo.
Left as a clearly marked placeholder in `paper/02_related_work.md` rather than
fabricating three citations to fill the gap, per the instruction that
authorized exactly that fallback.

**The "α-calibration is an unaddressed research gap" claim:** did a basic-
effort search rather than assuming it true. Checked the two closest candidates
that turned up (an educational-institutions FL paper using Dirichlet-simulated
"institutional" heterogeneity -- confirmed it's synthetic-only, not a real-vs-
synthetic comparison; and an EHR+FL heterogeneity paper on AKI/sepsis
prediction across 7 real hospitals -- confirmed it uses only real institutional
splits, no synthetic Dirichlet comparison at all). Neither contradicts the gap
claim. Phrased as "we did not find" in the paper draft, not "first," since a
two-search effort is evidence of absence in the weak sense, not proof.

---
