# Decisions Log

Every non-obvious choice: what, date, why, alternatives rejected.
Append-only — supersede entries rather than deleting. This becomes the paper's Methodology section.

`D-XXX` numbering added 2026-08-18, retrofitted onto existing entries in
chronological order. **D-009 through D-014 are referenced elsewhere (see the
2026-08-18 "D-numbering reconciliation" entry near the end of this file) but
have no corresponding entry in this log** — either they were decided outside
this file (a separate note, a conversation with a teammate) or the numbering
got ahead of what's actually written up. Flagged, not fabricated: no content
has been invented to fill the gap.

---

## D-001 · 2026-08-18 — Dataset: full UCI Heart Disease, all 4 sites (920 records)

**What:** Use all four processed UCI Heart Disease files (Cleveland 303, Hungarian 294,
Switzerland 123, VA 200 = 920 records), each site treated as one simulated hospital/client.

**Why:** Cleveland alone (303 records) split across 4-5 federated clients leaves ~30-75
rows per client, which is too little to distinguish a real heterogeneity effect from
sampling noise. Using all four sites, each with its own natural (non-synthetic) collection
site, also gives us a genuine source of client heterogeneity to layer the Dirichlet
partitioning on top of.

**Rejected alternative:** Cleveland-only (303 records). Common in tutorials because it's
the cleanest of the four, but too small for a federated study with 4-5 clients and 4
alpha values x 3-5 seeds.

---

## D-002 · 2026-08-18 — Gradients: `diff_method="adjoint"`, not parameter-shift

**What:** All VQC training uses PennyLane's adjoint differentiation method on
`lightning.qubit`.

**Why:** Adjoint differentiation computes exact gradients in roughly one forward pass
plus one backward pass, regardless of the number of trainable parameters. Parameter-shift
requires two circuit evaluations *per parameter*, so cost scales linearly with parameter
count. For a full run grid (2 quantum arms x 4 alpha values x 3-5 seeds x many federated
rounds), parameter-shift is 30-70x slower and does not finish in the available time
(~50 hours total for two people). This is a simulator-only decision — parameter-shift
is required on real quantum hardware because adjoint differentiation needs access to the
full statevector, which isn't available on physical devices.

**Rejected alternative:** `diff_method="parameter-shift"` — the hardware-compatible
method. Not viable here because we are simulation-only and it doesn't fit the compute
budget. Documented as a limitation: this project's results describe simulator behavior,
not necessarily what training cost would look like on real hardware.

---

## D-003 · 2026-08-18 — Quantum device: `lightning.qubit`, not `default.qubit`

**What:** All quantum circuits run on PennyLane's `lightning.qubit` device.

**Why:** `lightning.qubit` is a C++ backend (vs. `default.qubit`'s pure-Python/NumPy
backend) and is substantially faster for circuit simulation at our scale (6 qubits),
especially combined with adjoint differentiation, which `lightning.qubit` implements
natively in C++.

**Rejected alternative:** `default.qubit` — PennyLane's reference simulator. Correct but
much slower; would not meaningfully change the science, only the wall-clock time, so
there's no reason to accept the slowdown.

---

## D-004 · 2026-08-18 — Circuit: 6 qubits, PCA to 6 components

**What:** Features are reduced via PCA to 6 principal components before encoding; the
VQC uses one qubit per feature (angle encoding), so 6 qubits total.

**Why:** One-qubit-per-feature angle encoding is the simplest, most defensible encoding
scheme to explain in an oral exam — no feature-to-qubit multiplexing to justify. 6 qubits
keeps classical simulation fast (statevector size grows as 2^n) and keeps the circuit
shallow enough to avoid barren plateaus dominating the training signal, which is the
binding constraint given the time budget, not the modeling ceiling of the dataset.

**Rejected alternative:** More qubits (e.g. 8-10) with less aggressive PCA, to retain
more variance from the original 13 features. Rejected because simulation cost grows
exponentially with qubit count and this project's binding constraint is total compute
time across ~40+ runs (2 quantum arms x 4 alpha x 3-5 seeds), not marginal predictive
accuracy from a couple of extra components.

---

## D-005 · 2026-08-18 — Local epochs per federated round: E=5, not E=1

**What:** Each client runs 5 local gradient steps on its own data per federated round,
not 1. Applies to all federated arms (2, 3, 4, 5) once built, and used in the timing
spike from this point on.

**Why:** With E=1, a client's local model barely moves before the round's parameters get
averaged back into the global model. That means almost no client drift accumulates
between rounds, which in turn means: (a) there is no measurable heterogeneity penalty to
detect as alpha shrinks, defeating the point of the study, and (b) FedProx's proximal
term has nothing to correct for, so it becomes indistinguishable from plain FedAvg —
also defeating the point of running Arm 3 at all. E=5 gives clients enough local movement
for both effects to actually show up.

**Rejected alternative:** E=1 (the original timing-spike default). Rejected for the
reasons above — it would produce a federated loop that "works" in the sense of running,
but can't answer the research question, and can't distinguish Arm 2 from Arm 3.

**Consequence:** re-timing the spike at E=5 (see `labbook.md`, 2026-08-18) increased the
per-run wall-clock roughly 5x, which pushed the extrapolated grid time out of the "full
grid, 5 seeds" band. Resolved by D-015 (grid size retained, round-count optimization
proposed instead).

---

## D-006 · 2026-08-18 — `chol == 0` treated as missing, at every site, not just Switzerland

**What:** In `scripts/data_loader.py`, any record with `chol == 0` has that value
replaced with NaN before any other processing (scaling, imputation, PCA).

**Why:** 0 mg/dl serum cholesterol is not a value a living patient can have — it's
almost certainly a missing-value code used instead of the UCI convention (`?`) at some
sites. `CLAUDE.md` flagged this for Switzerland specifically, but running the actual
numbers showed it is **not** Switzerland-only:

| site | records | chol==0 | raw `?` missing chol |
|---|---|---|---|
| cleveland | 303 | 0 (0.0%) | 0 |
| hungarian | 294 | 0 (0.0%) | 23 (7.8%) |
| switzerland | 123 | 123 (100.0%) | 0 |
| va | 200 | 49 (24.5%) | 7 |

Switzerland: literally every record. VA: about a quarter of records, coded the same
way. Hungarian's missing `chol` is coded correctly as `?` already, not zero.

**Correction (D-018):** this recoding is an *inference* from physiological
impossibility, not a fact documented by the dataset creators. The UCI documentation
does not describe `chol=0` as a missing-value code. Widely done in the literature, but
must be stated as an inference in the paper, not asserted as fact.

**Rejected alternative:** Only recoding Switzerland's `chol` as missing (matching the
literal wording in `CLAUDE.md`). Rejected once VA's zero-rate turned up — recoding only
Switzerland would have left ~24.5% of VA's cholesterol values silently wrong (treated as
real 0 mg/dl readings) rather than missing.

**Superseded by D-016:** `chol` itself is now dropped from the feature set (see below) —
this entry's recoding logic still matters for the missingness *report*, but `chol` is no
longer a modeling feature.

---

## D-007 · 2026-08-18 — `ca` and `thal`: dropped, not imputed

**What:** Both columns are dropped from the feature set entirely.

**Why:** Both columns are ~90-99% missing at every site except Cleveland (hungarian
99.0%/90.5%, switzerland 95.9%/42.3%, va 99.0%/83.0% for ca/thal respectively). At that
missingness level, any imputed value for those sites is essentially fabricated —
there's almost no real signal left in the column to impute *from* at 3 of 4 sites.
Decided by Prithvi.

**Rejected alternatives:**
- *Impute globally* (one imputer fit on all sites pooled) — rejected because Cleveland's
  near-complete data would dominate the fitted statistic used to fill in the other three
  sites' near-total missingness, which isn't meaningfully different from just assigning
  those sites a constant.
- *Impute per-site* — rejected because at 83-99% missing per site, there's too little
  real per-site data to estimate a per-site imputation statistic from; risks imputing
  noise and presenting it as signal.

**Consequence, superseded by D-016:** originally left 11 features for PCA. `chol` is
now also dropped (D-016), leaving 10 — see D-019 below for the discovery that a
*consistently applied* version of this same missingness rule leaves far fewer.

---

## D-008 · 2026-08-18 — Dirichlet partitioner, pooled across sites

**What:** `scripts/partitioner.py` pools all 920 records together (site identity
dropped) and splits them by class into simulated clients. For each class independently,
a per-client share vector is drawn from `Dirichlet(alpha, ..., alpha)` (numpy
`default_rng(seed).dirichlet`) and used to assign that class's shuffled row indices to
clients via cumulative proportions.

**Why pool across sites instead of partitioning within each site:** the Dirichlet
alpha is meant to be a single controlled independent variable for "how non-IID are the
clients." The four real sites already have their own fixed, uncontrollable skew (pct
positive ranges 36.1% to 93.5% by site — see labbook, 2026-08-18). Pooling first means
the Dirichlet skew is the only thing being swept; real site identity is kept only as a
separate reference plot, not mixed into the controlled variable.

**Original client count (5) superseded by D-017** — see below.

**Rejected alternative:** partitioning within each site separately (Dirichlet applied
per-site rather than pooled). Rejected because Switzerland only has 123 records and 8
class-0 examples total — a per-site Dirichlet split at low alpha would leave some
clients with 0-1 examples of a class, which is a sample-size artifact, not a
heterogeneity effect worth reporting.

---

## D-015 · 2026-08-18 — Grid size confirmed: full 4 α × 5 seeds, no cut

**Decision:** Retain 4 α values and 5 seeds for all arms despite the E=5 re-time (D-005).

**Arithmetic:** 18.5 min/quantum run × 2 quantum arms × 4 α × 5 seeds = 40 runs ≈ 12.3
hrs, plus ~3.1 hrs for the natural-partition condition. ~16 hrs with buffer. Classical
arms are negligible by comparison. This is a single overnight run against a
multi-week schedule.

**Why not cut:** seeds are what let us report variance rather than single-point
results, and α values are the independent variable. Cutting either weakens the central
claim to save compute we can afford.

**Preferred optimisation instead:** verify whether 50 communication rounds are
required. If models converge by ~30 rounds, reducing the round budget cuts the grid
~40% at no scientific cost, and rounds-to-convergence is a reported metric regardless.
Convergence check requested as Task 3 of the 2026-08-18 session — see labbook.

---

## D-016 · 2026-08-18 — `chol` dropped, consistent with D-007

**Decision:** `chol` is dropped from the retained feature set.

**Why:** Under D-006, `chol = 0` is a missing-value code. Switzerland is 100% zero, so
after conversion Switzerland has 0% measured cholesterol and the column fails the
≥85%-at-every-site rule exactly as `ca` and `thal` did (D-007). Retaining `chol` while
dropping `ca` and `thal` on the same rule would be inconsistent application of a stated
criterion.

**Cost acknowledged:** cholesterol is clinically relevant. A Cleveland-only sensitivity
analysis was proposed (referenced as D-014) to quantify what this exclusion costs, but
**no such analysis exists in this repo** — flagged as an open item, not fabricated.

**Consequence — feature retention crisis (see D-019 below):** applying this same
≥85%-at-every-site rule *consistently* to every remaining candidate column (not just
the ones already under suspicion) leaves only 4 features (`age`, `sex`, `cp`,
`restecg`), which is below the PCA-to-6 floor. Not resolved in code — reported to
Prithvi per the explicit stop condition in the task instructions.

---

## D-017 · 2026-08-18 — Client count fixed at 4 for both partitioning schemes

**Decision:** Dirichlet partitioning uses 4 clients, matching the 4 natural sites.
Supersedes the 5-client choice in D-008.

**Why:** comparing natural institutional partitioning against synthetic Dirichlet
partitioning (referenced as objective D-009, not separately documented in this log —
see the numbering-gap note at the top of this file) requires that the two conditions
differ in partitioning method only. If the synthetic condition uses 5 clients and the
natural one uses 4, any observed difference could be due to client count rather than
partitioning method. Matching client count isolates the variable.

**Consequence:** the timing spike (`scripts/timing_spike.py`) still uses 5 synthetic
clients — not yet reconciled with this 4-client decision. Flagged, not yet fixed.

---

## D-018 · 2026-08-18 — Dataset provenance, licensing, and two facts for the paper

**Citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart
Disease [Dataset]. UCI Machine Learning Repository. doi:10.24432/C52P4X. Licensed CC BY
4.0 — attribution is a licence condition and must appear in the paper.

**Origin paper:** Detrano, R., Jánosi, A., Steinbrunn, W., Pfisterer, M., Schmid, J.,
Sandhu, S., Guppy, K., Lee, S., & Froelicher, V. (1989). "International application of
a new probability algorithm for the diagnosis of coronary artery disease." Cite for
provenance.

**Documentation findings:** source documentation confirms 76 original attributes with
a 14-attribute subset used in all published work; `thal` coded 3/6/7; `slope` 1/2/3;
`ca` 0-3; `num` 0-4 with the standard binarisation distinguishing 0 from 1-4.

**Two facts to state explicitly in the paper, as inference rather than documented
fact:**
1. The "Switzerland" file combines **two** institutions — University Hospital Zurich
   and University Hospital Basel. That client is therefore not a single site, which
   matters for how "natural site heterogeneity" is described.
2. `chol = 0` is **not** documented by the dataset creators as a missing-value code
   (see D-006 correction above). Our treatment of it as missing is an inference from
   physiological impossibility, widely observed in the literature but undocumented.

---

## D-019 · 2026-08-18 — Feature retention rule, applied consistently, fails the PCA-6 floor

**What:** `scripts/feature_retention_check.py` applies the same rule invoked in D-016
(retain only if missingness <= 15% at every site, i.e. ">=85% present") to every
remaining candidate column, not just the ones already dropped.

**Result:**

| column | cleveland | hungarian | switzerland | va | verdict |
|---|---|---|---|---|---|
| age | 0.0 | 0.0 | 0.0 | 0.0 | keep |
| sex | 0.0 | 0.0 | 0.0 | 0.0 | keep |
| cp | 0.0 | 0.0 | 0.0 | 0.0 | keep |
| trestbps | 0.0 | 0.3 | 1.6 | 28.0 | DROP |
| chol | 0.0 | 7.8 | 100.0 | 28.0 | DROP |
| fbs | 0.0 | 2.7 | 61.0 | 3.5 | DROP |
| restecg | 0.0 | 0.3 | 0.8 | 0.0 | keep |
| thalach | 0.0 | 0.3 | 0.8 | 26.5 | DROP |
| exang | 0.0 | 0.3 | 0.8 | 26.5 | DROP |
| oldpeak | 0.0 | 0.0 | 4.9 | 28.0 | DROP |
| slope | 0.0 | 64.6 | 13.8 | 51.0 | DROP |
| ca | 1.3 | 99.0 | 95.9 | 99.0 | DROP |
| thal | 0.7 | 90.5 | 42.3 | 83.0 | DROP |

Only **4 columns survive**: `age`, `sex`, `cp`, `restecg`. Below the PCA-to-6 floor.

**Why this matters:** `fbs` (61% missing at Switzerland) and `slope` (64.6% missing at
Hungarian, 51% at VA) fail this rule as badly as `chol` did, and `trestbps`,
`thalach`, `exang`, `oldpeak` all fail it at VA specifically (~26-28% missing there,
which looks like a block of jointly-missing columns rather than independent random
missingness — see labbook). Applying the ≥85%-at-every-site rule to justify dropping
`chol` (D-016) but not to these columns would be exactly the inconsistency D-016 itself
was trying to avoid.

**Not resolved here.** Per the explicit stop condition in the task instructions ("If
fewer than 6 features survive, stop and tell me"), this is reported rather than
decided. Candidate resolutions, none chosen yet:
- Lower the PCA target below 6 (breaks the "one qubit per feature" framing in D-004).
- Use a laxer or column-type-aware threshold (e.g. distinguish "physically impossible
  value, definitely a missing-code" columns like chol/ca/thal from "elevated but
  plausible missingness, imputable" columns like the VA block).
- Impute the moderately-missing columns (trestbps/thalach/exang/oldpeak at VA,
  fbs/slope at Switzerland/Hungarian) despite the criterion used for ca/thal/chol,
  explicitly on the grounds that 72-75% real data per site is not comparable to the
  ~1-15% real data ca/thal/chol have at the worst sites.

---

## D-numbering reconciliation, 2026-08-18

`D-001` through `D-008` above are retrofitted onto pre-existing entries in this file,
in chronological order — not renumbered from any prior canonical scheme, since none
existed before today. Retrofitting confirmed `D-006` and `D-007` line up exactly with
the content referenced under those numbers in D-016, which is reassuring: the
numbering scheme appears to be real and consistently used elsewhere, not accidental.

**Still open:** `D-009` through `D-014` are referenced (D-009: an objective comparing
natural vs. Dirichlet partitioning; D-010: a minimum-client-size guard; D-014: a
Cleveland-only sensitivity analysis for the chol/ca/thal exclusions) but have no
corresponding entries here, and the corresponding code/analysis does not exist in this
repo either — in particular, `scripts/partitioner.py` has no minimum-client-size logic
of any kind. Need Prithvi to clarify whether these were decided/built elsewhere (and
should be backfilled into this log) or still need to be written up and implemented.

---

## D-020 · 2026-08-18 — Run-level parallelism: 4 concurrent processes, ~90% efficiency

**What:** Grid runs (each a separate `(arm, alpha, seed)` combination) can be launched
as independent OS processes rather than one after another. Tested empirically with
`scripts/parallel_throughput_test.py` on this development machine (16 logical cores):
1 baseline run, then 4 concurrent runs under default threading, then 4 concurrent runs
with `OMP_NUM_THREADS` capped to `cpu_count // 4 = 4` (to check whether
`lightning.qubit`'s internal OpenMP threading was oversubscribing cores against the
process-level parallelism).

Workload used for the test: same circuit as the timing spike, 4 clients (D-017), 230
rows/client (4x230=920, matches real dataset size), `LOCAL_STEPS=5` (D-005), but only
10 rounds instead of 50 — reduced only so the throughput test itself finishes in
minutes rather than tens of minutes; round count doesn't change per-round contention
behavior, which is what this test measures.

**Result:**

| condition | wall-clock | vs. 4x baseline (942.1s) |
|---|---|---|
| baseline (1 run) | 235.5s | — |
| 4 concurrent, default env | 260.8s | 3.61x speedup, 90% efficiency |
| 4 concurrent, `OMP_NUM_THREADS=4` | 259.6s | 3.63x speedup, 91% efficiency |

Capping threads-per-process made no meaningful difference (259.6s vs 260.8s) — at 6
qubits (64 amplitudes), there's little internal linear algebra for OpenMP to
parallelize within a single circuit evaluation, so process-level parallelism doesn't
have to fight itself for cores. No special threading configuration needed.

**Sanity check against D-015:** scaling this test's 10-round, 4-client baseline
(235.5s) to the real 50-round run gives ~1177.7s (~19.6 min), close to D-015's
1107.6s (~18.5 min, measured at 5 clients/180 rows-per-client instead of 4/230) — the
two independent measurements agree to within ~6%, which is reassuring given they used
different client counts and were run at different times.

**Consequence for the grid:** applying the observed ~90% efficiency to the real quantum
grid (40 runs: 2 quantum arms x 4 alpha x 5 seeds, D-015) in batches of 4 concurrent
processes: ~10 batches x ~1304s/batch (1177.7s x 1.107 contention factor) ≈ 13,040s ≈
**~3.6 hours wall-clock**, down from the ~12.3-13 hour sequential estimate in D-015.
This does not change D-015's decision (grid size was already confirmed, not driven by
compute budget) — it changes how much slack there is around it, and means the
convergence-check-driven round-count optimization (D-015's preferred lever) has less
pressure behind it than it did before this test, though it's still free to pursue.

**Not tested:** whether >4-way parallelism (this machine has 16 logical cores, so 8x
or even 16x might scale further) helps beyond the 4x case asked for here. Flagged as a
follow-up, not run without being asked.

**Caveat:** measured on the development machine, not necessarily the machine the real
grid will run on. Re-verify `cpu_count` and re-run this test on the actual execution
machine before relying on the 3.6-hour estimate.

---

## Numbering note, 2026-08-18

Prithvi's 19 August session prompt asked to log `D-021` through `D-023` "pasted
separately," but that content never arrived in the actual message. The four entries
below (feature finalization, partitioner guard, federated loop interface, validation
gate results) are numbered `D-021` through `D-024` as the next sequential numbers
after `D-020`, **not** copied from Prithvi's intended content, which I don't have. If
Prithvi's actual `D-021`-`D-023` differ from what's recorded here, these will need
renumbering — flagged, not silently assumed to match.

---

## D-021 · 2026-08-18 — Final feature set: 6 features, no PCA, selected by worst-site availability

**What:** `age`, `sex`, `cp`, `restecg`, `thalach`, `exang` — the 6 columns with the
highest worst-site availability. PCA removed entirely; each feature maps directly to
one qubit's RY rotation angle.

**Per-site availability (data pulled fresh, `scripts/data_loader.py`):**

| column | cleveland | hungarian | switzerland | va |
|---|---|---|---|---|
| age | 100.0% | 100.0% | 100.0% | 100.0% |
| sex | 100.0% | 100.0% | 100.0% | 100.0% |
| cp | 100.0% | 100.0% | 100.0% | 100.0% |
| restecg | 100.0% | 99.7% | 99.2% | 100.0% |
| thalach | 100.0% | 99.7% | 99.2% | 73.5% |
| exang | 100.0% | 99.7% | 99.2% | 73.5% |

**Why no PCA:** with only 6 raw features surviving the availability cutoff, PCA down
to 6 components would be an orthogonal rotation of a 6-dimensional space into another
6-dimensional space — not dimensionality reduction, just an uninterpretable rotation
of already-interpretable clinical features. Direct mapping (one clinical feature, one
qubit) is simpler to defend in an oral exam: "qubit 3 is `restecg`" is a real,
checkable claim; "qubit 3 is 0.31*age - 0.08*chol + ..." is not, once chol/ca/thal are
gone anyway. **Supersedes the PCA half of D-004** — D-004's qubit count (6) and circuit
structure (angle encoding, 3-layer ansatz) are unchanged, only the "PCA to 6
components" framing is dropped.

**Residual missingness in kept features, and why it's imputed rather than dropped:**
`restecg` (up to 0.8% missing), `thalach`/`exang` (up to 26.5% missing, at VA
specifically) still have some missingness. Unlike the excluded columns (61-100%
missing at their worst site), these sites still have 73.5-99.2% real data — worth
imputing (median, fit on train split only, `scripts/preprocessing.py`) rather than
dropping the column or the rows.

**Scaling:** `MinMaxScaler` to `[0, pi]`, fit on the training split only (per the
leakage warning in `CLAUDE.md`). `RY(0) = |0>`, `RY(pi) = |1>` — the full single-qubit
rotation range. Test-split values can fall slightly outside `[0, pi]` since the scaler
is fit on train only (observed max on one test split: 3.304 vs pi=3.14159) — expected
and correct (no leakage), not a bug; an RY angle slightly past pi is not an error,
just continues rotating past `|1>`.

**Consequence:** Arm 1's expected accuracy band moves down from the original "~83-85%"
(13-feature Cleveland-only literature baseline) to "~75-80%" (6 raw features, all 4
sites) — confirmed by Prithvi in this session's task instructions ahead of actually
running it. See D-024 for the real number.

---

## D-022 · 2026-08-18 — Minimum-client-size guard: floor=15 rows, reject-and-redraw

**What:** `scripts/partitioner.py`'s `dirichlet_partition` now rejects any Dirichlet
draw where a client's total row count falls below `MIN_CLIENT_SIZE=15`, and redraws
(different sub-seed derived from `seed * 100_000 + attempt`) up to `MAX_ATTEMPTS=500`
times. Raises `RuntimeError` if no valid draw is found within that budget, rather than
looping forever or silently returning a degenerate partition.

**Why floor=15:** local training uses `LOCAL_STEPS=5` (D-005) full-batch gradient
steps per client per round. 15 rows is small but enough that a full-batch gradient
over 15 examples is a real (if noisy) signal, not a single-point estimate. Not derived
from a formal calculation — a round number chosen to be clearly above "degenerate"
(the n=1 client seen in an earlier, unguarded 5-client run) and clearly below "would
meaningfully constrain the alpha=0.1 skew we're trying to study." Open to revision if
it turns out to bind in practice.

**Verified working, not just present:** at the locked alpha grid ({100, 1.0, 0.5,
0.1}, seed=0), the guard did not need to fire — the unguarded draw already had
min(client sizes)=41, above the floor. Confirmed the guard mechanism itself works with
a stress test outside the locked grid: at alpha=0.05, the guard fired and successfully
redrew on most seeds tested (1-20 attempts across 5 seeds); at alpha=0.01, it correctly
raised `RuntimeError` after exhausting 500 attempts, rather than returning a bad
partition silently. Neither 0.05 nor 0.01 is in the actual experiment grid — this was
a deliberate stress test of the guard mechanism, not a claim that the guard fires
during real experiments.

---

## D-023 · 2026-08-18 — Federated loop + interface implementation (Arm 1, Arm 2)

**What:** Built the shared infrastructure per the `CLAUDE.md` interface contract:
`scripts/models.py` (`LogisticRegressionModel` — plain logistic regression via
full-batch gradient descent, not sklearn, specifically so it can satisfy
`get_params`/`set_params`/`fit(X,y,epochs)`/`predict_proba`), `scripts/aggregators.py`
(`fedavg`), `scripts/federated_loop.py` (`run_centralized`, `run_federated`), and
`scripts/run_grid.py` (resumable experiment runner). Full contract recorded in
`docs/INTERFACE.md`, frozen as of this entry.

**Why a custom logistic regression instead of sklearn's:** sklearn's
`LogisticRegression.fit(X, y)` has no `epochs` parameter and no clean flat
get/set-params vector matching the interface contract's shape — wrapping it to fit the
contract would add more code than just writing gradient descent directly, and the
custom version is fully inspectable (every line of the training loop is visible),
which matters for an oral defense.

**Confirmed no model-type branching:** `scripts/federated_loop.py` contains no `if
model_type == "quantum"` or equivalent, by inspection. It only calls the four `Model`
interface methods and the aggregator callable.

**Resumable logging (Task 4), tested with an actual kill:** `scripts/run_grid.py`
appends one row per `(arm, alpha, seed)` combination to `results/runs.csv`
immediately after that run completes, and skips any combination already present on
restart. Tested for real: launched the full Arm1+Arm2 grid (25 combinations) with an
artificial per-run delay (`RUN_GRID_TEST_DELAY_SEC=1`, needed only because a real run
takes ~10-30ms — too fast to interrupt meaningfully otherwise), let it run for ~6s,
killed the actual worker process (`kill -9` on the real PID — note: `$!` under
git-bash's process emulation captured the wrong PID on the first attempt and the kill
silently failed; had to find and kill the real `python.exe` PID via `ps aux`, worth
remembering if this is retested), confirmed 19 of 25 combinations had been logged,
restarted without the delay, confirmed the restart printed `skip ...` for exactly
those 19 and only executed the remaining 6, and confirmed no duplicate `(arm, alpha,
seed)` rows exist in the final file. Test passed.

---

## D-024 · 2026-08-18 — Arm 1 / Arm 2 validation gate results

**What:** Ran the full Arm 1 (5 seeds) + Arm 2 (4 alpha x 5 seeds = 20 combinations)
grid via `scripts/run_grid.py`. Results (`results/runs.csv`):

| gate | expected | observed | verdict |
|---|---|---|---|
| Arm 1 accuracy | ~75-80% | 77.50% mean (std 3.04%, 5 seeds) | PASS |
| Arm 1 accuracy not >90% (leakage check) | <90% | 77.50% | PASS, no leakage signal |
| Arm 2 @ alpha=100 vs Arm 1 | within ~2% | 77.28% vs 77.50%, gap 0.22 pct pts | PASS |
| Arm 2 across alpha sweep | monotonic-ish decline | alpha=100: 77.28%, 1.0: 77.39%, 0.5: 76.96%, 0.1: 77.50% | **NOT CLEARLY PASSING** -- see below |
| Repeated seed -> identical output | identical | bit-identical params and metrics across two independent runs of the same (arm2, alpha=1.0, seed=2) | PASS |

**The alpha-sweep gate is flat/noisy, not declining.** The four alpha-group means span
only 0.54 percentage points (76.96% to 77.50%), smaller than the within-group standard
deviation at every alpha (3.3-3.7%) -- i.e. the differences between alpha values are
well within noise for 5 seeds, and alpha=0.1 (most skewed) actually has the *highest*
observed mean, not the lowest. Reported as observed rather than adjusted to fit the
expected pattern.

**Why this might not be a bug:** the most important gate (`alpha=100` vs Arm 1, "the
gate that matters most" per this session's instructions) passed cleanly, and the
repeated-seed determinism gate passed cleanly, which together suggest the loop itself
is wired correctly rather than broken. A plausible, not-yet-verified explanation:
`LogisticRegressionModel` is linear and `FEDERATED_ROUNDS=20`/`LOCAL_EPOCHS=5` may
converge to a similar decision boundary regardless of how skewed each round's client
data is, at only 4 clients -- i.e. the classical arm may simply be robust to this
amount of heterogeneity at this model capacity, which would itself be a legitimate
(if less dramatic) finding for the paper rather than a defect. Not confirmed. No
hyperparameter was adjusted to try to force a declining pattern -- that would be
tuning toward an expected result rather than reporting what happened, which `CLAUDE.md`
explicitly rules out.

**Not resolved here, flagged for Prithvi:** whether this needs investigation (more
seeds, more rounds, a different alpha grid, or just documenting "classical Arm 2 was
robust to label skew at this scale" as a real result) before treating Arm 2 as
validated infrastructure for Arm 3/4/5.

---

## D-025 · 2026-08-18 — Diagnostic session: 5-fold CV with a stable per-seed client identity

**What:** `scripts/cv_protocol.py` replaces the single 736/184 split with 5-fold
stratified CV x 10 seeds. Client assignment (Dirichlet at a given alpha, or the
natural 4-site split) is drawn **once per seed over the full 920-row pool**, not
re-drawn per fold. A client's per-fold training data is (client's rows) intersect
(fold's training rows); its per-fold test data is (client's rows) intersect (fold's
held-out rows).

**Why draw the client assignment once per seed, not once per fold:** this gives
every client a stable identity across all 5 folds of a seed -- client 2's
characteristic skew is the same distribution in every fold. That's what makes
"worst-client accuracy" trackable as a single quantity per (seed, condition) rather
than 5 unrelated per-fold random groupings that happen to also be called "client 2."
The alternative (re-partition per fold) would make per-client metrics much noisier
without a clear benefit.

**Consequence:** every one of the 920 records is used for global testing exactly
once per seed, giving an effective global test size of 920 and cutting the standard
error roughly 8-10x (see `docs/diagnostic_report.md`, noise floor). Every client also
gets its own held-out slice every fold, enabling genuine per-client evaluation.

**Caveat:** at alpha=0.1, per-client test-fold sizes go as low as n=1 (median 31.5).
Not a bug -- an expected consequence of splitting an already-skewed ~46-row average
client slice 4 ways under a ~184-row fold. Flagged in the diagnostic report rather
than smoothed over.

---

## D-026 · 2026-08-18 — MLP parameter count: 17, matched to the real VQC (18), not the requested ~36

**What:** `scripts/models_mlp.py`'s `MLPModel` (6 -> 2 -> 1, tanh hidden, sigmoid
output, both layers biased) has 17 trainable parameters.

**Why 17, not the ~36 the diagnostic task prompt requested:** the actual frozen VQC
(`docs/circuit_diagram.txt`, D-004) has 18 trainable parameters -- 6 qubits x 3
layers x 1 RY/qubit/layer -- not 36. The stated purpose of parameter-matching (Task 4)
is to remove model capacity as a confound when later comparing against the real VQC in
Arm 4. Matching to a number that doesn't correspond to the real circuit would defeat
that purpose. 17 is the closest achievable count with a standard single-hidden-layer
architecture and clean integer hidden-unit count (h=2 -> 12+2+2+1=17; no integer h
hits exactly 18 with a plain biased single-hidden-layer design).

**Not resolved, flagged:** where the "~36" estimate came from is unclear -- possibly a
different assumed ansatz (e.g. 2 rotations per qubit per layer would give 36). If the
real circuit changes before Arm 4 is built, this MLP's parameter count should be
re-matched to whatever the circuit actually is at that time, not to this number.

---

## D-027 · 2026-08-18 — `run_federated` interface amendment: optional divergence tracking

**What:** `scripts/federated_loop.py:run_federated` gains a `track_divergence: bool =
False` parameter. When `True`, returns `(model, divergence_per_round)` instead of just
`model`, where `divergence_per_round[r]` is the mean pairwise L2 distance between
client parameter vectors after local training but before aggregation, in round `r`.

**Why an amendment rather than a new function:** the alternative (a separate
`run_federated_with_divergence`) would duplicate the entire training loop body for one
extra measurement, risking the two copies drifting apart. Keeping one function with an
opt-in flag means there is exactly one place the federated training loop is defined,
which matters for an oral defense of "one loop, not five scripts."

**Backward compatibility:** default `False` preserves the exact original return
contract. `scripts/run_grid.py` (Arm 1/Arm 2 grid, frozen interface, D-023) does not
pass this argument and is unaffected -- verified by inspection, not re-run, since the
change is additive and the default path is untouched.

**Consequence for `docs/INTERFACE.md`:** the frozen loop contract is amended, not
reopened. This is the first change to a frozen interface since the freeze (D-024) --
noted here explicitly per the freeze's own instruction: "if a new arm seems to require
editing shared infrastructure, stop and raise it." This wasn't a new arm, it was a new
measurement need on the existing arms, and the change is purely additive.

---

## D-028 · 2026-08-18 — Diagnostic session findings: heterogeneity penalty exists, measured in the wrong place

**What:** Full findings in `docs/diagnostic_report.md`. Summary: the classical
alpha-sweep flatness observed in D-024 was real but incomplete -- global accuracy
genuinely is flat for logistic regression across the entire alpha range (noise floor
now ~0.3pp, 10x tighter than before, so this is not a power problem). But **worst-client
accuracy declines monotonically as alpha falls, for both LR and MLP**, and **client
parameter divergence rises monotonically as alpha falls, for both models, with tight
error bars**. MLP additionally shows a global-accuracy penalty at alpha=0.1 that LR
does not show, consistent with (not proof of) convexity mediating the effect's
visibility at the global level.

**Natural partition (objective D-009):** does not sit outside the synthetic Dirichlet
range on any metric -- comparable to a moderate synthetic skew (~alpha=0.5-1.0), not
more damaging than the most extreme synthetic condition tested.

**Decision-table row supported:** a combination of "divergence rises, global flat,
mechanism fires" (rows 1) and "worst-client declines, penalty exists, wrong
measurement location" (row 2), with row 3 (convexity mediates) supported specifically
at the global-metric level. Row 4 (natural worse than synthetic) explicitly **not**
supported. Full row-by-row verdict in the diagnostic report, Section 6-7.

**No course of action recommended here** -- interpretation reserved for Prithvi, per
the task instruction.

---

## D-029 · 2026-08-20 — Worst-client/global-accuracy gap is not a novel finding; literature grounding required (Ayuvi)

**What:** Literature search on FL fairness / client-level performance disparity under
non-IID data. Full summary: `docs/reference/fl_fairness_literature.md`. 5 papers cited
in `paper/02_related_work.md`: Mohri et al. (ICML 2019, agnostic FL), Li et al. (ICLR
2020, q-FFL), Li et al. (MLSys 2020, FedProx -- also our own Arm 3 baseline), Liu
(arXiv:2507.12983, 2025, FedGA), and Naseer & Shoaib (arXiv:2605.08992, 2026).

**Why this matters for the paper, explicitly:** Naseer & Shoaib (2026) already report
the same qualitative pattern D-028 found -- global/mean accuracy insulated from damage
that concentrates in worst-client accuracy, worsening sharply under increasing
heterogeneity -- via a controlled label-skew sweep, on a text-classification task. We
must not present "worst-client accuracy degrades while global accuracy stays flat" as
this project's discovery. The paper's contribution has to be framed as (a) confirming
this pattern on EHR tabular data rather than text, and (b) extending the comparison to
a variational quantum classifier vs. classical models, which none of the 5 cited papers
address.

**Alternatives rejected:** presenting D-028's finding without literature grounding was
not seriously considered -- `CLAUDE.md` requires citing prior work rather than claiming
an established phenomenon as a discovery, and the diagnostic report itself flagged this
as an open question for the literature to resolve.

---

## D-030 · 2026-08-20 — `plots.py` figure design: data source, worst-client definition, natural-partition presentation (Ayuvi)

**What:** Built `scripts/plots.py`, producing the three primary-result figures
(worst-client accuracy vs. alpha, global accuracy vs. alpha, client divergence vs.
alpha) at dpi=200, plus the shared-axes pairing between the first two figures.

**Data source:** built against `results/diagnostic_results.csv` and
`results/diagnostic_divergence.csv`, not `results/runs.csv`. `runs.csv` only has one
global-accuracy row per run -- no per-client breakdown -- so it cannot drive a
worst-client figure. The diagnostic CSVs have the required `client` and `round`
granularity. Extending to Arm 4/5 later means adding their results file paths to
`RESULTS_SOURCES`/`DIVERGENCE_SOURCES` at the top of the file, provided the new CSV
reuses the same long-format columns; a missing source file is skipped with a warning,
not a crash, verified by testing with a nonexistent `results/runs_arm4.csv` path added.

**Worst-client accuracy definition:** minimum accuracy across a run's non-`global`
`client` rows, per (arm, model, condition, seed, fold) replicate, then mean/std across
replicates for the error band. This uses all seed x fold replicates (up to 50) as the
error-band source, not only the 10 seeds -- a superset of what the task asked for, on
the reasoning that more replicates is strictly more informative for the same
error-band purpose; flagged here as a deviation from the literal instruction ("error
bands from the 10 seeds") in case that specific replicate count matters for the paper's
methodology section.

**Arm 1 is structurally absent from the global-accuracy figure:** it has per-client
alpha-conditioned rows (used in the worst-client figure, since the diagnostic session
also evaluated the centralized model on each Dirichlet client's local test slice, as a
"would federation even help" baseline) but no alpha-conditioned *global*-accuracy rows
-- its global accuracy doesn't depend on how the test set happens to be partitioned
into clients. An early version of this script plotted an empty, misleading legend entry
for Arm 1 there; fixed by skipping any series with zero rows after alpha-filtering
before it reaches the legend.

**Natural-partition reference line placed in the legend, not as inline text:** an
earlier version annotated each series' natural-partition value with floating text next
to its horizontal reference line. With 4 series clustered close together (worst-client
figure), the text overlapped illegibly and ran past the axes edge at alpha=0.1
(colliding with the real alpha=0.1 data point). Moved to a labeled legend entry per
series instead -- more entries in the legend, but every one is readable.

**Consistent series color across the figure pair:** matplotlib's default color cycling
is per-axes, so when Arm 1 drops out of the global-accuracy figure, Arm 2 would have
been reassigned Arm 1's former colors there -- breaking the "one figure moves, one
doesn't, compare them directly" point of the pairing. Fixed with an explicit color map
built once from the full (arm, model) key set before any alpha-filtering, reused across
both figures.

---

## D-031 · 2026-08-20 — Reframing: worst-client degradation is cited, not claimed (Ayuvi)

**What:** The project's central empirical pattern -- global accuracy flat/mildly
declining under non-IID skew while worst-client accuracy collapses -- is confirmed
prior art, not this project's discovery. Directly verified: q-FFL (Li, Sanjabi,
Beirami, Smith, ICLR 2020), Appendix Table 10 ("Effects of data heterogeneity and the
number of devices on unfairness"), reports the same shape under FedAvg (q=0): Average
accuracy 89.2%->83.0%->82.6% vs. Worst-10% accuracy 70.9%->36.8%->25.5% as heterogeneity
increases (Synthetic IID -> (1,1) -> (2,2), 100 devices). This is the same paper
already cited in D-029 for its q-FFL method, whose appendix hadn't been checked at that
point.

**Why this is a decision and not just a citation update:** it changes how the Results
section must be written. "We find that worst-client accuracy degrades sharply while
global accuracy stays flat" is a discovery claim and is no longer accurate to make
unqualified -- it must be written as replication of an established 2020 result, on a
new data modality (EHR tabular, Dirichlet-α skew) and extended to a new comparison
(quantum vs. classical) that q-FFL does not address.

**Relationship to the existing guardrail:** `CLAUDE.md`'s guardrail section already
states this is a characterization study, not a claim of novelty in the headline
quantum-vs-classical result. D-031 does not introduce a new constraint -- it makes that
guardrail concrete for a specific piece of the paper (the worst-client/global-accuracy
contrast, independent of the quantum question) that could otherwise have been written
as if it were an original observation, the way D-028's diagnostic report language
("we observed") could read if quoted without this context.

**Not yet resolved:** whether NIID-Bench (Li, Q. et al., ICDE 2022) independently
reports the same pattern under Dirichlet-α skew specifically (as opposed to q-FFL's
synthetic non-IID construction) -- would be a more directly comparable prior-art source
for this project's exact partitioning method. Flagged as an open item in
`docs/labbook.md`, not resolved here.

**Alternatives rejected:** leaving the citation as a general "related work" mention
(D-029's original framing) without this appendix-table-level verification was
considered sufficient until this session -- rejected once the specific quantitative
match was found, since a vague "this has been studied before" citation is weaker
protection against an accidental discovery claim than a specific, checked source.

---

## D-032 · 2026-08-20 — NIID-Bench does not report per-client accuracy; open item from D-031 closed

**What:** Checked NIID-Bench (Li, Diao, Chen, He, ICDE 2022, arXiv:2102.02079) directly
-- both the GitHub repo (Xtra-Computing/NIID-Bench) and the paper's full text -- for
whether it reports per-client or worst-client accuracy under its Dirichlet sweep, per
the open item flagged in D-031.

**Result: it does not.** NIID-Bench reports only aggregate/global top-1 accuracy across
all parties, in every setting benchmarked. It is **not** a second direct precedent for
our worst-client/global-accuracy contrast -- that hope (raised as an open item in D-031)
does not pan out. q-FFL's Appendix Table 10 remains our only directly-verified prior-art
source for that specific contrast.

**What NIID-Bench does support:** Section V-A2 (Finding 2): "No algorithm consistently
outperforms the other algorithms in all settings." Table III shows FedProx beating,
tying, or losing to FedAvg depending on dataset (e.g. CIFAR-10, Dirichlet(0.5): FedAvg
68.2% vs. FedProx 67.9%; rcv1: FedAvg 48.2% vs. FedProx 70.3%). This is directly usable
as a characterization-study precedent if Arm 3 (FedProx) is built: a validation-gate
failure (Arm 3 not beating Arm 2 at low alpha) would not be a bug in our implementation,
it would match established literature.

**FedProx interface check (unrelated question, verified while reading the same code):**
`scripts/federated_loop.py:52-53` calls `local_model.set_params(global_params.copy())`
immediately before `local_model.fit(X_c, y_c, epochs=local_epochs)`, for every client,
every round. So a FedProx implementation can snapshot the global params vector as its
proximal-term anchor inside the model's own `fit`, with no interface change -- confirmed
directly from the frozen loop code, not by asking Prithvi.

---

## D-033 · 2026-08-20 — Correction to D-031: disparity is prior art, flatness and convexity are not (supersedes framing, not the citation)

**What:** D-031 established that q-FFL Appendix Table 10 is prior art for our
worst-client/global-accuracy result, and characterized this as making the whole
pattern "replication of an established 2020 result." That framing overstated
the overlap. Corrected here, and in `docs/reference/fl_fairness_literature.md`
entries [2] and [5].

**Precisely, what q-FFL's Table 10 actually shows (re-derived from the same
numbers cited in D-031):** under FedAvg, their Average accuracy declines a real
6.6pp across their heterogeneity sweep (89.2% -> 82.6%, 100 devices). It is
*smaller* than their Worst-10% decline (44.7pp), but it is not flat. Our own
LR's global accuracy, by contrast, is flat within noise across our full α
sweep (0.7625 to 0.7651). **The disparity (worst-client damage exceeds global
damage) is prior art. The flatness of the global metric specifically is not —
it is a genuine feature of our specific result, not already published.**

**Also not established by q-FFL:** any convex-vs-non-convex model comparison —
their dataset is a synthetic linear/softmax construction with no such axis, so
our D-028 finding (MLP shows a global-accuracy penalty at extreme skew that LR
does not) has no counterpart in their work. Their device counts (50, 100) are
also an order of magnitude above our 4 clients, and their own table shows
*fewer* devices producing *more* uniform (less disparate) accuracy — so their
trend does not extrapolate freely down to our client count in either
direction.

**Same correction applied to a second source, found independently while fixing
the first:** the Naseer & Shoaib (2026) entry [5] made the identical
overstatement — claiming their paper shows "global accuracy insulated from
damage." Checked their Table 3 directly: it is not. TextCNN's average accuracy
ranges 86.6%-97.8% across their α sweep (11.2pp), DistilBERT+LoRA 80.8%-93.6%
(comparable range) — both heterogeneity-sensitive at the global level, not
insulated. Corrected the same way: cited for the disparity (worst-client damage
exceeds global damage), not for global-accuracy flatness specifically.

**Consequence for the Results section:** must be written as "the disparity
between average and worst-client accuracy under heterogeneity is established
prior art (q-FFL, Table 10); our specific finding is that in a convex model
(LR) on EHR tabular data, the average/global metric doesn't just decline less
than the worst-client metric, it stays flat, and this insulation itself breaks
down in a non-convex model (MLP) at extreme skew — a convexity-mediated
effect with no precedent in the cited literature." Not "we replicate an
established 2020 result" (too strong) and not "we discovered this pattern"
(D-031's original, more strongly wrong framing).

**Why this matters enough to log as its own decision rather than silently
editing D-031:** `decisions.md` is append-only by convention — corrections
supersede, they don't overwrite. This entry exists so a reader following D-031
sees the correction rather than inheriting an overstated claim silently fixed
elsewhere.
## D-034 · 2026-08-19 — Worst-client accuracy is the primary metric; global is secondary

**What:** Following the diagnostic session (D-028, `docs/diagnostic_report.md`),
worst-client accuracy is reported first everywhere going forward, with global
accuracy reported second. Decided by Prithvi.

**Why:** the diagnostic showed global accuracy stays flat across the entire alpha
sweep for both classical models, while worst-client accuracy declines monotonically
for both -- the global metric was hiding the effect this project measures.

**Model roles confirmed:** the 17-parameter MLP (D-026) is the matched classical
comparator for the actual 18-parameter VQC (D-004). Logistic regression remains in
all reporting as the convex reference, not replaced by the MLP.

**Numbering note:** this session's instructions again referenced decisions "pasted
separately" that did not arrive in the actual message (third occurrence of this
pattern -- see the numbering-reconciliation entries earlier in this file). This entry
records only what was stated inline in the actual message received.

---

## D-035 · 2026-08-19 — Primary metric changed to worst-client performance

Logged verbatim, per instruction:

Worst-client accuracy (minimum across clients, evaluated on each client's own
held-out data) becomes the primary outcome metric; global pooled accuracy is
retained as secondary. Evidence: with the noise floor reduced to 0.3-1.0pp via
5-fold CV x 10 seeds, global accuracy is flat across the entire alpha sweep for
logistic regression, while worst-client accuracy declines monotonically for both
model classes (LR 69.4%->64.8%; MLP 69.9%->51.5%). Mean pairwise client parameter
divergence also rises monotonically, confirming the mechanism directly rather than
by inference. Interpretation: the heterogeneity penalty at this scale is a
distributional/fairness effect rather than an average-performance effect.
Aggregation protects the global model while individual clients absorb the damage;
measuring only global accuracy conceals this entirely. Attribution caution:
degradation of client-level performance under non-IID conditions is established in
the federated fairness literature. We confirm it; we do not claim it.

*(This formalizes, with full evidence and attribution caution, what D-034 recorded
as a summary decision on 2026-08-18 -- not a duplicate, D-034's brief form and this
entry's verbatim form are both kept per the append-only rule.)*

---

## D-036 · 2026-08-19 — Convexity mediates whether the penalty surfaces globally

Logged verbatim, per instruction:

Evidence: the MLP shows a global accuracy drop at alpha=0.1 (76.9%->72.2%) that
logistic regression never exhibits, while both show worst-client degradation (MLP
18.4pp vs LR 4.6pp). Interpretation: for convex objectives, parameter averaging
approximates the pooled optimum largely independently of partitioning, confining
damage to individual clients. Non-convex models drift further and the damage
propagates to the global model. Consequence: the MLP (17 trainable parameters) is
the correct classical comparator for the VQC (18 parameters), matched in both
capacity and convexity. Logistic regression is retained as a reported convex
reference. Reporting both prevents this from being model selection.

---

## D-037 · 2026-08-19 — Natural institutional heterogeneity is milder than commonly-used synthetic settings

Logged verbatim, per instruction:

Evidence: the natural four-site partition does not exceed the synthetic Dirichlet
range on any metric, falling at approximately alpha = 0.5-1.0. Interpretation:
alpha = 0.1, widely used in the federated learning literature to represent
realistic non-IID conditions, is more severe than the heterogeneity observed in a
real four-site clinical archive. Synthetic partitioning at commonly-used
concentration values may overstate real-world heterogeneity. This directly answers
objective D-009 and contradicts our prior expectation that synthetic skew would
understate real institutional differences; the expectation was recorded before
measurement and is reported as refuted. Required hedging: single dataset, four
institutions, label-skew comparison. We report a calibration point, not a general
claim.

---

## D-038 · 2026-08-19 — Protocol: 5-fold stratified CV x 10 seeds

Logged verbatim, per instruction:

5-fold stratified CV x 10 seeds, replacing the single 736/184 split. Partitioning
occurs inside each fold's training set, leaving the federated protocol unchanged.
Noise floor 3.1pp -> 0.3-1.0pp.

*(Formalizes D-025's implementation-level entry from 2026-08-18 as a named
protocol decision.)*

---

## D-039 · 2026-08-20 — Arm 4 (VQC + FedAvg) results: trained properly, penalty magnitude between LR and MLP

**What:** Full 250-replicate sweep completed (5-fold CV x 10 seeds x 5 conditions),
identical protocol to the classical diagnostic. Full writeup: `docs/arm4_report.md`.

**Confirmed the VQC trained:** sanity check (Task 2, 2 clients, alpha=100, 15 rounds)
showed loss decreasing monotonically 1.096 -> 0.648, not plateaued. Not a barren
plateau or dead gradients.

**Worst-client accuracy declines monotonically for VQC too** (0.6488 -> 0.5763,
alpha=100->0.1, a 7.25pp drop), same qualitative pattern as both classical models.
In magnitude, the decline sits **between** the convex reference LR (4.66pp) and the
matched non-convex comparator MLP (18.46pp) -- more heterogeneity-sensitive than LR,
less than MLP, at every alpha tested. The quantum worst-client curve does **not**
fall faster than the MLP's -- it's roughly a third as steep.

**Wall-clock:** Arm 4 costs ~13,318x the matched MLP's per-run wall-clock (measured:
518.4s mean per VQC replicate vs 0.0389s mean per MLP replicate, same protocol,
directly timed for comparison). Total Arm 4 compute: 36.0 CPU-hours, 8.97 hours
actual wall-clock with 4-way parallelism (near-perfect ~4.0x speedup, better than
D-020's ~90% efficiency estimate -- this workload apparently has less inter-process
contention than the classical timing-spike workload did).

**Consequence for Arm 5:** per the pre-committed branch (session instructions),
VQC trained properly, so Arm 5 (circular-mean aggregation) proceeds. Its per-replicate
cost should be near-identical to Arm 4's (the aggregator function is O(n_clients),
not the bottleneck -- VQC local training dominates >99.9% of wall-clock either way),
so no fresh timing estimate was required before launching it -- see Arm 5 entry below.

---

## D-040 · 2026-08-20 — Arm 5 (VQC + circular-mean aggregation) built and launched

**What:** `scripts/aggregators.py:circular_mean` added -- `atan2(sum(w*sin(theta)),
sum(w*cos(theta)))` per parameter, weighted by client size. Only meaningful for
angle-valued parameters (the VQC's rotation angles); not applied to LR/MLP. The
D-007 ablation citing A2G-QFL.

**Against the frozen interface, no changes to `federated_loop.py` or the existing
`fedavg`:** verified by a direct smoke test (`circular_mean` passed to the
unmodified `run_federated`, produced correctly-shaped output) before building the
full worker/orchestrator (`scripts/arm5_worker.py`, `scripts/arm5_orchestrator.py`
-- near-exact copies of the Arm 4 versions, only the aggregator import and output
paths differ).

**Launched the full 250-replicate sweep** (same protocol as Arm 4: 5-fold CV x 10
seeds x 5 conditions) after a single-replicate correctness check (385.3s, consistent
with Arm 4's per-replicate timing). No fresh runtime estimate was requested from
Prithvi before this launch -- per the pre-committed branch instruction ("if VQC
trained properly, build Arm 5, run the same protocol") and because the cost is
already a known quantity from Arm 4 (same circuit, same rounds, same client
structure -- only the aggregator function changes, and that function is O(n_clients),
not the bottleneck). Results land in `results/runs_arm5.csv`
(`results/arm5_diagnostic_results.csv` / `results/arm5_diagnostic_divergence.csv`
after merging, same pattern as Arm 4) -- reported once the sweep completes.

---

## D-041 · 2026-08-20 — Arm 5 results: circular-mean aggregation makes no measurable difference

**What:** Full 250-replicate sweep completed (7.48hr wall-clock, 29.12 CPU-hr).
Full writeup: `docs/arm5_report.md`.

**Result:** circular-mean aggregation produces worst-client accuracy, global
accuracy, and client divergence essentially identical to FedAvg (Arm 4) at every
condition -- identical to 4 decimal places at alpha=100 and alpha=1.0, differences
in the 3rd decimal elsewhere, smaller than run-to-run seed noise. A reportable null
result, not an inconclusive one (250 replicates, tight agreement across the whole
sweep).

**Why, most likely:** circular mean only diverges numerically from a linear mean
near the angle wraparound boundary. Given the trained parameters (small learning
rate, narrow initialization, modest round count) most likely never range that
widely, the two aggregators end up computing near-identical averages in this
regime. Not verified by directly inspecting parameter trajectories -- a plausible
explanation, not a proven one.

**Answers the D-007 ablation:** circular-mean aggregation does not change
worst-client degradation relative to FedAvg in this data. The aggregator choice is
not what determines VQC heterogeneity sensitivity at this scale.

---

## D-042 · 2026-08-20 — Capacity control for Arm 4: weakened MLP, calibration mismatch, and a training degeneracy

**What:** Built a deliberately weakened classical MLP (hidden=1, 9 params; federated
rounds reduced 20->4; local epochs reduced 5->1) to test whether the VQC's smaller
worst-client decline vs the matched MLP (D-039) is a capacity artifact rather than a
model-family effect. Calibrated once, against alpha=100/seed=0 only, to match the
VQC's alpha=100 worst-client baseline (0.6488) before running the full sweep. Full
writeup: `docs/arm4_capacity_control_report.md`.

**Calibration did not generalize:** seed=0 gave 0.6483 (0.05pp from target); the
full 10-seed sweep gave 0.5236 (12.5pp undershoot). Not re-tuned after seeing this
-- reported as measured, per instruction ("match the baseline accuracy only, then
report what the sweep gives").

**Discovered a training degeneracy that invalidates the direct comparison:** the
weakened model's trained parameters are bit-identical across every alpha condition
(verified directly -- same seed/fold, different alpha, identical parameters to every
printed digit and identical `predict_proba` output). Cause: with `local_epochs=1`
(one full-batch gradient step per client per round) and FedAvg (size-weighted mean),
the aggregated update is mathematically identical to a single full-batch gradient
step on the *pooled* data, independent of partition, by linearity of the sum-of-
per-example-gradients each client computes. Confirmed this is real, not a bug: local
(pre-aggregation) client divergence is genuinely nonzero and rises with
heterogeneity (0.034->0.352, alpha=100->0.1) -- individual clients do diverge --
but aggregation exactly erases that difference in the global model under this
specific configuration.

**Consequence:** the weakened MLP's measured 36.77pp worst-client decline is real as
a number but is an **evaluation-composition effect** (one fixed model scored against
increasingly skewed per-client test slices), not a **training-heterogeneity effect**
(the VQC and full MLP each end up as genuinely different trained models per
condition; this config does not). Not directly comparable to the VQC's or full
MLP's declines for that reason. **Does not resolve the original capacity-confound
question.**

**Kept, not discarded:** the degeneracy itself is a useful finding for how future
capacity/budget calibration in this project should be checked (compare trained
parameters across conditions before trusting a reduced-budget degradation number).

---

## D-043 · 2026-08-20 — D-041 circular-mean explanation verified directly: confirmed

**What:** D-041 attributed the Arm 5 (circular-mean) vs Arm 4 (FedAvg) null result to
trained rotation angles never reaching the wraparound boundary, but this was
unverified -- the original sweeps saved metrics and a scalar divergence, not raw
parameter vectors. Re-ran a 20-replicate representative sample (5 conditions x 2
seeds x 1 fold, both arms) with the training loop reimplemented inline specifically
to capture every client's raw parameter vector every round. Full writeup:
`docs/arm5_angle_verification.md`.

**Result: confirmed.** Across 28,800 captured client angle values, max |theta| =
1.7927 -- 1.35 radians short of pi (3.1416), comfortably inside the region where
circular mean and linear mean agree. 0.0000% of values exceed 0.9*pi. Checked
specifically whether the most heterogeneous condition (alpha=0.1) pushes angles
closer to the boundary -- it doesn't; alpha=100 has the largest magnitudes in this
sample, no trend toward the boundary as heterogeneity increases.

**No alternative explanation needed.** The original account in D-041 was correct, not
just plausible -- driven by the small learning rate (0.1), narrow initialization
(0.1*N(0,1)), and modest round count (20), independent of alpha.

---

## D-044 · 2026-08-20 — Protocol parameters recovered from source (round count, worst-client evaluation)

**What:** Two protocol parameters, undocumented as explicit decisions until now, recovered
directly from the actual sweep code that produced Arms 1, 2, 4, 5 (not from the invalidated
E=1 capacity control, not inferred, not taken from markdown).

**Federated round count: 20.** Identical constant `FEDERATED_ROUNDS = 20` in
`scripts/run_grid.py:35`, `scripts/arm4_worker.py:28`, `scripts/arm5_worker.py:22`,
`scripts/run_diagnostic.py:38`. `LOCAL_EPOCHS = 5` (D-005) likewise consistent across all four.

**Worst-client accuracy is computed on each client's own held-out slice of that fold's test
set, not a shared/global test set.** `scripts/arm4_worker.py:62`:
`client_test = split_by_client(X_test, y_test, assign, df_test_fold)` -- `X_test`/`y_test` is
the fold's held-out 20% (fold-determined, not alpha-determined); `split_by_client`
(`scripts/cv_protocol.py:79-91`) slices it by the alpha-dependent client `assign`. Consequence
(flagged, addressed in Step 3 below): as alpha drops, per-client test slices become more
class-skewed, since the same alpha-dependent assignment determines both training partition and
test-slice membership. Some undetermined share of every reported worst-client degradation
number may reflect this evaluation-composition effect rather than a pure training effect --
not previously checked directly.

**Minimum is taken per (seed, fold), then averaged across the 50 replicates -- not pooled.**
The only committed-code source of this methodology is `scripts/plot_diagnostic.py:48-49`
(`groupby([...,"seed","fold"])["accuracy"].min()` then `groupby([...]).agg(["mean","std"])`).
Written for the classical diagnostic sweep; the identical pattern was replicated in ad-hoc
interactive analysis for the Arm 4/5 numbers in `docs/arm4_report.md`, but was never itself
saved as a script until this recovery -- a real documentation gap, now closed by this entry.

**Design change, this session:** the previous capacity control (D-042) plotted degradation
against alpha=100 baseline accuracy and treated that as a capacity axis. Rejected: baseline
accuracy is a joint outcome of capacity, architecture, convexity, and loss landscape, not
capacity alone -- our own data shows a many-to-one mapping (LR: 69.4% baseline, 4.6pp
degradation; MLP: 69.9% baseline, 18.4pp degradation -- same baseline, 4x the degradation).
Replaced with parameter count as the x-axis (controlled, family-neutral, but not a licensed
proxy either -- the LR point stays on the resulting plot specifically to bound how far it can
be trusted) and a bracketing framing rather than an explanatory one: not "does capacity explain
the VQC's degradation" but "does any MLP width reproduce it." See the full capacity-control
report (forthcoming this session) for the bracketing result.

---

## D-045 · 2026-08-20 — Worst-client methodology persisted as code (`scripts/worst_client.py`); a real bug found and fixed while verifying it

**What:** `scripts/worst_client.py` extracts the worst-client aggregation methodology
(D-044: per (arm, model, condition, seed, fold) minimum across clients, then mean/std
across replicates) into a reusable module reading any arm's long-format results CSV.

**Verification against Arm 4 / Arm 5 (single-arm files): exact match on first attempt.**
`results/arm4_diagnostic_results.csv` and `results/arm5_diagnostic_results.csv` each
contain only one arm, so grouping by model alone happened to be safe there. Reproduced
every published number in `docs/arm4_report.md`/`docs/arm5_report.md` to 4 decimal
places.

**Verification against the classical diagnostic (`results/diagnostic_results.csv`):
failed on first attempt, a real bug, not a data problem.** That file contains BOTH
arm1 (centralized, evaluated per-condition) and arm2 (federated, trained
per-condition) rows sharing the same `model` label ("LR", "MLP"). The first version of
`worst_client.py` grouped by `(model, condition, seed, fold)` only -- no `arm` column
-- which silently pooled arm1 and arm2 rows together, producing numbers that did not
match `docs/diagnostic_report.md` (e.g. LR alpha=100: 0.6917 reproduced vs 0.6942
published, a 0.25pp discrepancy that would have been larger elsewhere, e.g. alpha=0.1:
0.6424 vs 0.6476). **Not silently adjusted** -- root-caused (confirmed via
`df['arm'].value_counts()` showing 200 arm1 + 200 arm2 rows at exactly the condition
where the mismatch appeared), fixed by adding `arm` as a required grouping key, and
re-verified: now reproduces every published arm2 number in `docs/diagnostic_report.md`
exactly. Comment left in the module explaining why `arm` is mandatory, so this doesn't
regress.

**Consequence:** if a future results CSV analysis groups by `model` without `arm`, and
that CSV contains more than one arm sharing a model label, it will silently produce
wrong numbers the same way. `scripts/worst_client.py` is now the canonical, tested
entry point -- prefer it over ad-hoc groupby analysis for any future worst-client
reporting.

---

## D-046 · 2026-08-20 — E=5 confirmed to produce a genuine training effect (unlike E=1)

**What:** Reused already-captured angle data (`results/angle_capture/arm4_100_0_0.npz`,
`arm4_0.1_0_0.npz`, from D-043's verification sample) rather than re-running anything.
Compared the final-round aggregated global parameters at alpha=100 vs alpha=0.1, same
seed (0) and fold (0).

**Result:** max absolute elementwise difference = 0.9148 -- materially different, not
bit-identical. Confirms E=5 (`LOCAL_EPOCHS=5`, the real sweep's value, D-044) does NOT
reproduce the E=1 degeneracy found in the invalidated capacity control (D-042), where
the aggregated model was identical across every alpha condition. The real Arm 1/2/4/5
sweeps measure a genuine training effect, not pure evaluation composition, at the
protocol level -- though D-047 (below) shows composition still contributes a real,
non-negligible share of the *reported worst-client movement*, which is a distinct
question from whether the model itself changes at all.

---

## D-047 · 2026-08-20 — Composition-vs-training decomposition: dominant for LR, minor for MLP, VQC pending

**What:** Decomposed observed worst-client decline (alpha=100->0.1) into a
composition-only component (fixed alpha=100-trained model, evaluated against every
condition's test-slice composition) and a residual training component, for LR and MLP
(`scripts/composition_decomposition.py`, full 10-seed x 5-fold protocol, federated
training via FedAvg matching the real Arm 2 protocol exactly).

| model | observed decline | composition-only decline | % of movement that is composition | residual (training effect) |
|---|---|---|---|---|
| LR | 4.66pp | 3.82pp | **82.0%** | 0.84pp |
| MLP | 18.46pp | 4.99pp | 27.0% | 13.47pp |

**This materially changes how LR's result should be read.** D-028/D-039 reported LR's
worst-client decline (4.66pp) as evidence the heterogeneity penalty is real but
invisible in global accuracy. That's still true of the *global* flatness finding, but
the worst-client decline itself is now shown to be **82% evaluation-slice composition,
not training heterogeneity** -- LR's model barely changes with alpha (consistent with
its convexity), and most of what looked like "LR degrades a little" is actually "the
same LR model, scored against increasingly skewed held-out slices, naturally produces
a lower minimum by chance of which slice is worst." MLP's decline is mostly real
(73% residual training effect) by contrast.

**VQC decomposition running as of this entry** (50-replicate re-run at alpha=100 only,
evaluated against all 5 conditions' slicing, `scripts/vqc_composition_worker.py` /
`vqc_composition_orchestrator.py`) -- reproducibility of VQC training confirmed exactly
before launching (seed=0/fold=0/alpha=100 retrained, bit-identical to
`results/arm4_partial/100_0_0.json`). Result to follow in a subsequent entry.

**Committed locally, not pushed, per instruction.**

---

## D-048 · 2026-08-20 — Pre-registered prediction: VQC composition-vs-training decomposition

**What:** Recording a prediction before reading the VQC composition decomposition
result, which finished computing (50/50 replicates, 5616.7s) but has not yet been
analyzed as of this entry. Pre-registered by Prithvi, logged verbatim in intent:

If the VQC is genuinely low-capacity, its composition share should resemble LR's
(~82%) and its 7.3pp observed worst-client decline should shrink to something small
once the composition-only component is subtracted out. If instead the residual
(training-effect) component stays substantial, the capacity confound explanation for
the VQC's smaller-than-MLP degradation weakens considerably -- because it would mean
the VQC's decline is mostly a real training-heterogeneity effect, not an artifact of
being a weak/near-constant predictor scored against skewed slices, the same way MLP's
is mostly real.

**Why this is the prediction that matters most:** every other prediction in this
project's decision log has been about a specific mechanism (wraparound boundary,
E=1 degeneracy, etc.); this one is about whether the entire capacity-confound
concern that motivated D-042's redesign (D-044's rejection of the accuracy-axis
framing, the pending Step 4 parameter-count scatter) is still live at all. A small
VQC residual would suggest the scatter is designed around an effect that's mostly
not there. A substantial residual keeps the original question open and the scatter
meaningful as scoped.

**Result to follow in the next entry, computed after this one was written.**

---

## D-049 · 2026-08-20 — VQC composition decomposition result: prediction resolved, capacity confound not weakened

**What:** Completed the composition-only decomposition for VQC (50 replicates, alpha=100
training only, evaluated against all 5 conditions' test-slice composition,
`scripts/vqc_composition_worker.py`). Full comparison against the pre-registered
prediction (D-048):

| model | observed decline (a=100->0.1) | composition-only decline | % of movement that is composition | residual (training effect) |
|---|---|---|---|---|
| LR | 4.66pp | 3.82pp | 82.0% | +0.84pp |
| MLP | 18.46pp | 4.99pp | 27.0% | +13.47pp |
| VQC | 7.25pp | 8.50pp | **117.2%** | **-1.25pp** |

**Prediction outcome:** the "genuinely low-capacity" branch is confirmed, more strongly
than the prediction anticipated. VQC's composition share (117.2%) exceeds LR's (82.0%),
and the residual does not merely shrink -- it goes negative. Paired per-replicate
analysis (n=50, matched by seed/fold): mean paired residual = -0.0124 (SE=0.0107), i.e.
not strongly distinguishable from zero but clearly not substantially positive. **There is
no positive VQC-specific training-heterogeneity effect left once composition is
accounted for.** The capacity confound this control set out to test is not weakened by
this evidence -- if anything it is strengthened: the VQC's smaller-than-MLP observed
decline (D-039) is now explained by evaluation composition to at least the same degree
as LR's, not by a genuine intermediate training-heterogeneity sensitivity.

**Composition-only decline differs across models despite identical test slices** (LR
3.82pp, MLP 4.99pp, VQC 8.50pp) -- checked directly. Not explained by prediction
confidence magnitude: MLP's fixed alpha=100 model is more confidently separated than
LR's (mean |P-0.5|=0.29 vs 0.23, 9.2% vs 20.7% of predictions within 0.1 of the decision
boundary, one representative seed/fold), yet still shows the larger swing. Not explained
by class-recall asymmetry either (LR: 8.7pp recall gap between classes; MLP: 7.3pp,
similar). VQC's much larger swing is most simply explained by it being the weakest
baseline of the three fixed models (64.9% vs ~69% for LR/MLP) -- a weaker model's
predictions are more sensitive to which class happens to dominate a given slice. The
modest LR-vs-MLP gap itself does not trace to a single clean factor identified here.

---

## D-050 · 2026-08-20 — SUPERSEDES D-028 (in part): LR's worst-client decline was 82% evaluation composition

**D-028 is not edited or deleted** -- this entry supersedes one specific claim within it,
per the append-only rule.

**What D-028 claimed (2026-08-18):** "worst-client accuracy declines monotonically as
alpha falls, for both LR and MLP," presented under decision-table row 2 ("penalty
exists; we were measuring in the wrong place") -- i.e. LR's 4.66pp worst-client decline
was reported as evidence of a genuine heterogeneity penalty on individual clients, distinct
from and correcting the flat global-accuracy finding.

**What the decomposition showed (D-047, 2026-08-20):** LR's 4.66pp observed decline is
82.0% evaluation-slice composition (a fixed alpha=100-trained model, scored against
increasingly skewed held-out slices) and only 0.84pp (18.0%) a genuine training-time
heterogeneity effect. LR's model barely changes with alpha at all -- consistent with,
and now better explained by, its convexity (D-036) -- but the *worst-client accuracy
number itself* mostly reflects which specific slice happens to be evaluated, not how
much the model changed.

**Net effect:** row 2 of D-028's decision table ("penalty exists, wrong measurement
location") is **not wrong for MLP** (73% of its decline is real, D-047) but **was
overstated for LR**, where the "wrong measurement location" framing implied a real,
substantial hidden penalty that mostly is not there. The corrected reading: LR is
close to what D-028's row 1 already described ("mechanism fires, model absorbs it")
at the worst-client level too, not just the global level -- LR absorbs the
heterogeneity almost entirely, with only a small residual client-level effect.

---

## D-051 · 2026-08-20 — SUPERSEDES D-039 (in part): VQC's decline is not intermediate training-heterogeneity sensitivity

**D-039 is not edited or deleted** -- this entry supersedes one specific claim within it.

**What D-039 claimed (2026-08-20):** "In magnitude, the [VQC's 7.25pp] decline sits
**between** the convex reference LR (4.66pp) and the matched non-convex comparator MLP
(18.46pp) -- more heterogeneity-sensitive than LR, less than MLP" -- framed as the VQC
occupying a genuine intermediate position on a training-heterogeneity-sensitivity
spectrum between the two classical references.

**What the decomposition showed (D-049, 2026-08-20):** VQC's composition-only decline
(8.50pp) *exceeds* its observed decline (7.25pp) -- composition accounts for 117.2% of
the observed movement, and the residual training effect is slightly negative (-1.25pp,
not strongly distinguishable from zero given a paired SE of 1.07pp, but clearly not
substantially positive). The VQC shows **no positive training-heterogeneity effect**
once composition is accounted for -- more decisively than LR (82% composition), not an
intermediate case between LR and MLP on the training-effect axis at all.

**Net effect:** D-039's "sits between LR and MLP" claim remains numerically true of the
**observed** decline (7.25pp does fall between 4.66pp and 18.46pp), but the mechanistic
interpretation -- an intermediate degree of genuine heterogeneity-sensitivity -- is not
supported. The VQC's observed position between LR and MLP is better explained as: its
baseline (alpha=100) accuracy is the lowest of the three (64.9% vs ~69%), which alone
produces the largest composition-only swing of the three (D-049), and there is no
additional real training effect layered on top, unlike MLP's (D-047). This also
resolves D-042's original capacity-confound question, which the redesigned control
(D-044 onward) set out to test: the evidence does not support model family (quantum vs.
classical) as the explanation for the VQC's smaller observed decline than MLP's --
composition/baseline-accuracy differences explain it at least as well.

---

## D-052 · 2026-08-20 — Methodology note: the Arm1/Arm2 pooling bug is the argument for the reproducibility rule

**What:** D-045 found and fixed a real bug (worst-client numbers silently pooling arm1
and arm2 rows sharing a model label, because the first version of the extraction module
grouped by `model` without `arm`) while persisting the analysis as code. This entry
records why that episode belongs in the paper's Methodology section, not just the
decisions log.

**The argument:** the bug was invisible for two full days (2026-08-18 to 2026-08-20)
across multiple reports (`docs/diagnostic_report.md`, `docs/arm4_report.md`,
`docs/arm5_report.md`) and several rounds of interactive analysis, because the
ad-hoc computation that produced the published numbers happened to group correctly
by coincidence of how each query was written, while a from-scratch reimplementation
of "the same" methodology did not. **It was only caught because the numbers were
regenerated from committed code and diffed against the published values** -- an
audit that would not have happened under a weaker standard ("looks right," "matches
what I remember," or citing the report rather than the CSV). This is the concrete
case for CLAUDE.md's rule that no number goes in the paper unless it traces to a row
in a committed results file *and* to code that can regenerate it: the rule is not
process theater, it caught a real, silent, two-day-old error the first time it was
actually exercised.

---

## P-001 · 2026-08-20 — Per-person decision-ID prefixes adopted, to prevent numbering collisions

**What:** Following the D-029-047/D-029-033 numbering collision between this branch and
Ayuvi's (resolved by renumbering this branch's entries to D-034-052 -- see the commit
"docs: renumber D-029-047 -> D-034-052..."), decision IDs going forward use per-person
prefixes instead of a single shared sequential counter:

- **Prithvi's new entries: `P-001` onward** (this entry is `P-001`).
- **Ayuvi's new entries: `A-001` onward.**
- **All existing `D-*` numbers (D-001 through D-052 on this branch; D-001-028 and
  D-029-033 on main) are frozen as historical.** Not renumbered, not reused, not
  continued. Any new entry, by either person, gets a `P-` or `A-` number, never a new
  `D-` number.

**Why:** the collision happened because both branches independently continued the same
shared `D-NNN` counter from a common fork point, with no coordination mechanism for
allocating the next number across parallel branches. Per-person prefixes make that
structurally impossible -- each person's counter is theirs alone, so two people working
in parallel can never collide on an ID again, regardless of how long branches diverge
before merging.

**Consequence:** cross-references to old entries keep their `D-` prefix and are never
renumbered again (the opposite of what just happened to D-029-047 -- that renumbering
was a one-time cleanup of a collision that existed before this rule, not a precedent for
renumbering being routine). Anyone citing a decision in paper prose should always use the
full ID (`D-018`, `P-003`, `A-002`, etc.) so it stays unambiguous regardless of which
branch or session it originated in.

**Recorded in `CLAUDE.md`** so both Claude instances (this one and whichever instance
Ayuvi's sessions use) follow the convention without being told each time.

---

## P-002 · 2026-08-20 — Capacity scatter (Step 4) skipped: its own contingency fired

**What:** The parameter-count capacity scatter (5 MLP widths x 4 alpha x 30 seeds =
3000 runs, redesigned per D-044 onward) was never run. Decided by Prithvi.

**Why:** the scatter's launch was explicitly contingent from the start: "if most of
the VQC's decline turns out to be composition, tell me before running -- the scatter
was designed around a 7.3pp effect and may be measuring something much smaller." The
VQC composition decomposition (D-049) showed composition accounts for 117.2% of the
observed 7.25pp decline, with a residual of -1.25pp (not distinguishable from zero,
clearly not substantially positive). There is no positive training-heterogeneity
effect left to bracket. The 7.3pp effect the scatter was designed to explain does not
exist as a real quantity once composition is subtracted out -- the study, as scoped,
no longer has a question to answer.

**Not a failure of the contingency plan -- the plan worked exactly as designed.** The
whole point of pre-declaring "tell me before running" was to avoid spending a 3000-run
compute budget on a scatter built around an effect that might not survive the
decomposition. It didn't survive. Stopping here is the contingency firing correctly,
not an incomplete task.

**Consequence:** no MLP-width sweep exists in this repo. If a future question arises
that genuinely needs a parameter-count scatter (e.g. testing capacity effects on a
*different* quantity that does show a real residual), it should be redesigned around
whatever that quantity's real, decomposed effect size actually is -- not resurrected
from this scoping, which was sized for an effect now known not to exist.

---

## P-003 · 2026-08-22 — Arm 3 (FedProx) scoped to MLP only

**What:** Arm 3, the last unbuilt arm, is scoped to the MLP alone -- not LR, not the
VQC. Decided by Prithvi.

**Why:** FedProx's proximal term exists to restrain client drift and recover training
damage caused by it. The composition-vs-training decomposition (D-050, D-051/D-049)
showed LR's residual training effect is small (+0.84pp, 82% of its observed decline
is evaluation composition) and the VQC's residual is negative/indistinguishable from
zero (-1.25pp, composition exceeds 100% of its observed decline). Neither has
meaningful genuine drift damage for a proximal term to act on. Only MLP (+13.47pp
residual, D-047) has real damage. Building FedProx for LR or VQC would measure a
null effect layered on an already-established null effect -- not worth the compute
for the last arm.

---

## P-004 · 2026-08-22 — FedProx anchor: verified set_params-immediately-before-fit, no interface change needed

**What:** Before writing `FedProxMLPModel`, verified the precondition Prithvi
specified: does `federated_loop.py` call `set_params` with the global vector
immediately before every `fit()` call? Confirmed at `scripts/federated_loop.py`
lines 52-53 -- `local_model.set_params(global_params.copy())` directly followed by
`local_model.fit(X_c, y_c, epochs=local_epochs)`, no code between, every round,
every client.

**Consequence:** `fit()` can snapshot its own current parameters as its first action
and that snapshot is exactly the round's true global vector -- no staleness, no
interface change. The proximal term (`mu * (theta - anchor)` added to each
parameter's gradient, anchor = the snapshot) lives entirely inside
`FedProxMLPModel.fit()` in `scripts/models_mlp.py`. `federated_loop.py` and
`scripts/aggregators.py` were not touched.

**Verified, not just reasoned about:** ran `FedProxMLPModel` with `mu=0.0` through
the unmodified `run_federated` and confirmed it reproduces plain `MLPModel`'s trained
parameters bit-for-bit (`np.allclose` true to full precision) -- the proximal term
correctly vanishes at mu=0, and the implementation is provably compatible with the
frozen loop, not just argued to be.

---

## P-005 · 2026-08-22 — Arm 3 results: FedProx recovers 5-17% of MLP's residual damage, non-monotonically in mu

**What:** Full 750-replicate sweep (mu in {0.01, 0.05, 0.1}, same protocol as every
other arm) plus the composition decomposition, same method as D-044 onward. Full
writeup: `docs/arm3_report.md`.

**Result:** FedProx reduces MLP's residual training-heterogeneity effect from
+13.47pp (FedAvg, D-047) to +12.72pp (mu=0.01), +11.14pp (mu=0.05), +12.80pp
(mu=0.1) -- a 5.0-17.3% relative reduction, never close to eliminating the damage.
**Not monotonic in mu:** mu=0.05 (the literature-recommended value for this dataset)
shows the largest reduction; mu=0.01 and mu=0.1 show smaller, similar reductions to
each other. Reported as found, not smoothed into a monotonic story.

**Paired significance check** (same seed/fold, FedProx minus FedAvg worst-client
accuracy at alpha=0.1): mu=0.01 +0.87pp (SE 1.53pp, ~0.6 SE from zero), mu=0.05
+2.93pp (SE 1.91pp, ~1.5 SE), mu=0.1 +2.05pp (SE 1.77pp, ~1.2 SE). None reach
conventional significance. mu=0.05 comes closest, consistent with but not strong
confirmation of the literature-recommended value being well-chosen.

**FedProx does its mechanistic job -- divergence drops monotonically and cleanly**
(1.1088 at mu=0 -> 0.9578 at mu=0.1, final round, alpha=0.1) **but this does not
translate into a correspondingly large or monotonic recovery of worst-client
accuracy.** mu=0.1 restrains divergence far more than mu=0.01 (13.6% vs 2.9% lower
than FedAvg) but shows a near-identical residual (+12.80pp vs +12.72pp). Whatever
connects client-parameter divergence to worst-client accuracy damage is not a simple
monotonic relationship in this data -- stated as an open observation, not explained
further here.

**mu was not tuned toward a preferred outcome.** All three values (0.01, published
work's 0.05, and 0.1 as an upper comparison point) are reported as run, including the
non-monotonic pattern that a cleaner-looking report might have been tempted to
smooth over.

---

## P-006 · 2026-08-22 — Shared-test validation: LR/VQC decomposition corroborated, MLP magnitude disputed 3x

**What:** A second, independent estimate of the training effect per model (LR, MLP, VQC),
using a pooled held-out test set that is constant across alpha (no per-client
composition possible by construction), compared against the composition-decomposition
residuals (D-044 onward). No retraining -- the shared/global-accuracy metric already
computed for every arm turns out to already satisfy this protocol exactly
(`scripts/cv_protocol.py:fit_transform_fold` takes no alpha argument; verified
empirically bit-identical across repeated calls for the same seed/fold). Full writeup:
`docs/shared_test_validation.md`.

**Result:**

| model | shared-test decline | composition residual | agreement |
|---|---|---|---|
| LR | -0.26pp (1.6 SE from zero) | +0.84pp | agree -- both ~zero |
| VQC | +0.35pp (1.0 SE from zero) | -1.25pp | agree -- both ~zero |
| MLP | +4.62pp (4.4 SE from zero, real) | +13.47pp | **disagree materially, ~3x gap** |

**LR and VQC corroborated cleanly** -- two independent methods, same conclusion (no
real training effect). **MLP: both methods agree real damage exists, disagree on
magnitude by ~3x.** Not adjusted to improve agreement, per instruction. One
plausible (not verified, not resolved) account: the two methods estimate different
statistics -- composition's residual is the training-driven change in *worst-client*
accuracy specifically, shared-test is the training-driven change in *pooled/average*
accuracy; if MLP's training damage concentrates on whichever client is worst-off
rather than spreading evenly, a larger worst-client effect than pooled effect is what
that would look like, not necessarily an error in either method.

**Consequence, left for the Results write-up, not decided here:** MLP's true residual
training effect is somewhere between the two estimates (+4.6pp to +13.5pp), not a
single confirmed +13.47pp. Arm 3's FedProx results (P-005) were framed against the
higher figure -- whether that framing needs revision is Prithvi's call.

**Connects to established practice:** this shared-test protocol is exactly what
NIID-Bench (D-032, cited [8] in `paper/02_related_work.md`) uses -- their Table III
reports only pooled/global accuracy, never per-client.

---

## P-007 · 2026-08-22 — Mean-vs-minimum hypothesis for the MLP gap tested and rejected

**What:** Prithvi's hypothesis for P-006's MLP discrepancy: the composition residual
and the shared-test pooled decline are different statistics (minimum-over-clients vs.
mean), which need not agree even under a correct decomposition. Tested directly by
constructing a second, independent, alpha-*independent* fixed partition of each
fold's shared test set into 4 groups (plain seeded random split, same partition
reused for both alpha=100 and alpha=0.1 evaluation of a given replicate --
`scripts/shared_test_worst_group.py`), then taking the minimum accuracy across those
fixed groups instead of the pooled mean. No retraining for LR/MLP (exact existing
models reproduced deterministically); VQC used the smaller n=2 sample already saved
from the angle-capture work (D-043) rather than a ~14-hour full retrain, which was
not authorized and is flagged rather than run.

**Result:**

| model | shared-test pooled | shared-test worst-group (matched statistic) | composition residual |
|---|---|---|---|
| LR | -0.26pp | -0.06pp (SE 0.37pp, n=50) | +0.84pp |
| MLP | +4.62pp | **+5.00pp (SE 1.14pp, n=50, 4.4 SE from zero)** | **+13.47pp** |
| VQC | +0.35pp | -0.81pp (n=2 only) | -1.25pp |

**Hypothesis rejected.** For MLP, switching the statistic from mean to minimum --
while holding the test partition alpha-independent -- barely moved the number
(+4.62pp -> +5.00pp, well within each other's SE). It did not move toward the
composition residual's +13.47pp. The minimum operator itself is not what's driving
the gap.

**Revised account, per the instruction's own fallback:** what differs between the
decomposition's residual and both shared-test estimates is that the decomposition's
"composition-only" comparison uses the alpha-*dependent* Dirichlet client partition
(highly uneven group sizes and class composition, especially at low alpha), not a
plain even random split. This points to the minimum operator's sensitivity to the
*specific, widening, alpha-dependent* spread the Dirichlet partition produces, not
to minimum-vs-mean in general. **The composition-decomposition's MLP residual needs
an explicit caveat, not a clean +13.47pp citation.**

**Consequence for the paper's framing, not decided here:** the validated range for
MLP's genuine training effect narrows to roughly +4.6pp to +5.0pp (two independent,
mutually-consistent checks), with +13.47pp reported as the decomposition's own
figure, explicitly caveated as uncorroborated by either independent check run so
far. Whether Arm 3's framing (P-005) should be revised given this is Prithvi's call.

---

## P-008 · 2026-08-28 — Composition split recomputed under the shared-test training estimate; gap named as an unisolated model-partition interaction term

**What:** Recomputed the composition/training split for all three model families
using the shared-test training-effect estimate (P-006/P-007) in place of the
decomposition's own residual, and compared both splits side by side. Inputs: D-047/
D-049's observed decline and composition-only decline (fixed alpha=100-trained
model scored against every condition's real, alpha-dependent Dirichlet slice
composition); P-006/P-007's shared-test training-effect estimate (pooled decline
and matched-statistic worst-group decline, both on the alpha-independent shared
test set). `implied composition = observed decline − shared-test training effect`;
`implied composition % = implied composition / observed decline`.

| model | observed decline | decomposition composition-only (%) | decomposition residual (training effect) | shared-test training effect (pooled → worst-group) | implied composition under shared-test (%) | interaction (decomp residual − shared-test TE) |
|---|---|---|---|---|---|---|
| LR | 4.66pp | 3.82pp (**82.0%**) | +0.84pp | −0.26pp → −0.06pp | 4.72–4.92pp (**101–106%**) | +0.90 to +1.10pp |
| MLP | 18.46pp | 4.99pp (**27.0%**) | +13.47pp | +4.62pp → +5.00pp | 13.46–13.84pp (**73–75%**) | +8.47 to +8.85pp |
| VQC | 7.25pp | 8.50pp (**117.2%**) | −1.25pp | +0.35pp → −0.81pp | 6.90–8.06pp (**95–111%**) | −1.60 to −0.44pp |

**Composition dominates for all three families under the shared-test estimate —
stated plainly, since this is a stronger claim than previously reported.** Under the
decomposition alone, only LR (82.0%) and VQC (117.2%) looked composition-dominated;
MLP looked training-dominated (73% residual, 27% composition). Recomputed against
the shared-test training-effect estimate instead, **all three land at ≥~73%
composition** — LR ~101–106%, VQC ~95–111%, and MLP ~73–75%, inverting the
previously-reported MLP split (27% composition / 73% training → ~74% composition /
~26% training). This is consistent with the shared-test/decomposition agreement
already established for LR and VQC in P-006 and now extended to MLP by this
recomputation: composition, not training, is the larger driver of MLP's reported
worst-client decline once the training effect is pinned down independently.

**Framing the LR/VQC-vs-MLP discrepancy: not a failure of either method — the two
methods agree wherever the shared-test training effect is near zero, and diverge
only where a real training effect exists.** LR and VQC's interaction terms are small
(+0.9 to +1.1pp, and −0.4 to −1.6pp respectively) — both near the noise floor,
consistent with P-006's "agree" verdict for both. MLP's interaction term is
substantial and one-directional: **+8.47 to +8.85pp, roughly 8.5pp**, exactly the
scale flagged as a prediction before this recomputation. This pattern — agreement
when the training effect is ~0, divergence proportional to the training effect's
size — is what a **model-partition interaction term** absorbed into the
decomposition residual would look like: the decomposition's "composition-only" arm
is scored against the real, alpha-*dependent* Dirichlet partition (P-007's revised
account), so wherever a model's parameters genuinely shift with training, that shift
interacts with the same widening, uneven partition the composition-only arm also
uses — a term the decomposition's two-way split (composition vs. training) was never
built to isolate, and which `docs/decisions.md`'s own methodology notes already flag
as not isolated (see the "composition still contributes... which is a distinct
question" framing preceding D-047).

**No estimate declared correct.** Per instruction: MLP's genuine training effect is
reported as the range **4.6pp to 13.5pp**, bounded below by the shared-test estimate
and above by the decomposition residual, with the ~8.5pp gap between them named and
measured as the model-partition interaction term rather than resolved to a single
number. `docs/shared_test_validation.md` updated with this framing and the
three-family composition-split table.

---

## P-009 · 2026-08-28 — Second dataset (Diabetes 130-US Hospitals), classical arms only: characterisation complete, no per-record hospital ID exists

**Scope note first:** this project's own guardrails (`CLAUDE.md`) state "no extra
datasets." That line was written when the project was scoped single-dataset;
Prithvi has now explicitly directed, in-session, starting a second dataset for the
final stretch (no second semester, in-progress at Review-2, results at the final
review). Treated as the live instruction superseding the standing note, per
`CLAUDE.md`'s own "do not change without asking me" — this *is* Prithvi asking.
Logged here so the change of scope is traceable, not silently absorbed.

**What:** Downloaded and characterised Diabetes 130-US Hospitals (UCI,
`data2/SOURCE.md` for full provenance/citation/license — CC BY 4.0, same as
dataset 1). Data acquisition and characterisation only, per instruction — no
partitioning, no training. `scripts/dataset2_characterize.py`,
`results/dataset2_characterization.json`, full report in
`docs/dataset2_characterization.md`.

**Headline finding, reported plainly because it changes the task's premise:
there is no per-record hospital identifier in this dataset's public release.**
All 50 columns were inspected directly. "130 US hospitals" (Strack et al. 2014)
describes the source data warehouse (Cerner Health Facts), not an available
partition key in the released file — verified against the source publication,
not just inferred from a missing column. The closest categorical fields by
cardinality (`discharge_disposition_id`: 26, `medical_specialty`: 72 but 49.08%
missing, `admission_source_id`: 17) describe care pathway, not institution, and
using any of them as a stand-in for "hospital" would misrepresent the data.

**Other characterisation facts:**

| | |
|---|---|
| record count | 101,766 encounters, 71,518 unique patients (29.7% repeat-patient encounters — new leakage class, no analog in dataset 1) |
| features passing pooled ≥85%-present rule | 42 of 47 (dropped: `weight` 96.86% missing, `max_glu_serum` 94.75%, `A1Cresult` 83.28%, `medical_specialty` 49.08%, `payer_code` 39.56%) |
| new pathology missingness alone misses | 15 of 23 medication columns are >99% one value; 2 (`examide`, `citoglipton`) are 100% one value (zero variance) |
| target | `readmitted == "<30"` (binary early readmission), standard per the source paper; 11.16% positive — more imbalanced than dataset 1's target |
| class balance per hospital | not computable — no hospital field exists |

**Pipeline adaptation proposed (not implemented):** pooled ≥85% rule for
missingness plus a new near-zero-variance floor (threshold not yet chosen) for
the medication columns; Dirichlet(α) as the *only* client-construction method
for dataset 2 (no natural-partition arm is possible here — not manufactured
from a proxy field to preserve narrative symmetry with dataset 1);
`scripts/partitioner.py` already supports arbitrary client count `K`, so K=130
requires no new code.

**Pre-registered prediction, logged before measuring:** if the minimum
operator's sensitivity to partition spread (P-007/P-008's revised account of
the MLP gap) is a generic property of minimum-over-many-groups rather than
something specific to dataset 1's 4-client setup, then **130 Dirichlet clients
should show a substantially larger composition-only share of the observed
worst-client decline than 4-5 clients do, for the same model family and same
α.** Not tested here — characterisation only. Full statement and the
falsification condition in `docs/dataset2_characterization.md`.

**Holding, per instruction — no second-dataset experiment run yet.**

---

## P-010 · 2026-08-28 — Paper draft rewritten around the stronger "composition dominates all three" claim (P-008)

**What:** Rewrote the Abstract, §V-A, and the Conclusion of the paper draft
around P-008's finding. **Target-file note:** the instruction named
`paper_draft_v1.md`; that file (in `~/Downloads`, not tracked in this repo)
is the *pre-review* draft and still carries a known numerical error in its
VQC row (7.30/8.55pp vs. the correct 7.25/8.50pp) that was already found and
fixed in `paper_draft_v2.md` during the 2026-08-22 placeholder-review session
(see labbook). Edited `paper_draft_v2.md` instead — editing v1 would silently
reintroduce an already-corrected error — and flagged this substitution
explicitly rather than picking silently.

**Edits:** Abstract now states composition dominates all three model
families (not two of three), reports the LR/VQC agreement and the MLP's
4.6–13.5pp range with its ~8.5pp interaction term side by side, and does not
pick a winner. §V-A's table gained three columns (shared-test training
effect, implied composition under shared-test, interaction) alongside the
original three; prose explicitly states neither estimate is declared
correct. The Conclusion was rewritten to the same framing and recommends
that studies report both a decomposition and a shared-test estimate where
available, treating agreement as evidence of trustworthiness and
disagreement as evidence of the interaction term rather than of either
method being wrong.

**Flagged, not fixed:** §I's Contribution 2 and §II-B still say "two of
three" — out of the instructed scope (Abstract/§V-A/Conclusion only) and
now inconsistent with the rest of the document. Added as Flag 7 in the
draft's own "Flags for Author Review" section rather than silently edited
or silently left.

---

## P-011 · 2026-08-28 — Evaluation metric for dataset 2: balanced accuracy, chosen before any dataset-2 result was seen, and applied retroactively to dataset 1 for a matched comparison

**What happened:** Piloting dataset 2's federated LR/MLP grid (K=4, α=100
vs. α=1.0, one seed) before committing to the full run, both models
converged to "always predict not-readmitted" under plain 0.5-threshold
accuracy — confirmed not an undertraining artifact by running LR to 2,000
epochs of full-batch centralized gradient descent on the actual fold data:
the model's own maximum predicted probability *fell* with more training
(0.45 at 5 epochs → 0.18 at 2,000), moving further from the 0.5 threshold,
which is the genuine converged optimum for this feature set at an 8.8%
positive base rate, not a symptom of insufficient training. Under that
degenerate classifier, worst-client accuracy at every α is driven entirely
by each client's local class balance — the classifier itself never varies —
which mechanically produces ~100% composition / ~0% training effect for
every α pair tested. That is a property of the metric meeting a degenerate
classifier, not a finding about composition vs. training generalizing to
dataset 2, and reporting it as one would have been actively misleading.

**Decision, made by Prithvi before any dataset-2 grid result was seen (the
pilot above showed only that the metric was broken, not what any real
result under a fixed metric would be):** switch to **balanced accuracy**
(mean of per-class recall) in evaluation only — worst-client,
composition-only, and both shared-test statistics (pooled and worst-group).
Training is untouched: `models.py`/`models_mlp.py`'s `fit()` is not
class-weighted, no interface change, no new capability beyond what dataset
1 already has. This was Prithvi's call, not a default I picked — logged
here specifically so the ordering (metric fixed *before* seeing dataset-2
numbers, not adjusted after) is traceable, matching this project's own
evidentiary culture around not tuning results to a preferred outcome.

**Condition attached to the decision, also honored:** dataset 1's existing
composition-decomposition and shared-test results (D-047, D-049, P-006,
P-007) were computed under plain accuracy. Re-evaluating dataset 2 under a
different metric than dataset 1 would make the cross-dataset
methods-generalization claim (P-014) a comparison of different quantities,
not just different data. **Dataset 1 was therefore re-evaluated under
balanced accuracy too** — LR and MLP by deterministic reproduction (same
"no retraining needed" property already established in P-006: seeded
federated training is bit-identical on rerun, so this is re-evaluation of
the same trained models under a second metric, not a new experiment); VQC
from the existing n=2 saved-parameter sample (`results/angle_capture/
arm4_{100,0.1}_{0,5}_0.npz`), the same lower-powered substitute already used
in P-007 in place of an unauthorized ~14-hour full retrain (the 50-replicate
VQC composition-decomposition artifacts, `results/arm4_partial/*.json` and
`results/vqc_composition_partial/*.json`, store only accuracy/F1/AUROC
summary scalars, not raw predictions or parameters — insufficient to derive
balanced accuracy without retraining). Full comparison table (plain vs.
balanced accuracy, all three models, both decomposition and shared-test) in
`docs/dataset2_decomposition.md`.

---

## P-013 · 2026-08-28 — Balanced accuracy alone did not fix dataset 2: both models are constant classifiers everywhere; class-weighted training added, reversing part of P-011

**What happened after P-011:** running the full K=4/K=130 grid under balanced
accuracy produced bit-identical LR and MLP numbers at every single α/K pair
(e.g. both exactly 0.0pp decline at α=100→1.0, both exactly 2.0pp at
α=100→0.5 for K=4, both exactly 21.0pp at α=100→0.1). Investigated before
writing anything up: re-ran MLP the same way LR was checked in P-011 (up to
2,000 epochs of centralized full-batch training on real fold data) — MLP
**also** converges to a genuine constant "always predict not-readmitted"
classifier, identically to LR. Balanced accuracy correctly neutralizes
composition sensitivity where an evaluation slice contains both classes (a
constant classifier scores a flat, uninformative 0.5 there, by construction)
— but at extreme skew, single-class slices have no defined balanced accuracy
and fall back to plain accuracy, which *is* composition-sensitive. Since
neither model has a genuine training-driven residual (exactly 0.0pp at
every α/K pair, not merely small), 100% of every observed decline reported
under P-011's methodology was this single-class-slice fallback mechanism —
identical between LR and MLP because both are the same constant function.
The decomposition was vacuous: with no training effect at any K, the
pre-registered prediction (P-009) — composition share growing from K=4 to
K=130 — is untestable, since there was no room for it to grow from (already
100% at K=4).

**Decision, made before any weighted-training result was seen:** add
standard inverse-frequency ("balanced": w_c = n / (n_classes · n_c))
sample weighting to the gradient, recomputed fresh inside `fit()` from that
call's own `y` — new classes `WeightedLogisticRegressionModel` /
`WeightedMLPModel` (`scripts/models_weighted.py`), **not** an edit to
`models.py`/`models_mlp.py` in place, so every existing dataset-1 result
(Arms 1–5, D-001 through D-052, the paper draft) is untouched and still
traceable to the original unweighted implementation. This reverses part of
P-011 ("no interface change beyond dataset 1's existing capability") —
logged as a reversal, not a silent amendment, since the earlier call turned
out to be insufficient once measured rather than merely being a different
preference.

**Verification gate, run before committing to the full grid (per
instruction):** centralized fit on one real fold, both models. LR:
`pred_pos_rate` 0.65 (was 0.00), balanced accuracy 0.551. MLP: `pred_pos_rate`
0.48–0.41 depending on epoch budget, balanced accuracy 0.513 at 100 epochs
rising to ~0.58 (AUROC ~0.61) by 500–2,000 epochs. Both non-constant and
meaningfully above the 0.50 floor — gate passed, proceeding with the full
grid under the SAME protocol as before (20 rounds, 5 local epochs, no
learning-rate/architecture/feature changes, per the hard-stop condition).

**Condition attached, also honored:** dataset 1's LR/MLP composition
decomposition and shared-test estimate re-run a third time under weighted
training (reusing P-011's `dataset1_reeval_balanced.py` machinery
unchanged, only the model factory swapped to the weighted classes) so all
three states — original unweighted/plain-accuracy (D-047/D-049), unweighted/
balanced-accuracy (P-011), weighted/balanced-accuracy (this decision) — are
reported side by side for full traceability. VQC is untouched (not
requested, and dataset 1's target is close to balanced already).

**Hard stop, not exercised:** if weighted models had remained degenerate,
the instruction was to fall back to reporting dataset 2 as inconclusive
without tuning learning rate, architecture, or features to force a working
model. Not needed here — the gate passed on the first attempt.

Full numbers (dataset 1 three-way comparison, dataset 2 weighted K=4/K=130
grid) in `docs/dataset2_decomposition.md`.

---

## P-014 · 2026-08-28 — Dataset 2 grid complete: client-count prediction confirmed on the reliable metric; decomposition does not generalize as a standalone method

**What:** Full LR/MLP × K∈{4,130} × Dirichlet-α grid, class-weighted
training, balanced-accuracy evaluation, decomposition and shared-test both
applied, 10 seeds × 5 folds. K=130's α=0.1 cell excluded (verified
infeasible, not attempted — see P-009/P-010). Full table in
`docs/dataset2_decomposition.md` §2.

**Pre-registered prediction (P-009): confirmed, on the metric that can be
trusted.** The decomposition's own self-reported composition share does not
move consistently with K (grows at α=1.0, shrinks at α=0.5, for both
models) — but this is the decomposition's own residual becoming unstable
(negative residual, share >100%) at specific configurations, not a real
shrinking effect. The **shared-test-implied** composition share — computed
the same way as P-008's dataset-1 recomputation, and already established
there as the more trustworthy of the two — grows cleanly and substantially
from K=4 to K=130 in **all four** matched (model, α) comparisons available
at both client counts: LR 71–82%→93–94% (α=1.0), 73–78%→86–89% (α=0.5); MLP
73–78%→98–99% (α=1.0), 87–91%→96–98% (α=0.5). The minimum operator's
sensitivity to partition spread, the mechanism P-007/P-008 identified on
dataset 1, generalizes to a second dataset and a much larger client count.

**Second, unplanned finding, arguably the more important one: the
decomposition does not generalize as a standalone method.** On dataset 1,
LR agreed cleanly with the shared-test estimate (P-006/P-008) and only MLP
diverged, supporting a working assumption that the decomposition is
reliable except specifically where a model has a real training effect. On
dataset 2, **LR itself disagrees with the shared-test estimate at 3 of 5
measured configurations** — the decomposition's residual goes negative
(claiming composition more than fully explains the decline) at exactly the
points where the shared-test estimate finds a real, many-SE-from-zero
positive training effect (e.g. LR/K=4/α=0.5: decomposition residual
−3.10 pp vs. shared-test 1.04 pp, SE 0.24; LR/K=130/α=1.0: −4.11 pp vs.
0.95 pp, SE 0.20). The interaction term reaches −5 to −8.7 pp in these
rows — larger than dataset 1's entire MLP interaction (~8.5 pp, P-008) —
and its sign is not fixed (negative at 6 of 10 rows, positive at 4),
whereas dataset 1's interaction was uniformly positive. **The interaction
term is a property of the (model, partition, α) configuration, not of the
model family alone** — "LR is safe, MLP isn't" does not hold on a second
dataset. This directly reinforces P-010's rewritten Conclusion: the
decomposition should always be reported alongside a shared-test check, not
as a standalone estimate.

**Secondary finding from the class-weighting fix (P-013), applied to
dataset 1 too:** LR's genuine training residual under weighted training
moves from 0.84 pp (original, D-047) through 3.95 pp (weighted, balanced
accuracy) — part of dataset 1's original "LR barely trains" finding was
itself a smaller instance of the same under-incentivized-training mechanism
dataset 2 exposed at full scale under severe class imbalance. Both LR's and
MLP's decomposition residuals move *toward* their shared-test estimates
under weighting, not away — noted, not chased further; out of this task's
scope.

**What this does not test, restated:** no natural-partition arm is possible
(P-009, no per-record hospital ID); `A-003`'s natural-partition α ≈ 1.5
remains a dataset-1-only finding, neither confirmed nor challenged here.
Balanced accuracies achieved (LR ~0.55, MLP ~0.51–0.58) are modest and not a
claim about clinical model quality — no hyperparameter tuning was performed
(hard-stop condition honored).

---

## A-001 · 2026-08-22 — Evaluation-composition confound: literature check, largely unaddressed in the field

**What:** Literature and source-code search on whether the FL literature identifies and
separates the confound the decomposition analysis (D-044-D-052) found: because the same
client assignment determines both training partition and test slices, a fixed model's
worst-client accuracy can decline purely from evaluation-slice composition shifting with
α, independent of any genuine training effect. Full detail, per-source confidence levels,
and quoted code: `docs/reference/fl_evaluation_protocol_literature.md`.

**Result, stated plainly:** we did not find a paper that decomposes this. Checked four
sources directly, three at the source-code level (not just paper text):

- **pFL-Bench** (arXiv:2206.03655) — a comprehensive personalized-FL benchmark, confirmed
  (quoted) to use client-local train/test splitting with matching skew, with no evidence
  of decomposing composition from training effect. The strongest single piece of evidence
  this isn't already standard: a benchmark paper built specifically to standardize
  personalized-FL evaluation does not raise this issue.
- **q-FFL** (our own cited prior-art source for the disparity phenomenon, D-031/D-033) —
  verified directly from `generate_synthetic.py`: their synthetic-dataset train/test split
  happens *within* each device's own distribution, so their Table 10 numbers are
  themselves potentially exposed to this exact confound, unaddressed. This doesn't undo
  q-FFL as prior art for the disparity existing -- it means their numbers describe
  *observed* disparity, not necessarily one decomposed the way ours now is.
- **NIID-Bench** — verified directly from `utils.py`: uses a single shared/global test
  set, never partitioned per party. Our confound structurally does not apply to them, and
  this is also the mechanism behind D-032's finding that they never report per-client
  accuracy at all -- a shared test set makes a "per-client accuracy" number meaningless to
  compute in the first place.
- **Ditto** — verified from `fedbase.py`'s `test()` method: evaluates each client against
  its own local `test_data[u]`, not a shared set. Slightly lower confidence than q-FFL
  (codebase lineage inference for the specific per-dataset split, not an independently
  re-verified generation script), but the trainer-level mechanism is directly quoted, not
  assumed.

**Checked and rejected as already covering this:** the common personalized-vs-global
accuracy distinction in the pFL literature. That axis is about *which model* is evaluated
(personalized vs. shared); ours is about whether the *evaluation composition itself*
shifts with heterogeneity for a fixed model. Different question, not a relabeling.

**Consequence for the paper:** the decomposition (D-044-D-052) stands as this project's
strongest candidate original contribution -- not because the disparity phenomenon is
novel (it isn't, per D-031/D-033), but because separating composition from training
effect, specifically, does not appear to be established practice, checked at reasonable
effort across the closest four sources. Phrase as "we did not find," per instruction --
this is a four-source-plus-general-search check, not an exhaustive review.

**Numbering note:** this is the first `A-`-prefixed entry, per `P-001`'s new convention
(per-person prefixes; all `D-*` numbers, including this project's own D-029 through
D-033, are frozen as historical, not renumbered). Worth recording plainly: I should have
started at `A-001` for my very first decision entry this session (`CLAUDE.md`'s
per-person-prefix rule was already present when I onboarded), not continued the shared
`D-*` counter. That oversight is what produced the actual D-029-033/D-029-047 collision
`P-001` describes -- Prithvi had to renumber his own entries (D-029-047 -> D-034-052) to
resolve it. Flagged here rather than left implicit.

---

## A-002 · 2026-08-22 — plots.py completed against the full Arm 1-5 results; composition-decomposition figure added

**What:** Extended `scripts/plots.py` (built in D-030) to the now-complete result set and
added the primary decomposition figure per instruction. Two pieces:

1. **Arms 3, 4, 5 wired in.** Added `results/arm3_diagnostic_results.csv`,
   `arm4_diagnostic_results.csv`, `arm5_diagnostic_results.csv` (and their divergence
   counterparts) to `RESULTS_SOURCES`/`DIVERGENCE_SOURCES`. Verified their schemas match
   the original long-format columns exactly (`seed, fold, model, arm, condition, client,
   n, accuracy, f1, auroc` / `..., round, mean_pairwise_l2`) before adding -- no other code
   change was needed, confirming the extensibility design from D-030 actually holds up
   against real new arms, not just the hypothetical missing-file test done at the time.

2. **New script `scripts/composition_summary.py`** (does not edit any of Prithvi's
   existing decomposition scripts) consolidates the composition-vs-training decomposition
   (D-044-D-052) into `results/composition_decomposition_summary.csv`, so the numbers in
   `docs/arm4_report.md`'s headline table have a reproducible data source instead of only
   existing in that markdown table and one-off interactive runs. Reuses
   `composition_decomposition.composition_only_curve()` (imported, not reimplemented) to
   re-derive LR/MLP's composition-only curve (real federated training at alpha=100 only,
   ~9 seconds total, cheap) and aggregates the already-computed VQC per-replicate JSONs in
   `results/vqc_composition_partial/` (no training, just reading 50 existing files).
   **Verified the re-derived numbers against the already-reported ones before trusting
   them:** LR 4.666pp/3.825pp, MLP 18.462pp/4.998pp, VQC 7.251pp/8.493pp (observed/
   composition-only decline, alpha=100->0.1) -- match D-047/D-049's 4.66/3.82, 18.46/4.99,
   7.25/8.50 respectively (LR/MLP have tiny floating-point-scale differences from
   independent re-training; VQC matches to within rounding since it reads the same static
   files).

**New figure: `results/figs/composition_decomposition.png`.** Grouped bar chart, one
group per model (LR, MLP, VQC), three bars per group (observed decline, composition-only
decline, residual/training effect), in percentage points. Chosen over a line/scatter
design because the comparison is inherently three discrete numbers per model, and the
report's own headline table (`docs/arm4_report.md`) already uses exactly this
structure -- the figure should mirror the table it's illustrating, not invent a different
shape for the same comparison. VQC's residual bar visibly crosses zero (negative),
matching the "no measurable training-heterogeneity effect" finding directly.

**Title correction:** the worst-client figure's title previously read "(primary result)"
-- stale as of this task, since the decomposition figure is now the primary result per
instruction. Retitled to "(observed, pre-decomposition)" to make the relationship
explicit rather than leaving two figures both implicitly claiming to be primary.

**Not changed:** the worst-client/global-accuracy figure pair's design itself (matched
axes, natural-partition reference lines in the legend, shared color mapping) -- kept as
originally planned, per instruction, despite now carrying 9 series each (arm1 LR/MLP,
arm2 LR/MLP, arm3's three FedProx mu variants, arm4 VQC, arm5 VQC-circmean). Legibility
at this series count is workable but dense; flagged here rather than unilaterally
redesigning a figure the instruction explicitly said to keep as-is.

---

## A-003 · 2026-08-22 — Natural-partition alpha-calibration via total variation distance: 1.5 (95% CI 1.0-4.7), diverges from D-037's informal "~0.5-1.0"

**What:** Prompted by drafting the paper's natural-partition calibration claim, which needed a
value derived via a named formal distance metric rather than the informal comparison D-037
used. Wrote `scripts/alpha_calibration.py` (new file): weighted total-variation distance
between each client's local P(y=1) and the pooled P(y=1) (exact TV distance for a binary
label, = client-size-weighted mean of |p_k - p_global|), computed for the natural partition
(deterministic) and each Dirichlet alpha (mean/std across the same 10 seeds used everywhere
else in this project). Equivalent alpha via log-linear interpolation between the two
bracketing alpha conditions; 95% CI via percentile bootstrap (10,000 resamples of the
10 seed-level TV values per bracketing condition).

**Result:** natural partition TV distance = 0.1854. Dirichlet TV distances (mean, 10 seeds):
alpha=100: 0.0273; alpha=1.0: 0.2018; alpha=0.5: 0.2693; alpha=0.1: 0.3690. Natural brackets
between alpha=1.0 and alpha=100. Interpolated point estimate: **alpha ~ 1.54**. Bootstrap 95%
CI: **[1.00, 4.69]**.

**This diverges from D-037's "~alpha 0.5-1.0" language**, which is baked into every figure
legend in `scripts/plots.py`, `docs/decisions.md` D-037 itself, and this project's other
docs. Both can be simultaneously correct: D-037's comparison matched *downstream* metrics
(worst-client accuracy, parameter divergence magnitude) between natural and synthetic
conditions -- these reflect feature-distribution differences across real sites too, not just
label skew. This entry's TV-distance computation isolates *label-distribution* skew only,
which is all Dirichlet partitioning ever controls. Different question, can give a different
number. **Not resolved here** -- flagged for reconciliation before the paper's natural-
partition claim is finalized (which metric is "the" calibration number, stated explicitly,
not left as two unreconciled "natural ~ alpha X" claims across the project).

**Why total variation over Jensen-Shannon:** TV distance has an exact closed form for a
two-point (binary-label) distribution (=|p_k-p_global|); JS divergence would require a log
term and gives a different absolute scale without changing the qualitative
monotonic-in-alpha ordering. Documented choice, not an arbitrary one.

**CI width, reported honestly:** [1.00, 4.69] is wide, reflecting only 10 seeds and
substantial seed-to-seed variance at alpha=1.0 specifically (std=0.1049 against a mean of
0.2018, ~52% coefficient of variation). Not smoothed into a falsely precise-looking number.

---
