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

## D-029 · 2026-08-19 — Worst-client accuracy is the primary metric; global is secondary

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

## D-030 · 2026-08-19 — Primary metric changed to worst-client performance

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

*(This formalizes, with full evidence and attribution caution, what D-029 recorded
as a summary decision on 2026-08-18 -- not a duplicate, D-029's brief form and this
entry's verbatim form are both kept per the append-only rule.)*

---

## D-031 · 2026-08-19 — Convexity mediates whether the penalty surfaces globally

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

## D-032 · 2026-08-19 — Natural institutional heterogeneity is milder than commonly-used synthetic settings

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

## D-033 · 2026-08-19 — Protocol: 5-fold stratified CV x 10 seeds

Logged verbatim, per instruction:

5-fold stratified CV x 10 seeds, replacing the single 736/184 split. Partitioning
occurs inside each fold's training set, leaving the federated protocol unchanged.
Noise floor 3.1pp -> 0.3-1.0pp.

*(Formalizes D-025's implementation-level entry from 2026-08-18 as a named
protocol decision.)*

---

## D-034 · 2026-08-20 — Arm 4 (VQC + FedAvg) results: trained properly, penalty magnitude between LR and MLP

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

## D-035 · 2026-08-20 — Arm 5 (VQC + circular-mean aggregation) built and launched

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

## D-036 · 2026-08-20 — Arm 5 results: circular-mean aggregation makes no measurable difference

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

## D-037 · 2026-08-20 — Capacity control for Arm 4: weakened MLP, calibration mismatch, and a training degeneracy

**What:** Built a deliberately weakened classical MLP (hidden=1, 9 params; federated
rounds reduced 20->4; local epochs reduced 5->1) to test whether the VQC's smaller
worst-client decline vs the matched MLP (D-034) is a capacity artifact rather than a
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

## D-038 · 2026-08-20 — D-036 circular-mean explanation verified directly: confirmed

**What:** D-036 attributed the Arm 5 (circular-mean) vs Arm 4 (FedAvg) null result to
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

**No alternative explanation needed.** The original account in D-036 was correct, not
just plausible -- driven by the small learning rate (0.1), narrow initialization
(0.1*N(0,1)), and modest round count (20), independent of alpha.

---

## D-039 · 2026-08-20 — Protocol parameters recovered from source (round count, worst-client evaluation)

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

**Design change, this session:** the previous capacity control (D-037) plotted degradation
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

## D-040 · 2026-08-20 — Worst-client methodology persisted as code (`scripts/worst_client.py`); a real bug found and fixed while verifying it

**What:** `scripts/worst_client.py` extracts the worst-client aggregation methodology
(D-039: per (arm, model, condition, seed, fold) minimum across clients, then mean/std
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

## D-041 · 2026-08-20 — E=5 confirmed to produce a genuine training effect (unlike E=1)

**What:** Reused already-captured angle data (`results/angle_capture/arm4_100_0_0.npz`,
`arm4_0.1_0_0.npz`, from D-038's verification sample) rather than re-running anything.
Compared the final-round aggregated global parameters at alpha=100 vs alpha=0.1, same
seed (0) and fold (0).

**Result:** max absolute elementwise difference = 0.9148 -- materially different, not
bit-identical. Confirms E=5 (`LOCAL_EPOCHS=5`, the real sweep's value, D-039) does NOT
reproduce the E=1 degeneracy found in the invalidated capacity control (D-037), where
the aggregated model was identical across every alpha condition. The real Arm 1/2/4/5
sweeps measure a genuine training effect, not pure evaluation composition, at the
protocol level -- though D-042 (below) shows composition still contributes a real,
non-negligible share of the *reported worst-client movement*, which is a distinct
question from whether the model itself changes at all.

---

## D-042 · 2026-08-20 — Composition-vs-training decomposition: dominant for LR, minor for MLP, VQC pending

**What:** Decomposed observed worst-client decline (alpha=100->0.1) into a
composition-only component (fixed alpha=100-trained model, evaluated against every
condition's test-slice composition) and a residual training component, for LR and MLP
(`scripts/composition_decomposition.py`, full 10-seed x 5-fold protocol, federated
training via FedAvg matching the real Arm 2 protocol exactly).

| model | observed decline | composition-only decline | % of movement that is composition | residual (training effect) |
|---|---|---|---|---|
| LR | 4.66pp | 3.82pp | **82.0%** | 0.84pp |
| MLP | 18.46pp | 4.99pp | 27.0% | 13.47pp |

**This materially changes how LR's result should be read.** D-028/D-034 reported LR's
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
