# Quantum vs Classical Federated Learning on Non-IID EHR

## Summary

This project measures how increasing data heterogeneity across simulated
hospitals degrades federated learning, and whether variational quantum
classifiers degrade differently from classical models under identical
conditions. It is a simulation-only characterization study — the goal is
to describe what happens as heterogeneity increases, not to demonstrate an
advantage for either approach. Built for a university capstone
(BCSE497J, VIT Vellore); see [Academic context](#academic-context) below.

## Status

*Last updated 2026-08-22 — keep this section honest and current; see
`docs/labbook.md` for the session that produced this state.*

**Implemented and validated:**
- Data loading for all 4 UCI Heart Disease sites (920 records)
- Preprocessing: 6-feature selection by worst-site availability, no PCA,
  train-only-fit imputation + scaling to the angle-encoding range
- Dirichlet(α) partitioner with a minimum-client-size guard, plus the
  natural 4-site partition as a reference condition
- Shared federated loop + interface contract (`docs/INTERFACE.md`, frozen,
  one additive amendment since — optional divergence tracking, D-027)
- Arm 1 (centralized classical) and Arm 2 (classical + FedAvg): logistic
  regression (convex reference) and a 17-parameter MLP (matched to the
  VQC's 18 parameters, capacity + convexity comparator)
- **Arm 4 (VQC + FedAvg): full sweep complete.** VQC trained properly
  (sanity-checked before trusting any accuracy number) — see
  `docs/arm4_report.md`.
- **Arm 5 (VQC + circular-mean aggregation): full sweep complete.**
  Circular-mean is statistically indistinguishable from FedAvg across the
  whole sweep (identical to 4 decimals at α=100/1.0) — a real null result
  for the D-007 ablation, see `docs/arm5_report.md`.
- 5-fold stratified CV protocol (10 seeds), identical across all arms —
  noise floor ~0.3–1.0pp, down from ~3.1pp
- Resumable, crash-safe experiment logging, validated with a real
  process-kill test (classical grid) and in production over two
  unattended overnight runs (Arm 4: 8.97hr, Arm 5: 7.48hr, zero failures)
- `scripts/plots.py`: worst-client accuracy, global accuracy, and client
  divergence vs. α figures (dpi=200), extensible to new arms by adding a
  source file path, degrades gracefully when an arm's results file doesn't
  exist yet (D-030)
- FL fairness / worst-client-disparity literature search (D-029,
  `docs/reference/fl_fairness_literature.md`) — establishes that the
  worst-client-vs-global-accuracy gap is a known phenomenon in the FL
  fairness literature, not a novel finding of this project
- Worst-client movement decomposed into evaluation-composition and
  training-effect components (D-044 onward) — see the revised headline
  below
- **Arm 3 (FedProx, MLP only): full sweep complete — the last arm.**
  Scoped to MLP since LR/VQC have no genuine training-effect residual for
  a proximal term to act on (P-003). Recovers 5-17% of MLP's residual
  training damage across μ ∈ {0.01, 0.05, 0.1}, non-monotonically, none
  of it statistically robust — see `docs/arm3_report.md`.

**Primary metric (D-034/D-035): worst-client accuracy, reported first.**
Global accuracy is secondary. Rationale: global accuracy is flat across
the entire α sweep for the convex reference and stays near-flat for the
matched MLP except at the most extreme skew, while worst-client accuracy
declines monotonically for every model tested (LR, MLP, VQC) — the
heterogeneity penalty this project measures is a distributional effect on
individual clients, largely invisible in a pooled global score.

**Headline comparison, revised after the composition-vs-training decomposition
(D-044 onward):** the VQC's raw observed worst-client decline (7.25pp,
α=100→0.1) originally looked intermediate between LR (4.66pp) and MLP
(18.46pp). Decomposing it into evaluation-composition (a fixed model
scored against increasingly skewed test slices) and a genuine training
effect shows the opposite: **composition alone accounts for 117% of the
VQC's observed decline — there is no measurable positive
training-heterogeneity effect left (residual -1.25pp, not distinguishable
from zero).** LR's residual is small but positive (+0.84pp, 82% is
composition); MLP's is large and real (+13.47pp, only 27% composition).
The VQC's residual is the smallest of the three, not intermediate. The
VQC is non-convex yet behaves like the convex LR on this axis — flagged
as an open question in `docs/arm4_report.md`, not resolved by speculation.
Wall-clock cost is unaffected by this revision: ~13,300x the MLP's, per
training run, on this simulator. Full numbers: `docs/arm4_report.md`.

**Two follow-up checks that led to the revision above, both reported with
their limitations rather than smoothed over:**
- A capacity-matched (weakened) MLP was built to rule out the VQC's
  original apparent robustness being a capacity artifact. The calibration
  didn't generalize from its single-seed test to the full sweep, and the
  weakened configuration turned out to make FedAvg training mathematically
  partition-invariant (verified directly against the trained parameters)
  — so that specific control doesn't cleanly answer the question. See
  `docs/arm4_capacity_control_report.md`. The composition decomposition
  above (a separate, later analysis) is what actually resolved it.
- The D-041 circular-mean null result's explanation (angles never reach
  the wraparound boundary) was checked directly against real trained
  parameters, not just asserted — confirmed: max angle magnitude observed
  is 1.35 radians short of π, with no trend toward the boundary as
  heterogeneity increases. See `docs/arm5_angle_verification.md`.

**Capacity scatter (originally planned as a parameter-count sweep over
several MLP widths) was skipped (P-002):** its own pre-declared
contingency fired — with composition explaining 117% of the VQC's
decline, there is no substantial residual effect left to bracket.

**Not started:** nothing — Arms 1 through 5 all exist and have full sweeps.
This project's compute is done; remaining work is writing.

**Known open items:** decision IDs D-009–D-014 are referenced in project
history but not recorded in `docs/decisions.md` — see
`docs/decisions_index.md` for the specifics.

## Repository map

```
data/           Raw UCI site files + SOURCE.md (provenance, license, citation)
scripts/        All pipeline code -- data loading, preprocessing, partitioning,
                models, aggregators, the federated loop, the experiment runner
docs/           Decisions log, lab book, frozen interface, this index
paper/          Paper section drafts (numbered, one file per section)
results/        runs.csv (every run, including throwaways) + figures (regenerable,
                not committed -- see .gitignore)
RUNNING.md      Command / expected output / failure signature per component
```

## Quickstart

```bash
python -m venv .venv
.venv/Scripts/activate          # or .venv/bin/activate on Linux/Mac
pip install pennylane pennylane-lightning torch scikit-learn pandas matplotlib

# reproduce the classical grid (Arm 1 + Arm 2, resumable -- safe to interrupt)
python scripts/run_grid.py

# reproduce a single component
python scripts/data_loader.py       # missingness + class balance report
python scripts/partitioner.py       # Dirichlet partition report + figures
python scripts/preprocessing.py     # train/test split + scaling sanity check
```

See `RUNNING.md` for the exact command, expected output, and failure
signature for every component individually.

## Key decisions

Full reasoning and rejected alternatives live in `docs/decisions.md`
(indexed in `docs/decisions_index.md`). The load-bearing ones:

- All 4 UCI sites used (920 records), not Cleveland alone — too few rows
  per federated client otherwise (D-001).
- Adjoint differentiation, not parameter-shift — 30–70x faster in
  simulation; parameter-shift alone would not finish in the compute budget
  (D-002).
- 6 features, no PCA — selected by worst-site data availability, mapped
  one feature per qubit directly (D-021, supersedes the PCA half of D-004).
- Dirichlet(α) partitioning is pooled across sites and client-count-matched
  (4) to the natural-site condition, so partitioning method is the only
  variable being compared (D-008, D-017).
- `chol`, `ca`, `thal` all dropped from the feature set — each fails the
  same missingness-consistency check applied to the others (D-006, D-007,
  D-016).

## Dataset

UCI Heart Disease dataset, all 4 processed site files (Cleveland,
Hungarian, Switzerland, VA). Licensed CC BY 4.0.

> Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart
> Disease [Dataset]. UCI Machine Learning Repository.
> https://doi.org/10.24432/C52P4X

Full provenance, retrieval date, and known data-quality caveats:
`data/SOURCE.md`.

## Team and roles

| Name | GitHub |
|---|---|
| Prithvi Singh Nathawat | [@PrithviSinghNathawat](https://github.com/PrithviSinghNathawat) |
| Ayuvi | *(to be added)* |

Individual contribution is evidenced by commit history — see
`docs/README.md` for the commit convention and how to generate a
per-author summary (`git shortlog -sn`).

## Academic context

BCSE497J Project-I/II, VIT Vellore. Guide: *(to be added)*.
