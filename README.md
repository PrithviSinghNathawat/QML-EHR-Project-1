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

*Last updated 2026-08-20 — keep this section honest and current; see
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
  (sanity-checked before trusting any accuracy number). Worst-client
  accuracy declines monotonically with α, magnitude between the convex
  and non-convex classical references — see `docs/arm4_report.md`.
- **Arm 5 (VQC + circular-mean aggregation): built, sweep running** —
  results pending.
- 5-fold stratified CV protocol (10 seeds), identical across all arms —
  noise floor ~0.3–1.0pp, down from ~3.1pp
- Resumable, crash-safe experiment logging, validated with a real
  process-kill test (classical grid) and in production over a real
  9-hour unattended run (Arm 4)

**Primary metric (D-029/D-030): worst-client accuracy, reported first.**
Global accuracy is secondary. Rationale: global accuracy is flat across
the entire α sweep for the convex reference and stays near-flat for the
matched MLP except at the most extreme skew, while worst-client accuracy
declines monotonically for every model tested (LR, MLP, VQC) — the
heterogeneity penalty this project measures is a distributional effect on
individual clients, largely invisible in a pooled global score.

**Headline comparison so far:** the VQC's worst-client decline (7.25pp,
α=100→0.1) does **not** fall faster than the matched MLP's (18.46pp) — it
sits between the MLP and the convex LR reference. Wall-clock cost: ~13,300x
the MLP's, per training run, on this simulator. Full numbers:
`docs/arm4_report.md`.

**Not started:** Arm 3 (FedProx) — in progress on a teammate's branch in
parallel; this branch does not touch `federated_loop.py`, `data_loader.py`,
`partitioner.py`, `docs/INTERFACE.md`, or any quantum file, by agreement.

**Known open items:** decision IDs D-009–D-014 are referenced in project
history but not recorded in `docs/decisions.md` — see
`docs/decisions_index.md` for the specifics. Arm 5 results pending.

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
