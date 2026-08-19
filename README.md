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

*Last updated 2026-08-18 — keep this section honest and current; see
`docs/labbook.md` for the session that produced this state.*

**Implemented and validated:**
- Data loading for all 4 UCI Heart Disease sites (920 records)
- Preprocessing: 6-feature selection by worst-site availability, no PCA,
  train-only-fit imputation + scaling to the angle-encoding range
- Dirichlet(α) partitioner with a minimum-client-size guard, plus the
  natural 4-site partition as a reference condition
- Shared federated loop + interface contract (`docs/INTERFACE.md`, frozen)
- Arm 1 (centralized classical) and Arm 2 (classical + FedAvg)
- Resumable, crash-safe experiment logging to `results/runs.csv`

**Validation status:** 4 of 5 gates pass cleanly (Arm 1 accuracy, no
leakage, Arm 2-vs-Arm 1 agreement at α=100, seed determinism). The
Arm 2 across-α-sweep gate does **not** show a clear monotonic decline —
flagged, not yet resolved, see `docs/decisions.md` D-024.

**Not started:** Arm 3 (FedProx), Arm 4 (VQC + FedAvg — the headline arm),
Arm 5 (VQC + circular-mean aggregation, timeboxed). Per `docs/INTERFACE.md`,
these should be addable without modifying the existing loop or aggregator.

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
