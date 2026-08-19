# Decisions Index

Navigation table for `docs/decisions.md` — one line per decision. Read the
full entry in `decisions.md` for the actual reasoning and rejected
alternatives; this table is a lookup, not a substitute.

| ID | Date | Title |
|---|---|---|
| D-001 | 2026-08-18 | Dataset: full UCI Heart Disease, all 4 sites (920 records) |
| D-002 | 2026-08-18 | Gradients: `diff_method="adjoint"`, not parameter-shift |
| D-003 | 2026-08-18 | Quantum device: `lightning.qubit`, not `default.qubit` |
| D-004 | 2026-08-18 | Circuit: 6 qubits, PCA to 6 components *(PCA half superseded by D-021)* |
| D-005 | 2026-08-18 | Local epochs per federated round: E=5, not E=1 |
| D-006 | 2026-08-18 | `chol == 0` treated as missing, at every site, not just Switzerland |
| D-007 | 2026-08-18 | `ca` and `thal`: dropped, not imputed |
| D-008 | 2026-08-18 | Dirichlet partitioner, pooled across sites *(client count superseded by D-017)* |
| D-009 | — | **Missing.** Referenced (objective: natural vs. Dirichlet partitioning comparison) but never recorded in this repo. |
| D-010 | — | **Missing.** Referenced (minimum-client-size guard) but not recorded when asked about; actually implemented under D-022 instead. |
| D-011 | — | **Missing.** Referenced, no known content. |
| D-012 | — | **Missing.** Referenced, no known content. |
| D-013 | — | **Missing.** Referenced, no known content. |
| D-014 | — | **Missing.** Referenced (Cleveland-only sensitivity analysis for chol/ca/thal exclusion) — analysis does not exist in this repo. |
| D-015 | 2026-08-18 | Grid size confirmed: full 4α × 5 seeds, no cut |
| D-016 | 2026-08-18 | `chol` dropped, consistent with D-007 *(superseded by D-021's full feature set)* |
| D-017 | 2026-08-18 | Client count fixed at 4 for both partitioning schemes |
| D-018 | 2026-08-18 | Dataset provenance, licensing, and two facts for the paper |
| D-019 | 2026-08-18 | Feature retention rule, applied consistently, fails the PCA-6 floor *(resolved by D-021)* |
| D-020 | 2026-08-18 | Run-level parallelism: 4 concurrent processes, ~90% efficiency |
| D-021 | 2026-08-18 | Final feature set: 6 features, no PCA, selected by worst-site availability |
| D-022 | 2026-08-18 | Minimum-client-size guard: floor=15 rows, reject-and-redraw |
| D-023 | 2026-08-18 | Federated loop + interface implementation (Arm 1, Arm 2) |
| D-024 | 2026-08-18 | Arm 1 / Arm 2 validation gate results |

**D-009 through D-014 are a known gap** — referenced in session instructions
across multiple sessions but never actually supplied as content, and the
corresponding code/analysis (a natural-vs-Dirichlet comparison objective, a
formally-documented minimum-client-size guard, a Cleveland-only sensitivity
analysis) does not exist in this repo except where a later decision
(D-022) independently rebuilt the same functionality under a new number.
See the "D-numbering reconciliation" note in `docs/decisions.md` (between
D-020 and D-021) for the full history. If this content exists elsewhere
(a separate document, a conversation not captured here), it should be
backfilled into `decisions.md` under these IDs rather than left as a gap.
