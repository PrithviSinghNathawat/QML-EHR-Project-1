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
| D-025 | 2026-08-18 | Diagnostic session: 5-fold CV with a stable per-seed client identity |
| D-026 | 2026-08-18 | MLP parameter count: 17, matched to the real VQC (18), not the requested ~36 |
| D-027 | 2026-08-18 | `run_federated` interface amendment: optional divergence tracking |
| D-028 | 2026-08-18 | Diagnostic session findings: heterogeneity penalty exists, measured in the wrong place |
| D-034 | 2026-08-19 | Worst-client accuracy is the primary metric; global is secondary |
| D-035 | 2026-08-19 | Primary metric changed to worst-client performance (verbatim, with evidence + attribution caution) |
| D-036 | 2026-08-19 | Convexity mediates whether the penalty surfaces globally |
| D-037 | 2026-08-19 | Natural institutional heterogeneity is milder than commonly-used synthetic settings (answers D-009) |
| D-038 | 2026-08-19 | Protocol: 5-fold stratified CV x 10 seeds (named protocol decision) |
| D-039 | 2026-08-20 | Arm 4 (VQC+FedAvg) results: trained properly, penalty magnitude between LR and MLP |
| D-040 | 2026-08-20 | Arm 5 (VQC + circular-mean aggregation) built and launched |
| D-041 | 2026-08-20 | Arm 5 results: circular-mean aggregation makes no measurable difference vs FedAvg |
| D-042 | 2026-08-20 | Capacity control for Arm 4: weakened MLP, calibration mismatch, training degeneracy found |
| D-043 | 2026-08-20 | D-041 circular-mean explanation verified directly: confirmed |
| D-044 | 2026-08-20 | Protocol parameters recovered from source (round count=20, worst-client = own-slice, per-fold-then-averaged); capacity-control redesign to parameter-count bracketing |
| D-045 | 2026-08-20 | Worst-client methodology persisted as code; real bug found (missing arm grouping key) and fixed |
| D-046 | 2026-08-20 | E=5 confirmed to produce a genuine training effect (max param diff 0.91, unlike E=1's bit-identical) |
| D-047 | 2026-08-20 | Composition-vs-training decomposition: 82% composition for LR, 27% for MLP, VQC pending |
| D-048 | 2026-08-20 | Pre-registered prediction: VQC composition share should resemble LR's if capacity explains the effect |
| D-049 | 2026-08-20 | VQC decomposition result: composition = 117% of decline, residual negative -- capacity confound not weakened |
| D-050 | 2026-08-20 | SUPERSEDES D-028 (in part): LR's worst-client decline was 82% evaluation composition |
| D-051 | 2026-08-20 | SUPERSEDES D-039 (in part): VQC's decline is not intermediate training-heterogeneity sensitivity |
| D-052 | 2026-08-20 | Methodology note: the Arm1/Arm2 pooling bug is the argument for the reproducibility rule |

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

**Numbering convention, 2026-08-20 onward:** all `D-*` numbers are now frozen
(historical, never renumbered or continued). New entries use per-person
prefixes: `P-001` onward for Prithvi, `A-001` onward for Ayuvi. See P-001 in
`docs/decisions.md` for why.

| ID | Date | Title |
|---|---|---|
| P-001 | 2026-08-20 | Per-person decision-ID prefixes adopted, to prevent numbering collisions |
| P-002 | 2026-08-20 | Capacity scatter (Step 4) skipped: its own contingency fired |
