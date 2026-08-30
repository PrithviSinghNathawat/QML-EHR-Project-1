# Project Inventory

Compiled 2026-08-30. Factual inventory only — no interpretation, reframing, or
recommendation. Every number below is traced to a specific committed file,
named inline. Where a number required arithmetic on existing data (e.g.
computing a decline from two accuracy values already in a CSV), the source
file and the exact computation are named so it can be re-derived.

---

## 1. Results inventory

### 1.1 Composition-decomposition analyses (the table requested)

Columns: dataset, model, aggregation, client count (K), α pair, observed
worst-client decline, composition-only decline, decomposition residual,
shared-test training effect (pooled point estimate), shared-test-implied
composition share, whether the configuration clears the 2pp reporting floor
(§III-C of `paper/paper_draft_v2.md`, established P-021/P-022), and whether
the row currently appears in the paper.

Sources: `results/composition_decomposition_summary.csv` (dataset 1 LR/MLP
observed + composition-only, all conditions), `results/diagnostic_results.csv`
and `results/arm4_diagnostic_results.csv` (dataset 1 global/shared-test
accuracy, all conditions), `results/vqc_composition_partial/*.json` (dataset 1
VQC composition-only, all conditions), `results/dataset2_decomposition_weighted.json`
(dataset 2, all configurations). Dataset-1 rows beyond the α=100→0.1 headline
(i.e. 100→1.0, 100→0.5, 100→natural) were computed for this inventory by the
same arithmetic already used elsewhere in this project (`observed_decline =
100·(obs[100]−obs[α])`, etc.) — the underlying per-condition accuracy values
were already committed; the decline/share arithmetic on them had not
previously been written up as its own table.

| Dataset | Model | Aggregation | K | α pair | Observed (pp) | Comp.-only (pp) | Residual (pp) | Shared-test TE (pp) | Implied share (%) | Clears 2pp floor | In paper |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Heart Disease (1) | LR | FedAvg | 4 | 100→1.0 | 2.53 | 1.49 | 1.04 | −0.22 | 108.6% | Yes | No |
| Heart Disease (1) | LR | FedAvg | 4 | 100→0.5 | 3.43 | 3.29 | 0.13 | −0.15 | 104.5% | Yes | No |
| Heart Disease (1) | LR | FedAvg | 4 | 100→0.1 | 4.67 | 3.83 | 0.84 | −0.26 | 105.6% | Yes at raw value; **crosses below 2pp under the P-019/P-021 partition-size sweep at threshold=15** | Yes — reported in absolute terms (no % cited), per P-022 |
| Heart Disease (1) | LR | FedAvg | 4 | 100→natural | 0.10 | −0.54 | 0.65 | 0.38 | −274.6% | **No** | No |
| Heart Disease (1) | MLP | FedAvg | 4 | 100→1.0 | 2.64 | 1.43 | 1.21 | 0.08 | 97.1% | Yes | No |
| Heart Disease (1) | MLP | FedAvg | 4 | 100→0.5 | 4.77 | 3.01 | 1.76 | 0.11 | 97.7% | Yes | No |
| Heart Disease (1) | MLP | FedAvg | 4 | 100→0.1 | 18.46 | 5.00 | 13.46 | 4.62 | 75.0% | Yes | Yes |
| Heart Disease (1) | MLP | FedAvg | 4 | 100→natural | −0.86 | −0.04 | −0.81 | −0.03 | 96.2% | **No** (also: decline is negative — natural partition scores higher than α=100) | No |
| Heart Disease (1) | VQC | FedAvg | 4 | 100→1.0 | 2.66 | 3.29 | −0.64 | −0.18 | 106.9% | Yes | No |
| Heart Disease (1) | VQC | FedAvg | 4 | 100→0.5 | 5.44 | 5.21 | 0.23 | −0.24 | 104.4% | Yes | No |
| Heart Disease (1) | VQC | FedAvg | 4 | 100→0.1 | 7.25 | 8.49 | −1.24 | 0.35 | 95.2% | Yes | Yes |
| Heart Disease (1) | VQC | FedAvg | 4 | 100→natural | 0.74 | 1.08 | −0.34 | −0.60 | 181.1% | **No** | No |
| Diabetes 130 (2) | LR | FedAvg (weighted) | 4 | 100→1.0 | 0.80 | 0.65 | 0.16 | 0.14 (SE 0.07) | 82.5% | **No** | Yes — reported in absolute terms, per P-022 |
| Diabetes 130 (2) | LR | FedAvg (weighted) | 4 | 100→0.5 | 3.89 | 6.99 | −3.10 | 1.04 (SE 0.24) | 73.3% | Yes | Yes |
| Diabetes 130 (2) | LR | FedAvg (weighted) | 4 | 100→0.1 | 16.14 | 21.04 | −4.90 | 3.79 (SE 0.25) | 76.5% | Yes | Yes |
| Diabetes 130 (2) | MLP | FedAvg (weighted) | 4 | 100→1.0 | 0.99 | 0.87 | 0.12 | 0.22 (SE 0.08) | 77.8% | **No** | Yes — reported in absolute terms, per P-022 |
| Diabetes 130 (2) | MLP | FedAvg (weighted) | 4 | 100→0.5 | 3.25 | 3.92 | −0.67 | 0.44 (SE 0.12) | 86.5% | Yes | Yes |
| Diabetes 130 (2) | MLP | FedAvg (weighted) | 4 | 100→0.1 | 15.71 | 13.71 | 2.00 | 1.38 (SE 0.26) | 91.2% | Yes | Yes |
| Diabetes 130 (2) | LR | FedAvg (weighted) | 130 | 100→1.0 | 14.84 | 18.94 | −4.11 | 0.95 (SE 0.20) | 93.6% | Yes | Yes |
| Diabetes 130 (2) | LR | FedAvg (weighted) | 130 | 100→0.5 | 29.52 | 23.81 | 5.72 | 4.19 (SE 0.16) | 85.8% | Yes | Yes |
| Diabetes 130 (2) | LR | FedAvg (weighted) | 130 | 100→0.1 | — structurally infeasible, no valid Dirichlet(0.1) draw exists at K=130 under the 15-row minimum-client-size floor (verified directly, >1,000 draws attempted) — | | | | | N/A | Yes — table row present, marked infeasible |
| Diabetes 130 (2) | MLP | FedAvg (weighted) | 130 | 100→1.0 | 12.23 | 13.89 | −1.67 | 0.26 (SE 0.12) | 97.9% | Yes | Yes |
| Diabetes 130 (2) | MLP | FedAvg (weighted) | 130 | 100→0.5 | 26.89 | 20.64 | 6.26 | 1.21 (SE 0.26) | 95.5% | Yes | Yes |
| Diabetes 130 (2) | MLP | FedAvg (weighted) | 130 | 100→0.1 | — structurally infeasible, same reason — | | | | | N/A | Yes — table row present, marked infeasible |

**Implied-share values above use the single pooled point estimate** (not the
pooled→worst-group range reported for dataset 1's headline rows in the
paper); the paper's dataset-1 headline rows (100→0.1) additionally report a
range using the worst-group statistic (e.g. LR: 101–106%, computed from
−0.26 to −0.06pp) — see `docs/shared_test_validation.md` and
`docs/decisions.md` P-006/P-007/P-008 for that range's derivation.

### 1.2 Non-decomposition experiments (FedProx, Arm 5) — do not fit the schema above

| Arm | What | K | Config swept | Result location |
|---|---|---|---|---|
| Arm 3 | MLP + FedProx | 4 | μ ∈ {0.01, 0.05, 0.1} | `docs/arm3_report.md`, `results/runs_arm3.csv` (751 rows), `results/arm3_diagnostic_results.csv`, `results/arm3_diagnostic_divergence.csv` |
| Arm 5 | VQC + circular-mean aggregation | 4 | α ∈ {100, 1.0, 0.5, 0.1, natural} | `docs/arm5_report.md`, `docs/arm5_angle_verification.md`, `results/runs_arm5.csv` (251 rows), `results/arm5_diagnostic_results.csv`, `results/arm5_diagnostic_divergence.csv` |
| — | Weak/capacity-controlled MLP sweep | 4 | parameter-count bracketing | `docs/arm4_capacity_control_report.md`, `results/weak_mlp_diagnostic_results.csv`, `results/weak_mlp_diagnostic_divergence.csv` |
| — | Timing spike (E=1 vs E=5, VQC) | — | epochs=1 vs 5 | `results/runs.csv` (first 2 rows: `timing_spike_E1`, `timing_spike_E5`) |

### 1.3 Superseded/secondary variants of dataset-1 numbers (exist, not the paper's primary source)

| Variant | What it is | Location | Used in paper? |
|---|---|---|---|
| Unweighted, plain accuracy | The original D-047/D-049 numbers — this is what §V-A's table uses | `docs/decisions.md` D-047/D-049; reproduced in `results/composition_decomposition_summary.csv` | Yes — primary |
| Unweighted, balanced accuracy | LR/MLP re-evaluated with balanced accuracy, training unchanged (P-011) | `results/dataset1_reeval_balanced.json` | No |
| Weighted, balanced accuracy | LR/MLP re-evaluated with class-weighted training + balanced accuracy (P-013), for comparability with dataset 2's protocol | `results/dataset1_reeval_weighted.json` | No |
| Dataset 2, unweighted (original) | Both models collapse to a constant classifier; composition share saturates at 100% everywhere (P-011 finding) | `results/dataset2_decomposition.json` | No — superseded by the weighted run (P-013/P-014), which is what appears in the paper |

### 1.4 Alpha-calibration results (not decomposition, but a separate quantitative experiment)

| What | Result | Location | In paper |
|---|---|---|---|
| A-003: TV distance, mean only, 4-point grid, 10 seeds | α ≈ 1.5 (95% CI 1.0–4.7) | `scripts/alpha_calibration.py`, `docs/decisions.md` A-003 | Superseded — Abstract still cites this number (see §7) |
| P-016: TV + JS, mean + max, 25-point grid, 30 seeds | α ∈ [1.0, 1.7] depending on statistic (four point estimates: 0.991–1.727) | `scripts/alpha_calibration_fine.py`, `results/alpha_calibration_fine.json` | Yes — §V-D |

### 1.5 Minimum-partition-size robustness sweep (P-019/P-021/P-022)

Seven configurations × five thresholds (0, 5, 10, 15, 20 samples) = 35
re-analysed data points, covering both the raw decomposition share and the
shared-test-implied share. Full numbers: `results/partition_size_robustness.json`.
Summarized in `paper/paper_draft_v2.md` §V-G (two tables: observed decline by
threshold, and decomposition-share-vs-shared-test-implied-share by threshold).

---

## 2. Headline numbers currently in the paper (verbatim, with support)

### Abstract

| Claim | Exact figure | Supported in |
|---|---|---|
| Max estimator divergence | "as much as 8.7 percentage points" | §V.F table, LR K=4 100→0.1 interaction column (−8.69 to −8.43pp) |
| Client-count composition-share growth | "87–91% at 4 clients to 96–98% at 130" (MLP, α=0.5) | §V.F prose (P-022 revision) |
| Natural-partition α-calibration | "α ≈ 1.5 (95% CI: 1.0–4.7)" | Cited as supported by §V-D, but **§V-D itself was revised by P-016 to α ∈ [1.0, 1.7] — this exact figure no longer appears in §V-D** (see §7 below) |
| Contrast to informal comparison | "α ≈ 0.5–1.0" | Cited as "used elsewhere in this project," attributed to Flag 1 / D-037 |
| VQC training-time multiplier | "approximately 13,300×" | §V-C |

### §I Contributions

| Item | Exact figure | Supported in |
|---|---|---|
| Item 2 | "3 of 5 measured configurations" (LR disagreement count, dataset 2) | §V.F prose |
| Item 3 | "4 vs. 130 simulated clients," "α=0.5," "87–91% to 96–98%" | §V.F prose (P-022 revision) |
| Closing paragraph | D-048/D-049 (VQC composition-share prediction), P-009/P-014 (client-count prediction) — decision-ID citations, not standalone figures | `docs/decisions.md` |

### §VIII Conclusion

| Claim | Exact figure | Supported in |
|---|---|---|
| Dataset-1 MLP interaction | "roughly 8.5 percentage points" | §V-A (8.47–8.85pp range) |
| Dataset-2 LR disagreement count | "3 of 5 measured configurations" | §V.F prose |
| Max divergence | "up to 8.7 percentage points" | §V.F table (same source as Abstract) |
| Client-count composition-share growth | "87–91% at 4 clients to 96–98% at 130" | §V.F prose (P-022 revision) |
| VQC cost | "four-order-of-magnitude cost" (qualitative restatement of the 13,300× figure) | §V-C |

**Note:** the Abstract, §I, and §VIII no longer cite a specific percentage
for logistic regression's dataset-1 composition share (removed by P-022 —
LR is now reported there in absolute terms: "no measurable training effect;
observed decline small [4.66pp] and almost entirely composition [≈4.7pp]").

---

## 3. Extremes

**Across all 17 configurations that clear the 2pp floor** (Table 1.1, all
dataset-1 non-natural rows + all dataset-2 rows except the two K=4/α=100→1.0
rows):

| Statistic | Maximum | Minimum |
|---|---|---|
| Raw decomposition share (composition-only decline / observed decline) | **179.6%** — Diabetes 130, LR, K=4, 100→0.5 | **27.1%** — Heart Disease, MLP, 100→0.1 |
| Shared-test-implied share (this study's more trustworthy statistic) | **108.6%** — Heart Disease, LR, 100→1.0 | **73.3%** — Diabetes 130, LR, K=4, 100→0.5 |

**Including the four sub-2pp configurations** (which the paper does not
report as percentages): dataset-1 LR's natural-partition implied share
(−274.6%) and dataset-1 VQC's natural-partition implied share (181.1%) are
the numerical extremes in the full 24-row table, both driven by declines
under 1pp.

**Client-count comparison (K=4 vs. K=130), at every α tested at both counts:**

| Model | α | Shared-test-implied share, K=4 | Shared-test-implied share, K=130 | Change |
|---|---|---|---|---|
| LR | 1.0 | 82.5% (does not clear 2pp floor — reportable only as absolute: 0.80pp observed, 0.66pp composition) | 93.6% | not directly comparable as a percentage pair, per §III-C |
| LR | 0.5 | 73.3% | 85.8% | +12.5 points |
| MLP | 1.0 | 77.8% (does not clear 2pp floor — reportable only as absolute: 0.99pp observed, 0.77pp composition) | 97.9% | not directly comparable as a percentage pair, per §III-C |
| MLP | 0.5 | 86.5% | 95.5% | +9.0 points |

α=0.1 does not exist at K=130 (structurally infeasible — see Table 1.1),
so no comparison is possible at that α for either model.

---

## 4. Methods implemented in the codebase

### 4.1 Aggregation rules

| Rule | File | Used in |
|---|---|---|
| FedAvg (size-weighted mean) | `scripts/aggregators.py::fedavg` | Arms 1, 2, 3 (proximal term is client-side, aggregation is still FedAvg), 4; both dataset-1 and dataset-2 decomposition runs |
| Circular mean (angle-aware) | `scripts/aggregators.py::circular_mean` | Arm 5 only |

### 4.2 Model families / classes

| Class | File | Interface | Notes |
|---|---|---|---|
| `LogisticRegressionModel` | `scripts/models.py` | get_params/set_params/fit/predict_proba | Convex reference; unweighted training |
| `MLPModel` | `scripts/models_mlp.py` | same | 2 hidden units, 17 parameters |
| `FedProxMLPModel(MLPModel)` | `scripts/models_mlp.py` | same | Adds proximal term in `fit()`; used only in Arm 3 |
| `VQCModel` | `scripts/models_vqc.py` | same | 6 qubits, 3 layers, angle encoding, 18 parameters, PennyLane `lightning.qubit`, adjoint differentiation |
| `WeightedLogisticRegressionModel` | `scripts/models_weighted.py` | same | Inverse-frequency sample weighting inside `fit()`; used for dataset 2 and the P-013 dataset-1 comparison run — does **not** replace `LogisticRegressionModel`, exists alongside it |
| `WeightedMLPModel` | `scripts/models_weighted.py` | same | Same weighting scheme, MLP architecture |

### 4.3 Evaluation protocols and metrics

| Metric/protocol | Function | File(s) |
|---|---|---|
| Plain accuracy | `sklearn.metrics.accuracy_score`, direct calls | `composition_decomposition.py`, `cv_protocol.py`-based scripts (dataset-1 primary path) |
| Balanced accuracy (with single-class fallback to plain accuracy) | `_bal_acc` | `dataset2_decomposition.py`; imported into `partition_size_robustness.py` |
| F1 / AUROC | `sklearn.metrics.f1_score` / `roc_auc_score` | recorded per-client in `results/diagnostic_results.csv` and arm-specific CSVs, alongside accuracy |
| Worst-client accuracy (min over real Dirichlet/natural clients) | `worst_client_acc` / `worst_client_both` / inline `min(accs)` | Reimplemented separately in `dataset2_decomposition.py`, `dataset1_reeval_balanced.py`, `composition_decomposition.py`, `plots.py::worst_client_accuracy`, `worst_client.py` |
| Shared-test pooled accuracy (α-invariant held-out set) | `pooled_acc` / inline global-row computation | `dataset2_decomposition.py` (reusable function); dataset-1's equivalent is computed by grouping the existing `client=='global'` rows in `diagnostic_results.csv`/`arm4_diagnostic_results.csv` — **not a standalone function**, done inline/ad hoc per script (`dataset1_reeval_balanced.py`, this inventory's own §1.1 computation) |
| Shared-test worst-group accuracy (fixed, α-independent partition, minimum over groups) | `worst_group_acc` / `worst_group_both` / `worst_group_accuracy` | Implemented **three separate times** with near-identical logic: `dataset2_decomposition.py`, `dataset1_reeval_balanced.py`, `shared_test_worst_group.py` |
| Minimum-partition-size filtering | `filtered_min` | `partition_size_robustness.py` only |

**Direct answer to the specific question asked:** neither the shared-test
pooled evaluation nor the fixed-partition worst-group evaluation exists as a
single reusable, imported-everywhere code path. Both are reimplemented
per-script (pooled: inline in at least 2 places; worst-group: as a
near-duplicate function in 3 files) rather than factored into one shared
module. `dataset2_decomposition.py`'s versions (`pooled_acc`, `worst_group_acc`)
are the only ones imported by another script (`partition_size_robustness.py`,
and only for `_bal_acc`, not the group functions themselves).

### 4.4 Preprocessing / partitioning

| Component | File |
|---|---|
| Dataset 1 loading, feature selection, CV folds | `data_loader.py`, `cv_protocol.py`, `preprocessing.py` |
| Dataset 2 loading, feature selection (≥85% + near-zero-variance), CV folds | `dataset2_preprocessing.py`, `dataset2_cv_protocol.py` |
| Dirichlet partitioning (arbitrary K, reject-and-redraw floor) | `partitioner.py` |
| Natural (real-site) partitioning | part of `cv_protocol.py`/`data_loader.py`, dataset 1 only — no equivalent exists for dataset 2 (no per-record hospital ID) |

---

## 5. Unreported material (exists, not cited in the paper)

### 5.1 Figures

17 PNGs exist in `results/figs/`. **Exactly 1 is referenced in `paper/paper_draft_v2.md`** (`alpha_calibration_fine.png`, §V-D). The other 16:

| File | Apparent content (from filename/generating script) |
|---|---|
| `arm4_sanity_loss_curve.png` | VQC training loss sanity check |
| `client_count_composition_share.png` | Client-count vs. composition-share plot (`scripts/plot_client_count.py`) |
| `client_divergence.png` | Client parameter divergence |
| `client_divergence_vs_alpha.png` | Same, vs. α |
| `composition_decomposition.png` | Composition-decomposition bar chart (dataset 1) |
| `deck_architecture_loop.png` | System architecture diagram (A-006, for a presentation deck) |
| `deck_diagnostic_pair.png` | Diagnostic-pair diagram (for a presentation deck) |
| `feature_retention_curve.png` | Feature retention vs. missingness threshold |
| `feature_threshold_curve.png` | Related feature-threshold plot |
| `global_accuracy_vs_alpha.png` | Global accuracy vs. α |
| `partition_alpha_0.1.png`, `_0.5.png`, `_1.0.png`, `_100.png` | Per-client partition-size visualizations at each α |
| `partition_natural_vs_dirichlet.png` | Natural vs. Dirichlet partition comparison |
| `worst_client_accuracy.png` | Worst-client accuracy summary |
| `worst_client_accuracy_vs_alpha.png` | Same, vs. α |

### 5.2 Data/analyses

- **Dataset 1's natural-partition and intermediate-α (100→1.0, 100→0.5) composition-decomposition results** — the underlying per-condition accuracy data has existed in committed CSVs since the original composition-decomposition run (D-047/D-049), but the decline/residual/share arithmetic on those specific pairs had not been computed or written up before this inventory (§1.1 above, rows marked "In paper: No" for dataset 1).
- **§1.3's three secondary/superseded variants** of dataset-1's LR/MLP numbers (unweighted+balanced-accuracy, weighted+balanced-accuracy) — computed for cross-checking (P-011, P-013) but not cited in the paper text.
- **The original (unweighted) dataset-2 decomposition run** (`results/dataset2_decomposition.json`) — the vacuous, both-models-are-constant-classifiers result that motivated the class-weighting fix (P-011/P-013) — not cited in the paper; only its methodological consequence (the weighted protocol) is described (§IV.A).
- **Arm 3 (FedProx) and Arm 5 (circular-mean) full results** are described narratively in §V-B and §V-C of the paper (aggregate numbers), but their full per-replicate divergence data (`results/arm3_diagnostic_divergence.csv`, `results/arm5_diagnostic_divergence.csv`) and the weak/capacity-controlled MLP sweep (`docs/arm4_capacity_control_report.md`) are not otherwise referenced.
- **`docs/reference/fl_evaluation_protocol_literature.md`** and **`docs/reference/fl_fairness_literature.md`** — literature compilations; the fairness one is cited once in `docs/README.md` as feeding `paper/02_related_work.md`, but neither is cited from within `paper/paper_draft_v2.md` itself (only from the separate `paper/02_related_work.md` scaffold file — see §6).

---

## 6. Document inventory

### `docs/`

| File | Description | State |
|---|---|---|
| `decisions.md` | Append-only decision log, D-001 through A-006/P-022 | Living, actively maintained, current as of this session |
| `decisions_index.md` | Navigation table for the above | Living, actively maintained, current |
| `labbook.md` | Dated session log | Living, actively maintained, current |
| `README.md` | Documentation index and conventions | **Stale** — its file table lists only `decisions.md`, `decisions_index.md`, `labbook.md`, `INTERFACE.md`, `../RUNNING.md`, `reference/`, and one reference sub-file; does not list `arm3_report.md`, `arm4_report.md`, `arm4_capacity_control_report.md`, `arm5_report.md`, `arm5_angle_verification.md`, `diagnostic_report.md`, `shared_test_validation.md`, `dataset2_characterization.md`, `dataset2_decomposition.md`, or `reference/fl_evaluation_protocol_literature.md` |
| `INTERFACE.md` | Frozen model/aggregator interface contract | Complete, unchanged since 2026-08-18 freeze |
| `arm3_report.md` | Arm 3 (FedProx) results writeup | Complete |
| `arm4_report.md` | Arm 4 (VQC+FedAvg) results writeup | Complete |
| `arm4_capacity_control_report.md` | Weak/capacity-controlled MLP sweep writeup | Complete |
| `arm5_report.md` | Arm 5 (circular-mean) results writeup | Complete |
| `arm5_angle_verification.md` | VQC angle-wraparound verification | Complete |
| `diagnostic_report.md` | Original diagnostic-session report | Complete, historical |
| `dataset2_characterization.md` | Dataset 2 characterization (P-009) | Complete |
| `dataset2_decomposition.md` | Dataset 2 decomposition results writeup (P-014) | Complete |
| `shared_test_validation.md` | Shared-test validation writeup (P-006/P-007/P-008) | Complete |
| `circuit_diagram.png` / `.txt` | VQC circuit diagram | Complete |
| `reference/README.md` | Index for `reference/` | Complete |
| `reference/fl_evaluation_protocol_literature.md` | Literature compilation (Ayuvi, 2026-08-22) | Complete |
| `reference/fl_fairness_literature.md` | Literature compilation (Ayuvi, 2026-08-20) | Complete |
| `project_inventory.md` | This file | New, this session |

### `paper/`

| File | Description | State |
|---|---|---|
| `paper_draft_v2.md` | The canonical, full paper draft (moved from a shared Downloads copy, P-020) | Living, actively maintained, current — 392 lines |
| `01_introduction.md` | Scaffold stub (heading + HTML comment only) | **Stale/unused** — no content, superseded in practice by `paper_draft_v2.md`'s §I |
| `02_related_work.md` | Separate related-work document, 15,157 bytes of real content | Complete as its own document, but **not reconciled** with `paper_draft_v2.md`'s §II, which has different (though related) content |
| `03_methodology.md` | Scaffold stub | **Stale/unused** |
| `04_experiments.md` | Scaffold stub | **Stale/unused** |
| `05_results.md` | Scaffold stub | **Stale/unused** |
| `06_limitations.md` | Separate limitations document, actively maintained in parallel with `paper_draft_v2.md`'s §VII | Complete as its own document, overlapping but not identical content to §VII |
| `07_conclusion.md` | Scaffold stub | **Stale/unused** |

**Two parallel, partially-overlapping representations of the paper currently
exist in this repository**: the single-file `paper_draft_v2.md`, and the
`01`–`07` scaffold (of which only `02_related_work.md` and `06_limitations.md`
carry real content). This was noted as unresolved in P-020.

---

## 7. Open items

### Placeholders (`[[ ]]` markers), from the P-018 audit, current status

| Flag | Status |
|---|---|
| Flag 1 (α-calibration number vs. D-037's informal comparison) | Partially resolved — P-016 superseded the number itself; the reconciliation with D-037's separate "~0.5–1.0" claim remains open |
| Flag 2 ("six predictions" claim) | Resolved (P-018) |
| Flag 3 (A2G-QFL description incomplete) | **Open — explicitly left for Ayuvi, who owns the literature sections** |
| Flag 4 (citation [13] mismatch) | Resolved (P-018) |
| Flag 5 (VQC row numerical error) | Resolved (P-018) |
| Flag 6 (feature-exclusion sensitivity analysis) | Resolved — cut from scope, not merely caveated (P-018) |
| Flag 7 (§II-B "two of three" language) | Resolved (A-004) |
| Flag 8 (§IV.A/§VII dataset-2 companion content) | Resolved (A-004) |

### Newly observed in compiling this inventory, not previously logged

- **Abstract still cites the pre-P-016 α-calibration figure** — "α ≈ 1.5 (95% CI: 1.0–4.7)" — while §V-D itself was revised by P-016 to report α ∈ [1.0, 1.7] as a range across four statistics. The Abstract's specific figure no longer matches its own cited support section.
- **`docs/README.md`'s file table is stale**, missing 10 of the 17 files currently in `docs/`.
- **Two parallel paper representations** (`paper/paper_draft_v2.md` vs. the `01`–`07` scaffold) exist, flagged in P-020 as unreconciled.
- **This project's own stated evidentiary rule** ("if a number can't be traced to a row in `results/runs.csv`, it doesn't go in the paper" — `docs/README.md`) is not literally satisfied by most of the paper's current headline numbers: `results/runs.csv` has 31 rows (mostly early timing spikes); the composition-decomposition, dataset-2, alpha-calibration, and robustness-check numbers actually cited in the paper trace instead to separate files (`results/composition_decomposition_summary.csv`, `results/dataset2_decomposition_weighted.json`, `results/alpha_calibration_fine.json`, `results/partition_size_robustness.json`, and others named throughout this inventory) — each individually traceable, but not through the single file the rule names.
- **16 of 17 generated figures are not referenced in the paper draft** (§5.1).
