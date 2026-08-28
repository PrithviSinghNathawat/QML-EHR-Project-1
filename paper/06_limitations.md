# 6. Limitations

<!-- Append every constraint as we hit it, don't wait until the end. -->

- **Reduced feature set.** `ca` (number of major vessels) and `thal` (thalassemia
  status) were dropped rather than imputed, because both are ~90-99% missing at every
  site except Cleveland. Both are clinically relevant to cardiac risk in the source
  literature. Models here train on 11 raw features (reduced to 6 via PCA), not the full
  13. See `docs/decisions.md`, 2026-08-18.
- **Simulator-only timing.** All wall-clock numbers describe `lightning.qubit`
  simulation on CPU, using `diff_method="adjoint"`, which is not available on real
  quantum hardware (adjoint differentiation needs the full statevector). Training-time
  comparisons in this paper do not predict real-hardware training cost.
- **No per-record hospital identifier in the second dataset.** Diabetes 130-US
  Hospitals is described by its source publication (Strack et al., 2014) as spanning
  "130 US hospitals and integrated delivery networks," but the public release
  (`data2/raw/diabetic_data.csv`, 50 columns) contains no hospital/facility ID field —
  confirmed by inspecting every column, not assumed from documentation. "130 hospitals"
  describes the underlying data warehouse, not an available per-encounter partition
  key. Any client structure for this dataset is necessarily synthetic (Dirichlet), not
  a real institutional partition, unlike the heart-disease dataset's 4 processed sites.
  See `docs/dataset2_characterization.md` and P-009.
- **The composition-decomposition method does not generalize as a
  standalone estimate.** On dataset 1, logistic regression agreed with an
  independent shared-test check and only the MLP diverged (P-008),
  suggesting the decomposition might be reliable except where a model has
  a genuine training effect. On dataset 2 (P-014), logistic regression
  itself disagrees with the shared-test estimate at 3 of 5 measured
  configurations — including cases where the decomposition's residual is
  negative while the shared-test estimate finds a real, many-standard-
  error-from-zero positive training effect. The interaction term between
  the two methods is a property of the specific (model, partition, α)
  configuration, not of model family alone, and can take either sign. The
  decomposition should always be reported alongside a shared-test check,
  not used standalone. See `docs/dataset2_decomposition.md`.
- **Dataset 2's classical models required class-weighted training to learn
  anything at all.** Under dataset 1's original unweighted training
  objective, both LR and MLP converge to a constant "always predict
  majority class" classifier at 8.8% positive prevalence (verified to
  2,000 epochs) — not an artifact of insufficient training. Balanced
  accuracy alone corrects the resulting evaluation-metric degeneracy only
  where an evaluation slice contains both classes; single-class slices
  (common at extreme skew) still fall back to plain accuracy. Reported
  dataset-2 results use inverse-frequency class weighting during training
  in addition to balanced accuracy at evaluation (P-011, P-013);
  dataset-1 numbers reported for comparison use the same weighting for
  consistency, alongside the original unweighted figures for traceability.
- **Per-client test partitions get very small at the tail, worse on
  dataset 2, worst at K=130.** Dataset 1's smallest observed test
  partition at α=0.1 is n=1 (P-011 and earlier). Dataset 2 is worse at
  both client counts tested (P-017): at K=4, α=0.1, the smallest test
  partition is n=2; at K=130 — the client count behind the Section V.F
  headline confound-scaling result — 1 of 6,500 (seed, fold, client)
  cells has **zero** test rows, at both α=1.0 and α=0.5. Zero-row cells
  are correctly excluded from the worst-client minimum rather than scored
  (verified against `worst_client_acc` directly), so the reported "worst
  client" in those replicates is a minimum over 129 clients, not 130. The
  composition-share-grows-with-K finding is real, measured on real
  partitions — but individual per-replicate worst-client numbers at this
  regime should be read as noisy at the tail, not as precise per-client
  measurements. See `results/dataset2_partition_sizes.json` and
  `scripts/dataset2_partition_sizes.py`.
- **Natural-partition α-equivalence is a range, not a point (P-016).**
  Fine-grid calibration (TV distance and Jensen-Shannon divergence, both
  client-weighted mean and cross-client max, 25-point α grid, 30 seeds
  per point) narrows the earlier A-003 estimate (α≈1.5, CI 1.0–4.7) but
  does not collapse it to one number: mean-based statistics center near
  α≈1.5–1.7, max-based statistics (driven by the single most-skewed
  client) center near α≈1.0. The natural partition maps onto Dirichlet
  α in the range 1.0–1.7, depending on which statistic and which client
  (typical or worst) is asked. Separately, this still does not reconcile
  with `docs/decisions.md` D-037's informal "~0.5–1.0" comparison, which
  measures a different quantity (downstream metric matching, not label-
  distribution distance) — that reconciliation remains open.
