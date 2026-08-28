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
