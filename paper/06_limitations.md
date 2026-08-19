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
