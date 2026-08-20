# Verifying the D-036 Circular-Mean Null Explanation

2026-08-20, follow-up to `docs/arm5_report.md`. D-036 attributed the
circular-mean-vs-FedAvg null result to trained rotation angles never
reaching the wraparound boundary (circular mean only diverges numerically
from a linear mean near ±π). This was a plausible but unverified
explanation at the time — the original 250-replicate sweeps saved only
evaluation metrics and a scalar divergence number, not the raw trained
parameter vectors. This report re-runs a representative sample with the
raw angles captured and checks the explanation directly.

## Method

Re-ran 20 replicates (5 conditions × 2 seeds {0, 5} × 1 fold, both Arm 4
and Arm 5) with the federated training loop reimplemented inline (not
calling `run_federated`) so every client's local parameter vector could be
saved every round, before aggregation — `scripts/capture_angles_worker.py`.
Not the full 250-replicate grid; a sample spanning the whole α range
(deliberately including α=0.1, where drift is largest and wraparound, if
it happens anywhere, would be most likely) and two different seeds.

## Result: the explanation is confirmed

**Across all 28,800 captured client parameter values** (20 replicates ×
20 rounds × 4 clients × 18 angles):

| | value |
|---|---|
| min | -1.1390 |
| max | 1.7927 |
| max \|θ\| | 1.7927 |
| mean | 0.0337 |
| std | 0.3274 |
| 99.9th percentile \|θ\| | 1.7431 |
| fraction with \|θ\| > π/2 | 0.42% |
| **fraction with \|θ\| > 0.9π** | **0.0000%** |

π ≈ 3.1416. The largest angle magnitude observed anywhere in the sample
is 1.79 — **1.35 radians short of π**, comfortably inside the range where
a circular mean and a linear mean agree.

**By condition** — checking specifically whether the most heterogeneous
setting pushes angles closer to the boundary (it doesn't):

| condition | max \|θ\| | gap to π |
|---|---|---|
| α=100 | 1.7927 | 1.3489 |
| α=1.0 | 1.7818 | 1.3598 |
| α=0.5 | 1.5681 | 1.5735 |
| α=0.1 | 1.5271 | 1.6145 |
| natural | 1.5954 | 1.5462 |

If anything, α=100 (least heterogeneous) has the largest angle magnitudes
in this sample, not α=0.1 — there's no trend toward the boundary as
heterogeneity increases. Global (post-aggregation) parameters show the
same pattern (min -1.1245, max 1.7835).

## Conclusion

**The original explanation holds.** Trained rotation angles in this
project's VQC stay well inside the wraparound-sensitive region under every
condition tested — driven by the small learning rate (0.1), narrow random
initialization (0.1×N(0,1)), and modest round count (20), not by anything
alpha-dependent. Circular-mean and FedAvg aggregation computing
near-identical results (D-036) is explained by this, not by a
coincidence or a masked bug. No alternative explanation is needed.
