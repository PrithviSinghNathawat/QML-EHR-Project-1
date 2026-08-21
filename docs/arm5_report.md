# Arm 5 Report: VQC + Circular-Mean Aggregation (D-007 Ablation)

2026-08-20, revised 2026-08-20 same day after the Arm 4 composition-vs-training
decomposition (D-044 onward). **Note on scope of this revision:** the composition
decomposition itself (D-044-D-052) was run against Arm 4 (FedAvg) only, not
independently re-run for Arm 5 (circular-mean) -- there was no 50-replicate
alpha=100-only composition sweep launched for the circular-mean aggregator. What
follows for Arm 5 is inferred **by direct extension** from Arm 4's decomposition,
justified by the headline finding below, and that inference is stated explicitly
rather than presented as independently measured.

Full 5-fold CV x 10 seeds x 5 conditions sweep (250 replicates), identical protocol
to Arm 4, identical partitions and seeds. The only difference from Arm 4:
aggregation uses `circular_mean` instead of `fedavg`. Citing A2G-QFL.

---

## Headline result: no meaningful difference from FedAvg, on any of observed / composition / residual

| condition | worst-client (FedAvg, observed) | worst-client (circular-mean, observed) |
|---|---|---|
| 100 | 0.6488 | 0.6488 |
| 1.0 | 0.6222 | 0.6222 |
| 0.5 | 0.5944 | 0.5947 |
| 0.1 | 0.5763 | 0.5782 |
| natural | 0.6414 | 0.6405 |

At alpha=100 and alpha=1.0, the two aggregators produce **identical** observed
results to 4 decimal places. At the other conditions, differences are in the 3rd
decimal place -- smaller than run-to-run seed noise (std ~0.07-0.12 on worst-client
accuracy). Client parameter divergence is likewise identical or near-identical at
every condition.

**Because Arm 5's observed worst-client numbers are statistically indistinguishable
from Arm 4's at every condition, the Arm 4 decomposition (D-049) is inferred to
apply to Arm 5 by direct extension:** composition-only decline should also exceed
Arm 5's observed decline, and the residual training effect should also be close to
zero or slightly negative, the same as Arm 4's. **This has not been independently
verified** -- it would require its own 50-replicate alpha=100-only composition sweep
using `circular_mean`, which was not run. Flagged as a gap, not asserted as
confirmed.

**If the inference holds, Arm 5's contribution to the aggregation-choice question is
now narrower than originally framed:** not just "circular-mean and FedAvg produce
the same worst-client degradation" (D-041's original finding), but "neither
aggregator's worst-client number reflects a real training-heterogeneity effect for
the VQC in the first place" -- the D-007 ablation's original question (does
circular-mean change worst-client degradation relative to FedAvg) presupposed there
was a training-heterogeneity-driven degradation to potentially change. On the Arm 4
evidence, there mostly isn't one to be sensitive to.

---

## Why the null result: verified directly (unaffected by this revision)

D-043 attributed the circular-mean-vs-FedAvg agreement to trained rotation angles
never reaching the wraparound boundary (circular mean only diverges numerically from
a linear mean near +/-pi). This was checked directly against real trained
parameters (`docs/arm5_angle_verification.md`): across 28,800 captured client angle
values, max |theta| = 1.7927, 1.35 radians short of pi, with no trend toward the
boundary as heterogeneity increases. This explanation is independent of the
composition-decomposition question above and is unaffected by this revision -- it
explains why the two *aggregators* agree, not what the agreed-upon number means.

## Wall-clock

| | Arm 4 (FedAvg) | Arm 5 (circular-mean) |
|---|---|---|
| Total compute | 36.0 CPU-hr | 29.12 CPU-hr |
| Wall-clock (4-way parallel) | 8.97 hr | 7.48 hr |
| Per-replicate mean | 518.4s | 419.4s |

Unaffected by this revision -- consistent with ordinary machine-load variance
between two separately-scheduled overnight runs, not attributed to the aggregator
choice.

## Revised answer to the ablation question

**Does circular aggregation change worst-client degradation relative to Arm 4?** No,
observably. **But per the Arm 4 decomposition, there may be little or no genuine
training-heterogeneity-driven degradation for either aggregator to change** -- most
of what both arms report as "worst-client degradation" is now understood (by
extension from Arm 4's directly-measured result) to be evaluation-slice composition.
The aggregator-choice question and the composition-vs-training question turned out
to be entangled in a way the original ablation design didn't anticipate. A
circular-mean-specific composition decomposition would be needed to confirm this
extension rather than infer it.
