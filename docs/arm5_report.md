# Arm 5 Report: VQC + Circular-Mean Aggregation (D-007 Ablation)

2026-08-20. Full 5-fold CV x 10 seeds x 5 conditions sweep (250
replicates), identical protocol to Arm 4, identical partitions and seeds.
The only difference from Arm 4: aggregation uses `circular_mean`
(`atan2(sum(w*sin(theta)), sum(w*cos(theta)))` per parameter) instead of
`fedavg` (linear weighted mean). Citing A2G-QFL.

## Headline result: no meaningful difference from FedAvg

| condition | worst-client (FedAvg) | worst-client (circular-mean) | global (FedAvg) | global (circular-mean) |
|---|---|---|---|---|
| 100 | 0.6488 | 0.6488 | 0.7142 | 0.7142 |
| 1.0 | 0.6222 | 0.6222 | 0.7161 | 0.7161 |
| 0.5 | 0.5944 | 0.5947 | 0.7166 | 0.7170 |
| 0.1 | 0.5763 | 0.5782 | 0.7108 | 0.7107 |
| natural | 0.6414 | 0.6405 | 0.7202 | 0.7200 |

At alpha=100 and alpha=1.0, the two aggregators produce **identical**
results to 4 decimal places. At the other conditions, differences are in
the 3rd decimal place -- smaller than run-to-run seed noise (std ~0.07-0.12
on worst-client accuracy, see `docs/arm4_report.md`). Client parameter
divergence (final round) is likewise identical or near-identical at every
condition (e.g. alpha=0.1: 0.2875 vs 0.2877).

**Interpretation:** circular-mean aggregation does not change worst-client
degradation relative to FedAvg in this data. The most plausible reason:
circular mean only diverges numerically from a linear mean when angles
approach the wraparound boundary (near 0/2*pi, or when averaging angles on
opposite sides of the circle). Given the sanity-check loss curve (Task 2,
Arm 4) shows the model training over a modest number of rounds with a
small learning rate (0.1) from a narrow random initialization
(0.1*N(0,1)), the trained parameters most likely never range widely enough
to reach that regime -- so the two aggregation schemes end up computing
essentially the same average. This is a **reportable null result**, not
an inconclusive one: 250 replicates, tight agreement across the whole
sweep, not a small-sample fluke.

## Wall-clock

| | Arm 4 (FedAvg) | Arm 5 (circular-mean) |
|---|---|---|
| Total compute | 36.0 CPU-hr | 29.12 CPU-hr |
| Wall-clock (4-way parallel) | 8.97 hr | 7.48 hr |
| Per-replicate mean | 518.4s | 419.4s |

Arm 5 ran somewhat faster in total, consistent with ordinary machine-load
variance between two separately-scheduled overnight runs (the aggregation
function itself is a trivial O(n_clients) computation in both cases and
is not the bottleneck -- VQC local training dominates wall-clock either
way). Not attributed to the aggregator choice.

## Answering the ablation question

**Does circular aggregation change worst-client degradation relative to
Arm 4?** No, not measurably, across the full sweep. The heterogeneity
penalty (worst-client decline as alpha falls) is present and of
essentially identical magnitude under both aggregators. On this evidence,
the choice between FedAvg and circular-mean aggregation is not what
determines how much the VQC suffers under label skew in this regime.
