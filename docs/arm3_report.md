# Arm 3 Report: FedProx (MLP only)

2026-08-22. The last arm built and the last compute run for this project. Full
protocol match to every other arm: 20 federated rounds, E=5 local epochs, 5-fold
stratified CV, 10 seeds, identical partitions and seeds. μ ∈ {0.01, 0.05, 0.1}.
750 replicates (3μ × 5 conditions × 10 seeds × 5 folds), classical (MLP), ~75s
total, no parallelism needed.

**Scope: MLP only.** FedProx exists to restrain client drift via a proximal term.
The composition-vs-training decomposition (D-050, D-051/D-049) showed LR's residual
training effect is small (+0.84pp) and the VQC's is negative/indistinguishable from
zero (-1.25pp) — neither has meaningful genuine drift damage for a proximal term to
recover. Only MLP (+13.47pp residual, D-047) has real damage to act on. Building
FedProx for LR or VQC would be measuring a null effect on top of a null effect.

**Interface verification, done before writing any code:** `scripts/federated_loop.py`
lines 52-53 call `set_params(global_params.copy())` immediately before `fit()`, every
round, every client, no code between the two calls. This means `fit()`'s own current
parameters, snapshotted as its first action, are exactly that round's true global
vector — no staleness, no interface change needed. The proximal term lives entirely
in `FedProxMLPModel.fit()` (`scripts/models_mlp.py`); `federated_loop.py` and
`scripts/aggregators.py` were not touched. Correctness checked directly: μ=0
reproduces `MLPModel`'s trained parameters bit-for-bit through the unmodified loop.

---

## Headline: does FedProx recover the 13.47pp of genuine training damage?

**Mostly no — it recovers a modest, non-monotonic fraction of it, not the damage
itself.**

| μ | observed decline (α=100→0.1) | composition-only decline | % composition | **residual (training effect)** |
|---|---|---|---|---|
| 0 (FedAvg, D-047 reference) | 18.46pp | 4.99pp | 27.0% | **+13.47pp** |
| 0.01 | 17.64pp | 4.92pp | 27.9% | **+12.72pp** |
| 0.05 | 15.91pp | 4.77pp | 30.0% | **+11.14pp** |
| 0.1 | 16.61pp | 3.81pp | 22.9% | **+12.80pp** |

FedProx reduces the residual training-heterogeneity effect by 0.67-2.33pp across the
three μ values tested (5.0-17.3% relative reduction), never eliminating it. The
reduction is **not monotonic in μ**: μ=0.05 (the literature-recommended value for
this dataset) shows the largest reduction; μ=0.01 and μ=0.1 show smaller, similar
reductions to each other. Not smoothed into a monotonic story — this is what the
sweep gives.

**Paired significance check** (same seed/fold pairs, FedProx minus FedAvg worst-client
accuracy at α=0.1):

| μ | paired improvement | SE | roughly how many SE from zero |
|---|---|---|---|
| 0.01 | +0.87pp | 1.53pp | 0.6 |
| 0.05 | +2.93pp | 1.91pp | 1.5 |
| 0.1 | +2.05pp | 1.77pp | 1.2 |

None reach conventional significance (~2 SE). μ=0.05 comes closest and is
directionally consistent with it being the literature-recommended value, but this is
not a strong result — it should be read as "suggestive, not established" given n=50
and this noise level.

---

## FedProx does what it's mechanistically designed to do — divergence drops

| μ | client divergence, final round, α=0.1 |
|---|---|
| 0 (FedAvg) | 1.1088 |
| 0.01 | 1.0763 |
| 0.05 | 1.0277 |
| 0.1 | 0.9578 |

**Monotonically decreasing in μ, cleanly.** The proximal term is restraining
per-client drift exactly as designed — this is not in question. The tension worth
stating plainly: **a clean, monotonic reduction in the mechanism FedProx targets
(client parameter divergence) does not translate into a correspondingly large or
monotonic recovery of the outcome that matters (worst-client accuracy residual).**
Divergence at μ=0.1 is 13.6% lower than at μ=0 (FedAvg), but μ=0.1's residual
training-heterogeneity effect (+12.80pp) is not meaningfully better than μ=0.01's
(+12.72pp) despite μ=0.1 restraining drift much more. Whatever connects "how much
clients drift apart" to "how much the worst client's accuracy suffers" is not a
simple, direct, monotonic relationship in this data.

---

## Full observed numbers (for reference; read alongside the decomposition above, not alone)

| condition | FedAvg (μ=0) | μ=0.01 | μ=0.05 | μ=0.1 |
|---|---|---|---|---|
| 100 | 0.6991 | 0.6996 | 0.7030 | 0.7011 |
| 1.0 | 0.6727 | 0.6683 | 0.6665 | 0.6741 |
| 0.5 | 0.6514 | 0.6535 | 0.6539 | 0.6620 |
| 0.1 | 0.5145 | 0.5232 | 0.5438 | 0.5350 |
| natural | 0.7077 | 0.7078 | 0.7055 | 0.7037 |

Regenerable exactly from committed code: `scripts/worst_client.py results/arm3_diagnostic_results.csv`.

---

## Answer

FedProx, swept honestly across three μ values including the literature-recommended
0.05, **does not recover most of the MLP's genuine training-heterogeneity damage**.
It produces a real, mechanistically-expected reduction in client divergence, and a
small, non-monotonic, not-quite-significant improvement in the worst-client residual
— on the order of 5-17% of the damage, not a restoration. μ was not tuned toward a
preferred outcome; all three values are reported as run.
