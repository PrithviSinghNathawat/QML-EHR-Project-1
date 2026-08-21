# Arm 4 Report: VQC + FedAvg

2026-08-19/20, revised 2026-08-20 after the composition-vs-training decomposition
(D-044 onward). **This revision changes the headline number.** The original version
of this report led with observed worst-client decline (7.25pp) and read it as a
genuine, if intermediate, training-heterogeneity effect. That reading is superseded
(D-051): most of that decline is evaluation-slice composition, not training
heterogeneity, and the residual training effect is not distinguishable from zero.
The decomposed residual, not the observed decline, is now the headline number.

Full 5-fold CV x 10 seeds x 5 conditions sweep (250 replicates), identical protocol
to the classical diagnostic. Primary metric is worst-client accuracy (D-035); global
accuracy is secondary.

**VQC trained properly.** Task 2 sanity check (2 clients, alpha=100, 15 rounds):
loss decreased monotonically 1.096 -> 0.648 (~41%), still trending down at round 15,
not plateaued -- gradients are flowing, not a barren plateau. Figure:
`results/figs/arm4_sanity_loss_curve.png`. Separately, E=5 was confirmed to produce
a genuine training effect at the parameter level (max elementwise difference 0.91
between alpha=100 and alpha=0.1 trained models, D-046) -- the numbers below describe
real, alpha-dependent training, not a degenerate or untrained model.

---

## Headline: decomposed worst-client decline (alpha=100 -> alpha=0.1)

| | LR (convex reference) | MLP (matched comparator, 17p) | VQC (18p) |
|---|---|---|---|
| Observed decline | 4.66pp | 18.46pp | 7.25pp |
| Composition-only decline | 3.82pp | 4.99pp | **8.50pp** |
| % of observed movement that is composition | 82.0% | 27.0% | **117.2%** |
| **Residual (training effect)** | **+0.84pp** | **+13.47pp** | **-1.25pp** |

**The VQC shows no measurable training-heterogeneity effect once evaluation
composition is accounted for.** Composition alone (a fixed alpha=100-trained model,
scored against increasingly skewed per-client test slices) explains *more* than the
entire observed decline. The residual is slightly negative and, per a paired
per-replicate check (n=50, matched by seed/fold), not statistically distinguishable
from zero (mean -0.0124, SE 0.0107). MLP's residual (+13.47pp) is real by the same
test. LR's is small but positive (+0.84pp). The VQC's residual is the smallest of
the three, smaller than LR's -- not "intermediate between LR and MLP" as the
observed numbers alone suggested.

Full decomposition method and the LR/MLP-side derivation: `docs/decisions.md`
D-044-D-052 and the composition-only figures in `docs/diagnostic_report.md`.

---

## What this does and doesn't mean for the convexity account

D-036 (originally proposed alongside D-034's now-superseded framing) offered
convexity as the mechanism separating LR's small decline from MLP's large one: convex
objectives let FedAvg approximate the pooled optimum roughly independent of
partitioning, confining damage to individual clients; non-convex objectives let
per-client models drift further, and that drift propagates to the aggregated model.

**The VQC is non-convex** (a parameterized quantum circuit's loss landscape is not
convex in its rotation angles), yet its residual training effect (-1.25pp,
indistinguishable from zero) behaves like the convex LR's small residual (+0.84pp),
not like the non-convex MLP's large one (+13.47pp). Taken at face value, a
non-convex model showing convex-like behavior on this specific axis complicates a
purely convexity-based account of what separates LR from MLP.

**This is flagged as an open question, not resolved here.** Plausible directions
that have not been checked (none should be read as the answer without further
verification): the VQC's small parameter count (18) relative to MLP's decision
surface flexibility; the specific structure of a shallow, linearly-entangled ansatz
constraining how far local training can actually move the parameters regardless of
loss landscape convexity; or the small per-client data volumes at this problem size
simply not being enough to drive any model far from its initialization within 20
rounds, independent of convexity. Distinguishing between these would need further,
separate work. Stated as an open tension, not speculated into an answer.

---

## Global accuracy (secondary)

| condition | LR | MLP | VQC |
|---|---|---|---|
| 100 | 0.7625 | 0.7685 | 0.7142 |
| 1.0 | 0.7647 | 0.7677 | 0.7161 |
| 0.5 | 0.7640 | 0.7674 | 0.7166 |
| 0.1 | 0.7651 | 0.7223 | 0.7108 |
| natural | 0.7587 | 0.7688 | 0.7202 |

VQC's global accuracy is consistently 4-6pp lower than both classical models at every
condition, and flat across the sweep (0.7108-0.7202, a 0.94pp range) -- consistent
with the residual-training-effect finding above: the trained model barely changes
with alpha.

## Worst-client accuracy, observed (raw, pre-decomposition)

| condition | LR | MLP | VQC |
|---|---|---|---|
| 100 | 0.6942 | 0.6991 | 0.6488 |
| 1.0 | 0.6690 | 0.6727 | 0.6222 |
| 0.5 | 0.6600 | 0.6514 | 0.5944 |
| 0.1 | 0.6476 | 0.5145 | 0.5763 |
| natural | 0.6932 | 0.7077 | 0.6414 |

Kept for reference and reproducibility (`scripts/worst_client.py` regenerates this
table exactly from `results/arm4_diagnostic_results.csv`) -- but read alongside the
decomposition above, not on its own.

## Client parameter divergence

| condition | LR | MLP | VQC |
|---|---|---|---|
| 100 | 0.0626 | 0.2684 | 0.0491 |
| 1.0 | 0.2805 | 0.5773 | 0.1609 |
| 0.5 | 0.3717 | 0.7529 | 0.2141 |
| 0.1 | 0.4945 | 1.1088 | 0.2875 |
| natural | 0.2447 | 0.4879 | 0.1471 |

Monotonically increasing as alpha falls for all three models -- client-level drift
is real and present for the VQC too (confirmed directly against trained parameters,
D-046). **This is not in tension with the residual-training-effect finding above:**
individual clients do diverge from each other during local training, but the
*aggregated* model's sensitivity to which alpha it was trained under is what the
composition decomposition measures, and those are different quantities. Cross-model
magnitude comparisons here carry the same caution as before: VQC parameters are
angles in a different, bounded space from classical weights, so raw L2 magnitude is
not directly comparable across model families -- the monotonic trend is the
meaningful signal.

## Wall-clock

| | value |
|---|---|
| Arm 4 total compute (sum of all replicate times) | 129,591s = 36.0 CPU-hours |
| Arm 4 actual wall-clock (4-way parallel) | 32,299s = 8.97 hours |
| Arm 4 per-replicate mean | 518.4s (std 66.6s, range 389.2-838.0s) |
| MLP per-run (same protocol, timed directly for this comparison) | 0.0389s mean (5 trials) |
| LR per-run (from `results/runs.csv`, Arm 2) | 0.0244s mean |
| **Arm 4 / MLP wall-clock ratio** | **~13,318x** |
| Arm 4 / LR wall-clock ratio | ~21,244x |

This number is unaffected by the decomposition revision -- wall-clock cost is a fact
about the simulator and hardware, not about what the training curve means.

---

## Answering the three questions, revised

**1. Did the VQC train?** Yes -- confirmed by the sanity check loss curve, and
separately by the E=5-vs-E=1 parameter-difference check (D-046), before any accuracy
number was trusted.

**2. Does the quantum worst-client curve fall faster than the MLP's?** No -- and
more precisely now than the original answer: once composition is subtracted out,
the VQC shows **no residual training-heterogeneity effect at all** (-1.25pp, not
distinguishable from zero), while the MLP's is large and real (+13.47pp). The
original "VQC sits between LR and MLP" framing was true of the observed numbers but
is now understood to reflect the VQC's lower baseline accuracy interacting with
identical test-slice composition, not a genuine intermediate degree of
heterogeneity sensitivity.

**3. What is the wall-clock ratio?** Unchanged: ~13,318x versus the matched MLP,
~21,244x versus logistic regression, on this simulator and this hardware.
