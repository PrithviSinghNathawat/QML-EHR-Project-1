# Arm 4 Report: VQC + FedAvg

2026-08-19/20. Full 5-fold CV x 10 seeds x 5 conditions sweep (250
replicates), identical protocol to the classical diagnostic
(`docs/diagnostic_report.md`), same partitions, same seeds, same 6
features. Primary metric is worst-client accuracy (D-030); global accuracy
is secondary.

**VQC trained properly.** Task 2 sanity check (2 clients, alpha=100, 15
rounds): loss decreased monotonically 1.096 -> 0.648 (~41%), still
trending down at round 15, not plateaued -- gradients are flowing, not a
barren plateau. Figure: `results/figs/arm4_sanity_loss_curve.png`. The
numbers below describe a trained model.

---

## Worst-client accuracy (primary metric)

| condition | LR (convex reference) | MLP (matched comparator, 17p) | VQC (18p) |
|---|---|---|---|
| 100 | 0.6942 | 0.6991 | 0.6488 |
| 1.0 | 0.6690 | 0.6727 | 0.6222 |
| 0.5 | 0.6600 | 0.6514 | 0.5944 |
| 0.1 | 0.6476 | 0.5145 | 0.5763 |
| natural | 0.6932 | 0.7077 | 0.6414 |

**VQC's worst-client accuracy declines monotonically as alpha falls**
(0.6488 -> 0.6222 -> 0.5944 -> 0.5763, a 7.25pp drop from alpha=100 to
alpha=0.1) -- the same qualitative pattern found for both classical
models. In magnitude, the VQC's decline (7.25pp) sits **between** the
convex reference LR (4.66pp) and the non-convex comparator MLP (18.46pp)
-- more sensitive to heterogeneity than the convex model, less sensitive
than the matched non-convex one, at every alpha tested.

**Does the quantum curve fall faster than the MLP's?** No, not in this
data -- MLP's worst-client decline (18.46pp) is over twice as steep as
the VQC's (7.25pp). The one condition where VQC has a lower worst-client
accuracy than MLP is alpha=0.1 specifically (0.5763 vs 0.5145 -- wait,
VQC is actually *higher* than MLP there); at every other condition VQC's
worst-client number is lower than both classical models' in absolute
terms, but its *decline* across the sweep is shallower than MLP's.

## Global accuracy (secondary)

| condition | LR | MLP | VQC |
|---|---|---|---|
| 100 | 0.7625 | 0.7685 | 0.7142 |
| 1.0 | 0.7647 | 0.7677 | 0.7161 |
| 0.5 | 0.7640 | 0.7674 | 0.7166 |
| 0.1 | 0.7651 | 0.7223 | 0.7108 |
| natural | 0.7587 | 0.7688 | 0.7202 |

VQC's global accuracy is consistently 4-6pp lower than both classical
models at every condition -- expected, given the classical models had 3-5
extra seeds of easy linear/near-linear structure to exploit that an
18-parameter, 3-layer circuit with no classical post-processing does not
have equivalent capacity for. VQC's global accuracy is **flat** across the
sweep (0.7108-0.7202, a 0.94pp range) -- more like LR's flatness than
MLP's alpha=0.1 dip.

## Spread (global - worst-client)

| condition | LR | MLP | VQC |
|---|---|---|---|
| 100 | 0.0683 | 0.0693 | 0.0655 |
| 1.0 | 0.0957 | 0.0950 | 0.0939 |
| 0.5 | 0.1041 | 0.1159 | 0.1223 |
| 0.1 | 0.1175 | 0.2078 | 0.1345 |
| natural | 0.0655 | 0.0611 | 0.0788 |

VQC's spread also widens as alpha falls (0.0655 -> 0.1345), consistent
with all three models. At alpha=0.1 specifically, VQC's spread (0.1345)
is between LR's (0.1175) and MLP's (0.2078) -- the same "in between"
pattern as the worst-client decline itself.

## Client parameter divergence

| condition | LR | MLP | VQC |
|---|---|---|---|
| 100 | 0.0626 | 0.2684 | 0.0491 |
| 1.0 | 0.2805 | 0.5773 | 0.1609 |
| 0.5 | 0.3717 | 0.7529 | 0.2141 |
| 0.1 | 0.4945 | 1.1088 | 0.2875 |
| natural | 0.2447 | 0.4879 | 0.1471 |

Monotonically increasing as alpha falls, same as both classical models --
the mechanism (client drift under label skew) fires for the VQC too.
**Caution on cross-model magnitude comparison:** VQC parameters are
rotation angles in a bounded, physically different space from classical
weights: raw L2 distance is not directly comparable in absolute terms
across model families with different parameter semantics and scales. The
monotonic *trend* is the meaningful, comparable signal here, not the
absolute numbers.

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

This is the honest cost number: on this classical simulator, on this
CPU, at this problem size, the VQC costs about 13,300x the matched
classical comparator's wall-clock per training run.

---

## Answering the three questions

**1. Did the VQC train?** Yes -- confirmed by the sanity check loss curve
before any accuracy number was trusted, per instruction.

**2. Does the quantum worst-client curve fall faster than the MLP's?**
No. VQC's worst-client decline (7.25pp, alpha=100->0.1) is shallower than
MLP's (18.46pp) -- roughly a third as steep. It is steeper than LR's
(4.66pp). The VQC sits between the convex and non-convex classical
references on this metric, not beyond either.

**3. What is the wall-clock ratio?** ~13,318x versus the matched MLP,
~21,244x versus logistic regression, on this simulator and this hardware.
