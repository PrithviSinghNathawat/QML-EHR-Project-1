# Capacity Control for Arm 4: Weakened MLP

2026-08-20, follow-up to `docs/arm4_report.md`. Prithvi's concern: the VQC
shows less worst-client degradation than the matched MLP, but also starts
from a lower α=100 baseline (64.9% vs 69.9%) -- the apparent robustness
could be capacity (weaker learners are closer to constant predictors and
degrade less by construction) rather than model family. This report builds
a deliberately weakened MLP calibrated to the VQC's baseline and reports
what its degradation profile actually shows.

**Headline: the calibration did not fully hold on the full sweep, and the
resulting weakened model turned out to expose a training degeneracy that
makes its "degradation" number not directly comparable to the VQC's or
full MLP's. Both findings are reported in full below rather than
smoothed over.**

## Calibration (done once, against α=100 seed=0 only, before the full sweep)

Target: VQC's α=100 worst-client accuracy, 0.6488 (full 50-replicate mean).

Levers tried, in order (hidden units first per the instruction, then early
stopping since hidden units alone weren't enough):

| config | α=100 worst-client (seed=0 only) |
|---|---|
| hidden=2 (the existing matched MLP) | 0.6991 (full sweep) |
| hidden=1, rounds=20, local_epochs=5 | 0.6852 |
| hidden=1, rounds=20, local_epochs=1 | 0.6812 |
| hidden=1, rounds=10, local_epochs=5 | 0.6819 |
| hidden=1, rounds=5, local_epochs=1 | 0.6591 |
| **hidden=1, rounds=4, local_epochs=1** | **0.6483** (vs target 0.6488 -- 0.05pp gap) |

Locked this last config (9 parameters, down from 17) and ran the full
protocol (5-fold CV x 10 seeds x 5 conditions) without further adjustment.

## Calibration did not generalize from seed=0 to the full 10-seed sweep

| | α=100 worst-client |
|---|---|
| Seed=0 calibration estimate | 0.6483 |
| VQC target | 0.6488 |
| **Full 10-seed sweep, actual** | **0.5236** |

The full-sweep result undershoots the target by ~12.5pp, not the ~0.05pp
the single-seed calibration suggested. Seed=0 was not representative of
the 10-seed average for this particular (very lightly trained)
configuration -- worth remembering for any future single-seed calibration
in this project. **Not re-tuned after seeing this** -- reporting the
sweep as run, per instruction.

## A training degeneracy, discovered while checking the result

Worst-client accuracy for the weakened MLP declined sharply across the
sweep (0.5236 -> 0.1559, a 36.77pp drop, steeper than both the full MLP's
18.46pp and the VQC's 7.25pp). Before reporting that as "the weakened
model degrades more," it was checked directly -- and the check overturned
the natural reading.

**The trained global model is bit-identical across every alpha condition,
for a given seed and fold.** Verified directly: same seed/fold, alpha=100
vs alpha=0.1, the model's 9 trained parameters match to every printed
digit, and `predict_proba` on the first 10 test rows returns identical
floats. This holds across the whole sweep (global accuracy for `weak-MLP`
is exactly 0.6001 -- to 4 decimal places -- at every one of the 5
conditions, for every seed/fold).

**Why:** with `local_epochs=1` (a single full-batch gradient step per
client per round) and FedAvg (a client-size-weighted mean), the aggregated
global update is mathematically identical to a single full-batch gradient
step computed directly on the *pooled* data -- regardless of how that data
was partitioned into clients. (Each client's full-batch gradient is a sum
over its own rows; the size-weighted average of those sums *is* the
pooled full-batch gradient, by linearity.) The partition scheme becomes
invisible to the trained model under this specific configuration.

**Confirmed this isn't a measurement bug two ways:**
1. Per-client *local* parameter divergence (before aggregation, the same
   quantity tracked for Arm 4/5) is genuinely nonzero and rises
   monotonically with heterogeneity (0.034 at α=100 to 0.352 at α=0.1) --
   individual clients really do end up with different local updates.
2. But the *aggregated* global model erases that difference exactly, for
   the mathematical reason above.

**Consequence:** the weakened MLP's 36.77pp "decline" is not a training
heterogeneity effect -- the model powering every condition is the same
object. It's an **evaluation-composition effect**: the same fixed,
mediocre classifier scored against differently-skewed held-out client test
slices produces very different accuracy numbers purely because of what's
in each slice (at α=0.1, some clients' test sets are nearly single-class,
and a roughly-50/50 fixed classifier's accuracy on a near-pure-negative or
near-pure-positive slice swings hard depending on which way it happens to
lean). This is a real, measured number, but it is **not measuring the same
thing** the VQC's and full MLP's worst-client declines measure (both of
which come from genuinely different trained models per condition -- their
own final-round divergence is nonzero and condition-dependent, unlike this
config's).

## What this does and doesn't answer

**Does not** cleanly resolve the original capacity-confound question --
the calibrated weak-MLP's degradation number isn't an apples-to-apples
comparator, for the reason above, so it can't be used to argue the VQC's
relative robustness is or isn't a capacity artifact.

**Does** surface something worth keeping: a config that looks like
reasonable "early stopping" (`rounds=4, local_epochs=1`) can silently
collapse FedAvg into a partition-invariant training regime for models
this small. Any future capacity/training-budget calibration in this
project should check for this degeneracy directly (e.g. compare trained
parameters across conditions) before trusting a degradation number from a
heavily-reduced training budget.

**One directionally interesting fact despite the caveat:** local client
divergence *before* aggregation still rises monotonically with
heterogeneity even in this degenerate regime (0.034 -> 0.352) -- the
underlying drift mechanism this project measures is present even when
FedAvg's aggregation happens to cancel it out at the global level. That's
consistent with, not contradictory to, the rest of this project's
findings (D-028, D-034).

## Raw numbers

| condition | weak-MLP global | weak-MLP worst-client | weak-MLP local divergence (final round) |
|---|---|---|---|
| 100 | 0.6001 | 0.5236 | 0.0338 |
| 1.0 | 0.6001 | 0.3364 | 0.2016 |
| 0.5 | 0.6001 | 0.2274 | 0.2720 |
| 0.1 | 0.6001 | 0.1559 | 0.3516 |
| natural | 0.6001 | 0.4315 | 0.2087 |

(Global accuracy shown to 4dp is exactly 0.6001 at every condition, per
seed/fold, as described above.)
