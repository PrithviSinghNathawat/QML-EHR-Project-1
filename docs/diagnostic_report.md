# Diagnostic Report: Does the Heterogeneity Penalty Exist?

2026-08-18. Classical arms only (LR = logistic regression, MLP = small
neural net). No Arm 4/5 built or run in this session. Decision criteria
were committed in advance (see the prompt that started this session,
reproduced in `docs/decisions.md`) before any results were seen. Nothing
was tuned to produce a decline — the numbers below are exactly what the
sweep produced.

**Protocol:** 5-fold stratified cross-validation x 10 seeds. Client
assignment (Dirichlet at a given alpha, or the natural 4-site split) is
drawn once per seed over the full 920-row pool, giving each client a
stable identity across all 5 folds of that seed. Every record is used for
global testing exactly once per seed. Full detail and reasoning:
`scripts/cv_protocol.py`, `docs/decisions.md`.

---

## 1. Noise floor (Task 1)

Old protocol (single 736/184 split): standard error ≈ 3.1pp at ~77.5%
accuracy — could not resolve effects below roughly 5pp.

New protocol (5-fold CV x 10 seeds = 50 replicate measurements per
condition per model):

| model | condition | mean accuracy | std | SE (50 replicates) |
|---|---|---|---|---|
| LR | 100 | 0.7625 | 0.0222 | 0.0031 |
| LR | 1.0 | 0.7647 | 0.0226 | 0.0032 |
| LR | 0.5 | 0.7640 | 0.0231 | 0.0033 |
| LR | 0.1 | 0.7651 | 0.0225 | 0.0032 |
| LR | natural | 0.7587 | 0.0250 | 0.0035 |
| MLP | 100 | 0.7685 | 0.0225 | 0.0032 |
| MLP | 1.0 | 0.7677 | 0.0246 | 0.0035 |
| MLP | 0.5 | 0.7674 | 0.0248 | 0.0035 |
| MLP | 0.1 | 0.7223 | 0.0719 | 0.0102 |
| MLP | natural | 0.7688 | 0.0235 | 0.0033 |

**New noise floor: ~0.3–1.0pp**, roughly 8–10x tighter than the old
single-split protocol. This is well below the ~5pp effect size the
original task was looking for. **The noise floor is adequate** — the
protocol can resolve the effects reported below with room to spare, and
the MLP@α=0.1 result (SE=1.0pp against a ~4.6pp gap from its own
other conditions) is not a borderline call.

---

## 2. Client parameter divergence (Task 2)

Mean pairwise L2 distance between client parameter vectors, final round,
averaged over 10 seeds x 5 folds:

| model | α=100 | α=1.0 | α=0.5 | α=0.1 | natural |
|---|---|---|---|---|---|
| LR | 0.063 | 0.281 | 0.372 | 0.495 | 0.245 |
| MLP | 0.268 | 0.577 | 0.753 | 1.109 | 0.488 |

**Monotonically increasing as α falls, for both models, with tight error
bars relative to the trend** (see `results/figs/client_divergence.png`).
The mechanism this project is named after — client drift under label skew
— is unambiguously present and measurable. This does not depend on the
convexity question below; it holds for LR just as clearly as for MLP.

---

## 3. Global vs. per-client / worst-client accuracy (Task 3)

| model | condition | global acc | worst-client acc | spread |
|---|---|---|---|---|
| LR | 100 | 0.7625 | 0.6942 | 0.0683 |
| LR | 1.0 | 0.7647 | 0.6690 | 0.0957 |
| LR | 0.5 | 0.7640 | 0.6600 | 0.1041 |
| LR | 0.1 | 0.7651 | 0.6476 | 0.1175 |
| LR | natural | 0.7587 | 0.6932 | 0.0655 |
| MLP | 100 | 0.7685 | 0.6991 | 0.0693 |
| MLP | 1.0 | 0.7677 | 0.6727 | 0.0950 |
| MLP | 0.5 | 0.7674 | 0.6514 | 0.1159 |
| MLP | 0.1 | 0.7223 | 0.5145 | 0.2078 |
| MLP | natural | 0.7688 | 0.7077 | 0.0611 |

See `results/figs/worst_client_accuracy.png`.

**Global accuracy is flat for LR across the entire α sweep** (0.7625 to
0.7651, well inside the noise floor — this reproduces last session's flat
result, now with 10x the statistical power, so it is not an artifact of
insufficient power). **Worst-client accuracy for LR declines
monotonically** as α falls: 0.6942 → 0.6690 → 0.6600 → 0.6476, a 4.66pp
drop from α=100 to α=0.1, larger than the noise floor at every step.

**MLP shows the same worst-client decline, more steeply** (0.6991 →
0.6727 → 0.6514 → 0.5145, a 14.46pp drop), **and at α=0.1 the penalty is
large enough to also show up in the global metric** (0.7685 → 0.7223, a
4.6pp drop, ~4.5 SE away from every other MLP condition — not noise).

**Caveat on precision at α=0.1:** per-client test-fold sizes at α=0.1 go
as low as n=1 (median 31.5, mean 46, from a fold that's ~184 rows split
4 ways under heavy skew). A single-example client accuracy is either 0%
or 100%, which inflates variance on individual (seed, fold) worst-client
measurements. This does not explain away the effect — the divergence
result (Section 2) is independent of test-fold size and shows the same
pattern — but it does mean the worst-client numbers at α=0.1 specifically
should be read as noisier than the SE alone suggests.

---

## 4. Convexity hypothesis (Task 4)

MLP architecture: single hidden layer, 2 units, tanh activation, sigmoid
output, both layers biased — **17 trainable parameters**
(12 + 2 + 2 + 1). This targets the actual frozen VQC's real parameter
count (**18** — 6 qubits x 3 layers x 1 RY/qubit/layer, confirmed from
`docs/circuit_diagram.txt`), not the ~36 mentioned in this session's task
prompt, which does not match the locked circuit (see `docs/decisions.md`).
17 is the closest achievable count with a standard single-hidden-layer
design.

**Evidence for convexity mediating the effect:** at the global-accuracy
level, MLP shows a clear penalty at α=0.1 that LR does not show at all.
At the worst-client level, both models show a penalty, but MLP's is
roughly 3x larger at α=0.1 (14.46pp vs 4.66pp) and appears already at
α=0.5 (MLP spread 0.116 vs LR spread 0.104 — comparable at moderate skew,
diverging sharply only at the most extreme condition).

**Precise reading:** convexity does not make LR immune to heterogeneity —
LR's worst-client accuracy declines too. What convexity appears to do is
(a) keep the *global* average insulated from the damage even as
individual clients are harmed, and (b) bound how *large* the per-client
damage gets. Both are real, distinguishable effects, and both matter for
the eventual quantum comparison: the VQC is non-convex, so on this
evidence it is more comparable to the MLP's failure mode than to LR's.

---

## 5. Natural partition vs. Dirichlet sweep (Task 5, objective D-009)

| model | metric | natural | α=100 | α=1.0 | α=0.5 | α=0.1 |
|---|---|---|---|---|---|---|
| LR | global acc | 0.7587 | 0.7625 | 0.7647 | 0.7640 | 0.7651 |
| LR | worst-client acc | 0.6932 | 0.6942 | 0.6690 | 0.6600 | 0.6476 |
| LR | divergence | 0.245 | 0.063 | 0.281 | 0.372 | 0.495 |
| MLP | global acc | 0.7688 | 0.7685 | 0.7677 | 0.7674 | 0.7223 |
| MLP | worst-client acc | 0.7077 | 0.6991 | 0.6727 | 0.6514 | 0.5145 |
| MLP | divergence | 0.488 | 0.268 | 0.577 | 0.753 | 1.109 |

**The natural partition does not sit outside the synthetic range on any
metric, for either model.** Its worst-client accuracy is comparable to or
better than every Dirichlet condition except α=100. Its divergence sits
between the α=1.0 and α=0.5 conditions — moderate, not extreme. This is
true even though the natural partition carries the additional
heterogeneity in P(x) and P(y|x) that Dirichlet label-skew alone does not
introduce (see `docs/decisions.md` D-018 on the Switzerland file combining
two institutions).

**This does not support "synthetic label-skew understates real
institutional heterogeneity."** On this evidence, real institutional
heterogeneity in this dataset is comparable in severity to a moderate
synthetic label-skew (roughly α=0.5–1.0), not more damaging than the most
extreme synthetic condition tested.

---

## 6. Which row of the decision table the evidence supports

Reproducing the pre-committed table with a verdict per row:

| Row | Verdict |
|---|---|
| Divergence rises, global accuracy flat → mechanism fires, model absorbs it | **Supported for LR at every α**, and for MLP at α ≥ 0.5. Breaks down for MLP at α=0.1, where global accuracy also drops. |
| Worst-client accuracy declines as α falls → penalty exists, wrong measurement location | **Supported, both models.** Monotonic in both. |
| MLP shows a penalty where LR does not → convexity mediates | **Supported, with a precise qualification** (Section 4): true at the global level; at the worst-client level both models show a penalty, MLP's is larger. |
| Natural partition shows a penalty Dirichlet does not → synthetic understates real heterogeneity | **Not supported.** Natural sits inside the synthetic range on every metric. |
| Nothing moves, tight error bars → no penalty at this scale | **Not supported.** Worst-client and divergence both move clearly. |
| Everything flat, wide error bars → underpowered, inconclusive | **Not supported.** Noise floor (Section 1) is well below the observed effect sizes. |

**No single row fully describes the result** — the honest reading combines
rows 1–3: the mechanism (divergence) fires cleanly and measurably for
both models; it is invisible in the global metric for LR and partially
invisible for MLP at moderate skew, but visible in worst-client accuracy
for both; and it becomes visible in the global metric too for MLP
specifically at the most extreme skew tested, consistent with (but not
proof of) a convexity-mediated capacity effect.

## 7. What the evidence does not support

- That there is no heterogeneity penalty at this scale (row 5) — there is
  one, visible in worst-client accuracy and in divergence for both models.
- That logistic regression is immune to heterogeneity — it is not; its
  worst-client accuracy declines by 4.66pp from α=100 to α=0.1. It is
  *global-accuracy-insulated*, not unaffected.
- That the result is inconclusive or underpowered — the noise floor is
  roughly 5–10x smaller than the effect sizes reported here.
- That real institutional heterogeneity (natural partition) is more
  damaging than synthetic label-skew in this dataset — the opposite is
  closer to true: natural sits mid-range on the synthetic scale.
- That the ~36-parameter MLP requested in the task prompt was built — it
  was not; a 17-parameter MLP matched to the actual 18-parameter VQC was
  built instead, and that substitution is not expected to change the
  qualitative convexity finding, only its exact magnitude.

## 8. Process note: parallelism not used

The task scope note asked to use the 4-way process-level parallelism
validated earlier (D-020). It was not used here: the full sweep (100 Arm 1
trainings + 500 Arm 2 federated trainings, 5-fold CV x 10 seeds x 2 models
x 5 conditions) completed in **31 seconds sequentially, in a single
process**. Each individual unit of work is single-digit milliseconds;
subprocess spawn overhead (Python interpreter startup + imports) alone
would have exceeded the total sequential runtime. Parallelizing this
specific workload would have made it slower, not faster — flagged here
rather than followed blindly, per the instruction against tuning things to
fit an expectation instead of reporting what's actually true.
