# Validation: Shared-Test Degradation vs. the Composition Decomposition

2026-08-22. A pre-writing validation gate, not a new experiment — no retraining.
The composition decomposition (D-044 onward) estimates the training effect
*indirectly*: freeze the α=100 model, re-score it against every α's client-local
test slices, and treat the gap between that and the observed decline as the
training-driven residual. This is a second, *independent* way to estimate the same
underlying quantity: evaluate the already-trained per-α models on a single pooled
test set that never varies with α. Since the test data is constant, any accuracy
change across α must be training-driven — no evaluation-composition confound is
even possible in this protocol.

**No retraining was needed.** The "global accuracy" metric already computed for
every arm turns out to already satisfy this protocol exactly:
`scripts/cv_protocol.py:fit_transform_fold(df_train_fold, df_test_fold)` takes no
α/condition argument at all — the held-out test set is a pure function of
`(seed, fold)`. Verified empirically (not just by reading the signature): called it
three times for the same `(seed=0, fold=0)` and confirmed bit-identical `X_test`/
`y_test` arrays each time. The global-accuracy numbers already in
`results/diagnostic_results.csv` and `results/arm4_diagnostic_results.csv` are this
validation's raw data.

## Method

For each model (LR: Arm 2 FedAvg; MLP: Arm 2 FedAvg; VQC: Arm 4 FedAvg), and each of
the 50 `(seed, fold)` replicates: take the model as actually trained under α=100 and
under α=0.1 (no retraining, no freezing — the real per-α trained models), evaluate
both on that replicate's fixed shared test set, and compute the paired decline.

## Result

| model | shared-test decline (α=100→0.1) | SE | ~SE from zero | composition residual (D-044 onward) |
|---|---|---|---|---|
| LR | -0.26pp | 0.16pp | 1.6 | +0.84pp |
| MLP | **+4.62pp** | 1.05pp | **4.4** | **+13.47pp** |
| VQC | +0.35pp | 0.34pp | 1.0 | -1.25pp |

## Verdict, per model

**LR: agree.** Both methods land on "no real training effect" — shared-test is
small, slightly negative, and not statistically distinguishable from zero (1.6 SE);
the composition residual is small and positive (+0.84pp). Same qualitative
conclusion from two independent methods.

**VQC: agree.** Same pattern — shared-test is small and not significant (1.0 SE);
composition residual is small and negative. Both consistent with zero, both
consistent with each other.

**MLP: disagree materially.** Both methods agree a real training effect exists
(shared-test's +4.62pp is 4.4 SE from zero — not noise). They disagree sharply on
*magnitude*: composition says +13.47pp, shared-test says +4.62pp — roughly a 3x
gap, well outside what either method's own noise level would explain.

## Follow-up: is the MLP gap just a mean-vs-minimum statistic mismatch?

**Tested directly, 2026-08-22. It is not.** The candidate explanation above (pooled
= a mean, composition residual = about a minimum, and minimums are more sensitive to
spread) predicts that matching the *statistic* — taking a minimum on the same
alpha-independent shared test set — should move the shared-test estimate toward the
composition residual. It was tested by constructing a second, independent
partition: a fixed, seeded, alpha-*independent* assignment of each fold's 184 test
rows into 4 groups (not the Dirichlet client assignment — a plain random 4-way
split, same seed formula reused for both the alpha=100 and alpha=0.1 evaluation of a
given replicate, `scripts/shared_test_worst_group.py`), then took the minimum
accuracy across those 4 fixed groups instead of the pooled mean. No retraining for
LR/MLP (the exact already-existing trained models are deterministically reproduced
and evaluated on the new partition); VQC used the smaller n=2 sample already saved
from the earlier angle-capture work (full n=50 VQC retraining would cost ~14 hours
and was not authorized here — flagged, not run).

| model | shared-test **pooled** decline | shared-test **worst-group** decline | composition residual |
|---|---|---|---|
| LR | -0.26pp (SE 0.16pp) | -0.06pp (SE 0.37pp, n=50) | +0.84pp |
| MLP | +4.62pp (SE 1.05pp) | **+5.00pp (SE 1.14pp, n=50, 4.4 SE from zero)** | **+13.47pp** |
| VQC | +0.35pp (SE 0.34pp) | -0.81pp (n=2 only, not powered for a confidence claim) | -1.25pp |

**For MLP, switching from a mean to a minimum statistic — while keeping the test
partition alpha-independent — barely moved the number** (+4.62pp pooled vs. +5.00pp
worst-group, well within each other's SE). It did **not** move toward the
composition residual's +13.47pp. **The mean-vs-minimum hypothesis is rejected by
this data.**

## On the MLP discrepancy — reported, not resolved, narrowed by the follow-up

**Not adjusted to improve agreement**, per instruction, in either check. The
original candidate explanation (different statistics, mean vs. minimum) does not
hold up once tested directly — the minimum operator itself is not what's driving
the gap. What differs between the composition-decomposition's residual and *both*
shared-test estimates (pooled and fixed-worst-group) is that the decomposition's
residual is computed using the **alpha-dependent Dirichlet client partition** for
its "composition-only" comparison — which, especially at low alpha, produces highly
uneven client group sizes and class composition — while this validation's groups
are a plain, evenly-random, alpha-independent 4-way split. Per the instruction's own
fallback: **this points to the minimum operator's sensitivity to the *specific,
widening, alpha-dependent* spread the Dirichlet partition produces (not to minimum
vs. mean in general), and the composition-decomposition's residual for MLP needs an
explicit caveat rather than being reported as one clean +13.47pp number.**

## Composition split recomputed under the shared-test training estimate (P-008, 2026-08-28)

Task: recompute `implied composition = observed decline − shared-test training
effect` (and its %) for all three model families, and set it next to the
decomposition's own split, so both are visible together.

| model | observed decline | decomposition: composition-only (%) | decomposition residual (training effect) | shared-test training effect (pooled → worst-group) | implied composition under shared-test (%) |
|---|---|---|---|---|---|
| LR | 4.66pp | 3.82pp (82.0%) | +0.84pp | −0.26pp → −0.06pp | 4.72–4.92pp (**101–106%**) |
| MLP | 18.46pp | 4.99pp (27.0%) | +13.47pp | +4.62pp → +5.00pp | 13.46–13.84pp (**73–75%**) |
| VQC | 7.25pp | 8.50pp (117.2%) | −1.25pp | +0.35pp → −0.81pp | 6.90–8.06pp (**95–111%**) |

**Stated plainly: composition dominates for all three model families once the
training effect is pinned to the shared-test estimate.** Under the decomposition
alone, MLP looked training-dominated (73% residual). Recomputed against the
shared-test estimate, MLP's split inverts to ~74% composition / ~26% training —
matching the pattern already seen for LR and VQC. This is a stronger claim than
previously reported and is logged here for the writing phase, not asserted as the
final number (see below).

**Framing the gap correctly: the two methods agree wherever the training effect is
near zero, and diverge only where a real training effect exists.** LR's interaction
(decomposition residual minus shared-test training effect) is +0.9 to +1.1pp; VQC's
is −0.4 to −1.6pp — both small, both consistent with the "agree" verdict already
reached for these two models above. MLP's interaction is **+8.47 to +8.85pp
(roughly 8.5pp)** — the one case where a real training effect exists, and the one
case where the two methods diverge materially. That pattern is consistent with the
decomposition residual absorbing a **model-partition interaction term**: the
decomposition's composition-only arm is scored against the real, alpha-dependent
Dirichlet partition, so wherever a model's parameters genuinely shift with training,
that shift interacts with the same widening, uneven partition the composition-only
arm also uses. This is a term the composition-vs-training two-way split was never
built to isolate — flagged as such in this project's own methodology notes before
D-047 was ever run ("composition still contributes a real, non-negligible share of
the reported worst-client movement, which is a distinct question from whether the
model itself changes at all"). We have now measured that interaction at roughly
8.5pp for MLP. **Reported as a measured limitation of the two-way decomposition, not
a failure of either estimate.**

**No estimate is declared correct.** MLP's genuine training effect is reported as
the range **4.6pp to 13.5pp**: the shared-test estimate (4.6–5.0pp) as the lower,
directly-measured bound; the decomposition residual (13.47pp) as the upper bound,
now understood to include the unisolated interaction term; the ~8.5pp gap between
them is the interaction itself, named and measured, not resolved. Arm 3's FedProx
results (P-005, `docs/arm3_report.md`) were framed against the higher, now-partially-
explained figure — this recomputation strengthens (does not settle) the case for
revising that framing; still Prithvi's call, not decided here.

## Connection to existing practice

This shared-test protocol is exactly what NIID-Bench (Li et al., cited as [8] in
`paper/02_related_work.md`) uses — their Table III reports only pooled/global
top-1 accuracy across their Dirichlet sweep, never per-client. This validation's
method is the one already established in that precedent, not a novel construction.
