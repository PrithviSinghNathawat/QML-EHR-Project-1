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

**What this means for the paper's framing, left for Prithvi to decide:** MLP's
composition-based residual (+13.47pp) is corroborated in *direction* by two
independent methods, but its *magnitude* is now doubly unconfirmed — neither the
pooled (+4.62pp) nor the matched-minimum, alpha-independent-partition (+5.00pp)
estimate comes close to it, and the mean-vs-minimum explanation for that gap has
been tested and rejected. The most defensible reportable range narrows to roughly
+4.6pp to +5.0pp as the *validated* estimate, with +13.47pp reported as the
decomposition's own figure alongside an explicit caveat that it does not corroborate
under either independent check attempted so far. Arm 3's FedProx results (P-005,
`docs/arm3_report.md`) were framed against the higher, uncorroborated figure — this
narrows the case for revising that framing rather than settling it; not decided
here.

## Connection to existing practice

This shared-test protocol is exactly what NIID-Bench (Li et al., cited as [8] in
`paper/02_related_work.md`) uses — their Table III reports only pooled/global
top-1 accuracy across their Dirichlet sweep, never per-client. This validation's
method is the one already established in that precedent, not a novel construction.
