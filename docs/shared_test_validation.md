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

## On the MLP discrepancy — reported, not resolved

**Not adjusted to improve agreement**, per instruction. One plausible account,
offered as a consideration rather than a resolution: the two methods are not
strictly estimating the same statistic. The composition residual is specifically
the training-driven change in *worst-client* (minimum-subgroup) accuracy; the
shared-test decline is the training-driven change in *pooled/average* accuracy
across the whole fixed test set. If MLP's training-induced damage concentrates
disproportionately on whichever client ends up worst-off, rather than spreading
evenly across the population, a larger worst-client-training-effect than
pooled-training-effect is what that would look like — not necessarily evidence one
method is wrong. This has not been checked further and should not be read as
settling the discrepancy; it is offered so the comparison isn't reported as an
unexplained number with no candidate account, while still leaving the
interpretation open.

**What this means for the paper's framing, left for Prithvi to decide:** MLP's
composition-based residual (+13.47pp) is corroborated in *direction* and
*significance* by an independent method, but not in *magnitude* — the honest
range, given both estimates, is "somewhere between +4.6pp and +13.5pp of real
training damage," not a single confirmed number. Arm 3's FedProx results
(P-005, `docs/arm3_report.md`) were framed against the +13.47pp figure; whether
that framing should be revised given this validation is a decision for the
Results write-up, not made here.

## Connection to existing practice

This shared-test protocol is exactly what NIID-Bench (Li et al., cited as [8] in
`paper/02_related_work.md`) uses — their Table III reports only pooled/global
top-1 accuracy across their Dirichlet sweep, never per-client. This validation's
method is the one already established in that precedent, not a novel construction.
