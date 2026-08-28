# Dataset 2 — Composition-vs-Training Decomposition at K=4 and K=130 (P-014)

2026-08-28. Minimal scope, classical arms only, per instruction: LR and MLP
only, FedAvg only, Dirichlet only, decomposition applied, run at K=4 and
K=130 on the same α grid and protocol (20 rounds, 5 local epochs, 5-fold
stratified CV × 10 seeds) to isolate client count as the variable. No
quantum, no FedProx, no circular mean, no natural-partition arm (none is
possible — see below).

## 0. What had to be fixed before any of this was measurable

Two problems surfaced during setup, both logged in full in `docs/decisions.md`
(P-011, P-013) — summarized here for context:

1. **Repeat-patient leakage.** First-encounter-per-patient filter (Strack et
   al. 2014's own convention) applied before anything else: **71,518
   encounters remain** (from 101,766 raw). Positive rate (early readmission,
   `readmitted == "<30"`) drops from 11.16% to **8.80%** after the filter —
   confirming the leakage was real (repeat visits skewed toward higher
   readmission rates).
2. **Feature retention.** Pooled ≥85%-present rule (D-016/D-019's rule,
   re-evaluated on the 71,518-row filtered population, no site axis exists
   for this dataset) plus a new near-zero-variance filter (drop if one
   value's share ≥99%) that dataset 1 never needed: 15 of 23 medication
   columns are >99% a single value, 2 (`examide`, `citoglipton`) are
   literally constant. **24 features retained** (23 numeric/ordinal + a
   one-hot `race`), diagnosis codes (`diag_1/2/3`) explicitly excluded —
   high-cardinality ICD9 encoding is a real design decision outside
   "minimal scope," flagged rather than done silently
   (`scripts/dataset2_preprocessing.py`).

Then, at evaluation time, two more problems surfaced and were fixed in
sequence (both required stopping and asking — see P-011, P-013):

3. Plain 0.5-threshold accuracy is degenerate at 8.8% prevalence — fixed
   with balanced accuracy in evaluation only (P-011).
4. Balanced accuracy alone was insufficient: both LR and MLP, trained
   exactly as dataset 1's models are, converge to a genuine constant
   "always predict not-readmitted" classifier at every α and K (verified
   directly, up to 2,000 epochs). Fixed with inverse-frequency class
   weighting inside `fit()` (new classes in `scripts/models_weighted.py`,
   `models.py`/`models_mlp.py` untouched) — P-013.

Every number below uses: first-encounter filter, 24-feature retention,
balanced-accuracy evaluation, class-weighted training.

## 1. Dataset 1, re-evaluated three ways (P-011 condition, extended by P-013)

The metric and training changes were both forced by dataset 2 and then
applied retroactively to dataset 1 so the cross-dataset comparison below
compares like quantities. All three states reported for traceability —
none replaces the original D-047/D-049 numbers already in the paper.

| Model | State | Observed | Composition-only (share) | Residual | Shared-test pooled |
|---|---|---|---|---|---|
| LR | **original** (unweighted, plain accuracy — D-047) | 4.66 pp | 3.82 pp (82.0%) | 0.84 pp | −0.26 pp |
| LR | unweighted, balanced accuracy (P-011) | 5.91 pp | 5.78 pp (97.9%) | 0.12 pp | −0.09 pp |
| LR | **weighted, balanced accuracy (P-013, final)** | 9.29 pp | 5.34 pp (57.4%) | 3.95 pp | 3.27 pp |
| MLP | **original** (unweighted, plain accuracy — D-047) | 18.46 pp | 4.99 pp (27.1%) | 13.47 pp | 4.62 pp |
| MLP | unweighted, balanced accuracy (P-011) | 16.39 pp | 7.40 pp (45.2%) | 8.99 pp | 5.03 pp |
| MLP | **weighted, balanced accuracy (P-013, final)** | 12.63 pp | 6.79 pp (53.7%) | 5.85 pp | 2.86 pp |

Class weighting is not a free change even on the roughly-balanced dataset 1:
LR's genuine training residual moves from 0.84 pp (original) through 3.95 pp
(weighted) — LR was under-incentivized to learn *at all* under plain
unweighted training, so part of dataset 1's original "LR barely trains"
finding was itself a smaller instance of the same mechanism dataset 2
exposed at full scale. The decomposition/shared-test **residual and
shared-test pooled decline move closer together** for LR under weighting
(3.95 vs. 3.27, an interaction of ~0.7 pp — much tighter than the original
0.84 vs. −0.26). For MLP they also move closer (5.85 vs. 2.86, interaction
~3.0 pp, down from the original ~8.5 pp reported in P-008). **Under
class-weighted training, both models' decomposition residuals move toward
their shared-test estimates, not away** — worth noting as a secondary
finding, though not this task's primary question.

## 2. Dataset 2, full grid (LR/MLP × K=4/K=130 × α, class-weighted, balanced accuracy)

**K=130, α=0.1 is not included: not attempted.** Verified empirically
(`scripts/dataset2_cv_protocol.py`'s `InfeasiblePartition` check) that every
Dirichlet(0.1) draw over 130 bins produces at least one zero-row client —
0 of 1,000+ independent draws succeeded even at a floor of 1 row, let alone
D-022's floor of 15. Not a rare-draw problem fixable with more attempts; a
structural property of extreme-skew Dirichlet spread over many bins on this
dataset. The primary K=4-vs-K=130 comparison below therefore uses α=1.0 and
α=0.5 — the two conditions available at **both** K values — as the clean,
client-count-only comparison; K=4's own α=0.1 result is reported alongside
for completeness but has no K=130 counterpart.

| Model | K | α pair | Observed | Decomp. composition-only (share) | Decomp. residual | Shared-test pooled (SE) | Implied composition under shared-test (share) | Interaction |
|---|---|---|---|---|---|---|---|---|
| LR | 4 | 100→1.0 | 0.80 pp | 0.65 pp (80.6%) | 0.16 pp | 0.14 pp (0.07) | 0.57–0.66 pp (71–82%) | −0.08 to 0.01 pp |
| LR | 4 | 100→0.5 | 3.89 pp | 6.99 pp (179.6%) | **−3.10 pp** | 1.04 pp (0.24) | 2.85–3.03 pp (73–78%) | −4.14 to −3.96 pp |
| LR | 4 | 100→0.1 | 16.14 pp | 21.04 pp (130.4%) | **−4.90 pp** | 3.79 pp (0.25) | 12.35–12.61 pp (77–78%) | −8.69 to −8.43 pp |
| MLP | 4 | 100→1.0 | 0.99 pp | 0.87 pp (88.2%) | 0.12 pp | 0.22 pp (0.08) | 0.72–0.77 pp (73–78%) | −0.15 to −0.10 pp |
| MLP | 4 | 100→0.5 | 3.25 pp | 3.92 pp (120.6%) | −0.67 pp | 0.44 pp (0.12) | 2.81–2.96 pp (87–91%) | −1.11 to −0.95 pp |
| MLP | 4 | 100→0.1 | 15.71 pp | 13.71 pp (87.2%) | 2.00 pp | 1.38 pp (0.26) | 14.33–14.91 pp (91–95%) | 0.62–1.21 pp |
| LR | 130 | 100→1.0 | 14.84 pp | 18.94 pp (127.7%) | **−4.11 pp** | 0.95 pp (0.20) | 13.81–13.89 pp (93–94%) | −5.14 to −5.06 pp |
| LR | 130 | 100→0.5 | 29.52 pp | 23.81 pp (80.6%) | 5.72 pp | 4.19 pp (0.16) | 25.34–26.34 pp (86–89%) | 1.53–2.53 pp |
| MLP | 130 | 100→1.0 | 12.23 pp | 13.89 pp (113.6%) | **−1.67 pp** | 0.26 pp (0.12) | 11.97–12.08 pp (98–99%) | −1.92 to −1.81 pp |
| MLP | 130 | 100→0.5 | 26.89 pp | 20.64 pp (76.7%) | 6.26 pp | 1.21 pp (0.26) | 25.68–26.27 pp (96–98%) | 5.05–5.64 pp |

All shared-test SEs are well under their point estimates (many are 4–26
standard errors from zero) — the shared-test estimate is a real, precisely
measured signal in every row, not noise. **Bolded residuals are negative**:
the decomposition claims composition *more than fully* explains the
observed decline, at exactly the configurations (LR at K=4/α=0.5, K=4/α=0.1,
K=130/α=1.0; MLP at K=130/α=1.0) where the shared-test estimate finds a
clearly real, non-trivial positive training effect instead.

## 3. Does composition share grow with client count, as predicted (P-009)?

**Two different answers depending on which share is measured — and the
difference is itself the finding.**

**Using the decomposition's own self-reported share: no clean answer.**
Matched at α=1.0: LR 80.6% (K=4) → 127.7% (K=130), MLP 88.2% → 113.6% — grows,
consistent with the prediction. Matched at α=0.5: LR 179.6% (K=4) → 80.6%
(K=130), MLP 120.6% → 76.7% — **shrinks**, opposite the prediction. This is
not a real "shrinking composition" effect; it is the decomposition's own
residual becoming unstable (going negative, share exceeding 100%) at
different (model, K, α) combinations for reasons that don't track client
count monotonically — see Section 2's bolded rows.

**Using the shared-test-implied share (the reliable one — matches the P-008
finding that the decomposition's residual is the less trustworthy of the
two wherever a real training effect exists): a clean, consistent yes.**
Matched at α=1.0: LR 71–82% → 93–94%, MLP 73–78% → 98–99%. Matched at α=0.5:
LR 73–78% → 86–89%, MLP 87–91% → 96–98%. **All four matched comparisons grow
substantially from K=4 to K=130**, for both models, at both alpha levels
available at both client counts. **The pre-registered prediction is
confirmed** on the metric that can actually be trusted here.

## 4. Does the decomposition generalize beyond dataset 1?

**No — not as a standalone method.** On dataset 1, the decomposition and the
shared-test estimate agreed for LR and disagreed only for MLP (P-006/P-008),
which supported a working hypothesis that the decomposition was reliable
except where a model actually trains. On dataset 2, **LR disagrees with the
shared-test estimate at 3 of its 5 measured configurations** — the same
model family that was the "safe," agreeing case on dataset 1. At LR/K=4/α=0.5
and LR/K=130/α=1.0, the decomposition's residual is negative (claiming no
training effect, or a negative one) while the shared-test estimate is a
clearly real, many-SE-from-zero positive effect (1.04 pp/0.24 SE and
0.95 pp/0.20 SE respectively). The interaction term (decomposition residual
minus shared-test estimate) reaches **−5 to −8.7 pp** in these
configurations — larger in magnitude than the entire interaction P-008 found
for dataset 1's MLP (~8.5 pp), and opposite in sign at several points
(dataset 1's interaction was always positive; here it is negative at 6 of
10 rows and positive at 4).

**Reframing P-008's finding in light of this:** the interaction term is not
a fixed property of a model family (as "LR is safe, MLP isn't" would
suggest); it is a property of the *(model, partition, α)* configuration, and
it can be large in either direction. The decomposition should not be
reported as a standalone estimate on any new dataset without a shared-test
check alongside it — exactly the recommendation the rewritten paper
Conclusion (P-010) already makes, now with a second, independent
demonstration of why.

## 5. What this does *not* test

**No natural-partition arm, and no update to the α-calibration finding.**
Dataset 2 has no per-record hospital identifier (P-009) — there is nothing
to calibrate a natural partition against. `A-003`'s natural-partition
α ≈ 1.5 (95% CI 1.0–4.7) remains a dataset-1-only finding; nothing here
extends, confirms, or challenges it.

**Not a claim about real-world readmission prediction quality.** Balanced
accuracies here are modest (LR ~0.55, MLP ~0.51–0.58 depending on training
budget, both confirmed non-degenerate but far from strong classifiers) —
the point of this exercise is whether the composition-vs-training method
generalizes, not whether these are good clinical models. No hyperparameter
tuning was performed, per the explicit hard-stop condition.

## Summary

Composition dominates the observed decline in most configurations here too
(matching dataset 1's headline finding), and the pre-registered client-count
prediction is confirmed on the reliable (shared-test-implied) share — but
the decomposition method itself is measurably less reliable on dataset 2
than it appeared to be on dataset 1, disagreeing with the shared-test check
even for logistic regression, the model family that "worked" the first
time. The method generalizes as a *diagnostic paired with a shared-test
check*; it does not generalize as a standalone estimate.
