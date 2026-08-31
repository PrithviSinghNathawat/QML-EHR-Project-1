# Disentangling Evaluation Composition from Training Degradation in Federated Learning on Non-IID Clinical Data

*Draft v2 — placeholders filled from committed repo data and verified sources where possible. Every fill is sourced below and in the "Flags for Author Review" section at the end. Nothing here was filled from general knowledge alone where a real, traceable number existed — this project's own evidentiary rule ("if a number can't be traced to committed code, it doesn't go in the paper") was applied to this pass too.*

**Authors:** Ayuvi Chaudhary, Prithvi Singh Nathawat
**Affiliation:** School of Computer Science and Engineering, Vellore Institute of Technology, Vellore, India

---

## Abstract

Federated learning on non-independent and identically distributed (non-IID) clinical data is commonly evaluated by measuring each client's performance on its own held-out partition. When heterogeneity is treated as a swept experimental variable, this protocol introduces a confound: the client assignment determines both the training partition and the composition of each client's test set, so a model that has not changed can nonetheless appear to degrade. **We set out to propose a decomposition that separates genuine training degradation from this evaluation-composition artifact, and to validate it against a second, α-invariant shared-test protocol in the style of [6]. We did not end up with a validated correction. We end up instead with a diagnostic pair, and evidence that the correction we set out to build cannot be trusted alone under the exact conditions it was meant to diagnose — which we report as the finding, not as a setback to be minimised.** Across three local model families on a four-site clinical archive (logistic regression, a small multilayer perceptron, a variational quantum classifier) and, for the two classical families, a second, independently sourced clinical dataset at both 4 and 130 simulated clients, the two estimators **agree closely wherever the genuine training effect is near zero, and diverge — by as much as 8.7 percentage points, in either direction, unpredictably by model family — wherever a real training effect exists.** Critically, which model family is "safe" is not stable across datasets: logistic regression agreed with the shared-test estimate on the first dataset and disagreed with it, sometimes sharply, on the second. The divergence is therefore not a property of a model family; it is a property of the specific (model, partition, heterogeneity) configuration, and it cannot be predicted in advance from model identity alone. We name this divergence a model-partition interaction term that neither estimator isolates, and report it explicitly rather than resolving it to one number in every case where the two estimators disagree. A pre-registered prediction — that this interaction, and the composition confound generally, should worsen as client count grows — is confirmed on the second dataset's reliable statistic, at the heterogeneity level (α=0.5) where both client counts clear our own 2pp reporting floor for a percentage (Section III-C): the shared-test-implied composition share for the multilayer perceptron rises from 87–91% at 4 clients to 96–98% at 130, **meaning the confound is worst in exactly the many-client regime most federated learning studies actually use.** Our practical recommendation follows directly: prefer the shared-test protocol, which is already standard practice in evaluations such as [6] and is the more stable of the two estimators across both datasets we tested; where client-local evaluation under a heterogeneity sweep cannot be avoided, report both estimators and treat their divergence as a warning sign, not as noise to be averaged away. We additionally report that the first archive's natural inter-institutional heterogeneity corresponds to a Dirichlet concentration of approximately α ≈ 1.5 (95% CI: 1.0–4.7), milder than the α = 0.1 widely used to represent realistic non-IID conditions **and milder than the informal α ≈ 0.5–1.0 comparison used elsewhere in this project — see Flag 1**, and that the quantum classifier required approximately 13,300× the training time of a parameter-matched classical model while showing no measurable heterogeneity sensitivity. All results are reproducible from committed code.

**Index Terms** — federated learning, non-IID data, evaluation methodology, electronic health records, quantum machine learning.

---

## I. Introduction

Data protection regulation prevents clinical institutions from pooling raw patient records, motivating federated learning (FL), in which a shared model is trained across sites while data remains local and only model parameters are exchanged [1]. Real institutions differ systematically in patient population, referral pattern and recording practice, producing non-IID client data. FedAvg was identified as sensitive to this in its original formulation, and proximal regularisation was subsequently proposed to restrain the resulting client drift [2].

A parallel line of work established that aggregate performance provides no guarantee for any individual client: minimising a global objective can disproportionately disadvantage particular participants [3], [4]. This motivated fairness-oriented objectives that target the distribution of accuracy across clients rather than its mean [3], [5]. Consequently, client-level metrics — worst-client accuracy, accuracy variance — are now standard reporting practice.

**The problem we identify.** In studies that treat heterogeneity as a swept variable, client-local evaluation introduces a confound. The same client assignment that partitions training data also determines which test samples each client is evaluated on. As heterogeneity increases, each client's test partition becomes correspondingly skewed. A model whose parameters have not changed at all will therefore exhibit changing client-level performance. Observed degradation across a heterogeneity sweep conflates two distinct effects: genuine training damage, and a change in what is being measured.

This confound is structural rather than incidental. It does not arise in studies evaluating on a shared held-out set [6], and it is invisible at any single heterogeneity level, because a confound in a *trend* requires a trend to exist. It becomes visible only when heterogeneity is swept and evaluation is client-local — a combination that is uncommon, which may explain why we did not find prior work decomposing it.

**Contributions.**

1. **A diagnostic pair, not a single corrective method.** We construct two independent estimators of the genuine training-heterogeneity effect under a swept-heterogeneity, client-local evaluation protocol — a composition decomposition (applicable without re-training) and a shared-test re-evaluation on an α-invariant held-out set — and show that agreement between them is informative (it indicates no training effect worth arguing about) while disagreement is *also* informative: it identifies a model-partition interaction term that neither estimator isolates on its own. The pair is the contribution; neither estimator alone is presented as reliable.
2. **A demonstration that this interaction is unpredictable from model family alone.** On our first dataset, one classical model family (logistic regression) agreed with the shared-test estimator and only the non-convex family (a small multilayer perceptron) diverged from it — consistent with a "convexity protects you" story. On a second, independently sourced clinical dataset, that story fails: logistic regression itself diverges from the shared-test estimator at 3 of 5 measured configurations, sometimes by more than the first dataset's entire multilayer-perceptron interaction. We report this as evidence that the interaction term is a property of the specific (model, partition, heterogeneity) configuration, not a property of model family, and therefore cannot be predicted in advance and dismissed for models presumed "safe."
3. **Confirmation of a pre-registered prediction that the confound worsens with client count**, tested by isolating client count as the only varying condition (same dataset, same α grid, same protocol, 4 vs. 130 simulated clients): at α=0.5, the only heterogeneity level where both client counts clear our 2pp floor for a reportable percentage (Section III-C), the shared-test-implied composition share rises from 87–91% to 96–98% between the two client counts — the confound is largest in exactly the many-client regime most federated learning evaluations use.
4. A calibration of a real four-site clinical archive's inter-institutional heterogeneity against the synthetic Dirichlet scale the field uses.
5. Resource accounting for a variational quantum classifier under matched conditions, and an aggregation-geometry ablation with a verified null.

We registered two specific predictions before measuring their outcomes, rather than only after the fact: whether the quantum classifier's composition share would resemble the convex classical model's if its smaller observed degradation is a capacity artifact (logged as D-048, confirmed in D-049 — the share exceeded logistic regression's, more strongly than anticipated), and whether the composition confound would grow with client count (logged as P-009, confirmed in P-014 on the more reliable of our two estimators). Both predictions were confirmed by subsequent measurement.

---

## II. Related Work

### A. Federated optimisation under heterogeneity

FedAvg [1] established the standard protocol and identified non-IID client data as a source of degradation. FedProx [2] adds a proximal term to the local objective, penalising divergence from the global model received at the start of each round. NIID-Bench [6] provides the closest methodological precedent to our study design: a controlled comparison of federated algorithms across Dirichlet-partitioned non-IID settings, concluding that no algorithm dominates across conditions. It evaluates on a shared held-out set and is therefore structurally unaffected by the confound we describe.

### B. Client-level performance and fairness

Agnostic Federated Learning [4] established that the aggregate objective carries no per-client guarantee, proposing a minimax formulation over client mixtures. q-FFL [3] defines fairness as uniformity of the accuracy distribution across devices and reports, for the unweighted objective, that average accuracy falls approximately 6.6 percentage points across a heterogeneity sweep while worst-decile accuracy falls approximately 45 points. Ditto [5] treats fairness and robustness as jointly achievable through personalisation.

Our worst-client observation is therefore not new; client-level disparity under non-IID conditions is established prior art. **What we add is a decomposition of that disparity.** Applied to our own setting, it suggests a substantial share of the observed worst-client degradation — across all three model families we test — is attributable to evaluation composition rather than a genuine training-heterogeneity effect; Section V reports this in full, including where our two independent estimates of that share agree and where they diverge. We note that q-FFL's own reported sweep is itself conducted under client-local, matching-skew partitions, and is therefore **structurally exposed to the same confound we identify**. This is an exposure we have identified, not an error we have measured: we have not quantified the composition share of their effect sizes, which are considerably larger than ours in absolute terms, and we make no claim about whether or how much of their reported result would survive the same decomposition.

### C. Federated evaluation protocols

The distinction between local-test and global-test evaluation is recognised in personalised federated learning, where both are sometimes reported to separate personalisation from generalisation [15], and client bias in personalised FL evaluation has been noted [16]. This literature addresses evaluation *at* a given heterogeneity level. We did not find work decomposing the confound that arises when heterogeneity is *swept* under client-local evaluation.

### D. Synthetic partitioning

Dirichlet-based label partitioning [7] is the standard mechanism for synthesising non-identical client distributions and is the source of our independent variable. That work reports substantial global accuracy degradation for convolutional networks on ten-class vision data — a contrast we return to in Section VI, since our convex model shows none.

### E. Quantum federated learning

Variational quantum models have been federated across healthcare institutions [10], with Fisher-information-weighted aggregation proposed for non-IID conditions at fixed Dirichlet values [11]. A2G-QFL [12] observes that variational parameters are rotation angles on a periodic manifold, for which arithmetic averaging is not the appropriate operation, and proposes geometry-aware aggregation **(and, per our own check of [12], also a QoS/latency-based client-importance gain — the description here covers only the geometric half of their contribution; see Flag 3)**. This observation is theirs; we adopt it as an ablation baseline. Classical federated learning has been reported for other multi-institutional clinical prediction tasks [13], and specifically for this dataset family [14].

### F. Gap

We did not find prior work that decomposes observed client-level degradation into training and evaluation-composition components under a swept heterogeneity protocol, nor work calibrating a real multi-institutional clinical partition against the synthetic Dirichlet scale.

---

## III. The Decomposition

*(Unchanged from v1 — no placeholders in this section, and nothing here contradicted anything found in the repo.)*

### A. Motivation

Let clients be induced by an assignment parameterised by heterogeneity α. In the standard protocol this assignment determines both the training partition and, within each cross-validation fold, the test samples attributed to each client. Worst-client accuracy at heterogeneity α is

W(α) = min_k Acc( M(α), T_k(α) )

where M(α) is the model trained under α and T_k(α) is client k's test partition under α. Both arguments vary with α. Observed change in W therefore reflects change in the model, change in the evaluation partitions, or both.

### B. Procedure

We hold the model fixed and vary only the evaluation partitions. Define

W_comp(α) = min_k Acc( M(α₀), T_k(α) )

with α₀ the near-IID reference condition. Since M(α₀) does not depend on α, all variation in W_comp is attributable to evaluation composition. The residual

W_train(α) = [W(α₀) − W(α)] − [W_comp(α₀) − W_comp(α)]

estimates the training-heterogeneity effect.

### C. Properties and caveats

This is a first-order decomposition and does not isolate interaction terms; the residual should be read as an estimate rather than an exact partition.

The residual may be negative, as it is for one of our arms. A negative residual is read as no measurable training effect, not as a training benefit.

The composition term bundles two mechanisms: each client's test partition becoming individually harder, and the minimum operator selecting from a distribution of client accuracies whose spread widens with heterogeneity. We report them jointly and do not separate them.

The decomposition applies only where test data is partitioned by the same assignment used for training. Studies evaluating on a shared held-out set are structurally unaffected.

**Composition share is a percentage of a decline, and a percentage of a small quantity is not a stable statistic (P-021).** For either estimator, share = 100 − 100·(TE/observed), where TE is a training-effect estimate and observed is the observed worst-client decline. As observed → 0 for any fixed, nonzero TE, this ratio is unbounded — a property of the ratio itself, not of which estimator produced TE. **We therefore report a composition share only where the observed decline is at least 2 percentage points, both at the single reported estimate and, wherever a partition-size robustness sweep was run for that configuration (Section V-G), across the full sweep; below this floor, we report absolute values (observed decline, training effect, composition, all in percentage points) instead of a percentage.** This rule is general and applied consistently, not constructed for one result: it is triggered directly, without needing any robustness sweep, by two dataset-2 configurations whose raw observed decline already falls under 2pp (logistic regression and the MLP at K=4, α=100→1.0 — Section V.F), and it is triggered under the sweep by heart disease logistic regression, whose observed decline crosses below 2pp at a partition-size threshold of 15 (Section V-G) — the empirical case that motivated formalizing the rule.

---

## IV. Experimental Setup

### A. Dataset

The UCI Heart Disease archive [8], [9] comprises 920 records from four institutions: Cleveland Clinic Foundation (303), Hungarian Institute of Cardiology (294), V.A. Medical Center Long Beach (200), and the combined University Hospitals of Zurich and Basel (123). The final site therefore represents two institutions. The target is binarised by the standard convention distinguishing absence from presence of angiographic disease.

**Missing-value semantics.** A recorded zero denotes a measured negative finding and is retained as valid data; only `?` denotes an absent measurement. One exception applies: serum cholesterol of zero is physiologically implausible and treated as a missing-value code. All 123 Switzerland records and 24.5% of V.A. records carry zero cholesterol. This treatment is our inference; it is not documented as such by the dataset creators.

**Feature retention.** A feature measured at one institution but absent at others cannot form part of a matched feature set, and imputing a predominantly unmeasured column constitutes fabrication rather than imputation. We retain features measured in at least 85% of records at every site: `age`, `sex`, `cp`, `restecg`, `thalach`, `exang`. This threshold is not knife-edge — any value in (72.0%, 73.5%] yields the same six (`oldpeak`, the next-most-available excluded feature, is available in exactly 72.0% of V.A. records; `thalach`/`exang`, the least-available retained features, are available in exactly 73.5% of V.A. records — verified directly, `docs/decisions.md` D-021). It also coincides with our qubit budget, since angle encoding consumes one qubit per feature.

No dimensionality reduction is applied. Principal component analysis is ill-posed here: fitting globally requires pooling client data, violating the federated premise, while fitting per client yields client-specific projections that place parameters in different spaces and render aggregation meaningless.

Missingness indicators are not used, as missingness is strongly site-collinear in this archive and an indicator would function as a site identifier.

**Second dataset (Section V.F).** For the cross-dataset generalization check, we additionally use Diabetes 130-US Hospitals [17]: 101,766 encounters, reduced to 71,518 by first-encounter-per-patient filtering (one row per patient, the first encounter by encounter ID) to prevent a repeat visit from the same patient appearing on both sides of a train/test split — a leakage class absent from the first archive, which has one row per patient. Feature retention combines the same ≥85%-availability rule used for the first archive (pooled, since this dataset carries no per-site breakdown at all — see below) with a near-zero-variance filter (dropped if one value's share is ≥99%) that the first archive never required: 15 of 23 medication columns in this dataset are 99–100% a single value and would pass a missingness-only filter while carrying no signal. The two filters together retain 24 modeling features. The target is binarized as early readmission (`readmitted == "<30"`), matching the source publication's own framing; the positive rate is 11.16%, more imbalanced than the first archive's. Under the plain-accuracy, unweighted-training protocol used for the first archive, both classical model families converge to a constant majority-class classifier at this prevalence — confirmed as the genuine converged optimum, not undertraining, by running to 2,000 epochs and observing the model's own predicted probabilities move further from the decision threshold, not closer. Dataset 2 results therefore depart from the first archive's protocol in two respects: training uses inverse-frequency class weighting, and evaluation uses balanced accuracy (mean per-class recall) rather than plain accuracy; for a matched comparison, Section V.F reports the first archive's headline models under this same weighting scheme alongside its original unweighted numbers. Finally, no per-record hospital or facility identifier exists anywhere in the public release — confirmed by inspecting all 50 columns directly, not assumed from documentation — so unlike the first archive's four real sites, no natural-partition arm is possible for this dataset; client structure here is Dirichlet-only.

### B. Protocol

Four clients, matching the natural site count. Dirichlet(α) label partitioning with α ∈ {100, 1.0, 0.5, 0.1}, minimum client size 15 enforced by reject-and-redraw. Twenty communication rounds, five local epochs, 5-fold stratified cross-validation, ten seeds — 200 replicates per arm, giving a noise floor of 0.3–1.0 percentage points.

Local epochs are five rather than one because FedAvg with a single full-batch local epoch is algebraically equivalent to one centralised gradient step: the size-weighted mean of client gradients is the pooled gradient. Heterogeneity cannot affect training in that regime, which we confirmed empirically by observing bit-identical parameters across all α conditions at E = 1.

Worst-client accuracy is the minimum across clients of accuracy on each client's own held-out partition, computed per (seed, fold) and then averaged.

### C. Arms

| Arm | Local model | Aggregation |
|---|---|---|
| 1 | Classical | none (centralised) |
| 2 | Classical | FedAvg |
| 3 | MLP | FedProx, μ ∈ {0.01, 0.05, 0.1} |
| 4 | VQC | FedAvg |
| 5 | VQC | circular mean [12] |

Classical models: logistic regression as a convex reference, and a multilayer perceptron with two hidden units (17 trainable parameters). The variational quantum classifier uses six qubits, three layers and angle encoding (18 trainable parameters). The MLP is the matched comparator on parameter count and convexity; we note that parameter count is a proxy for capacity, not capacity itself.

### D. Implementation

PennyLane with the `lightning.qubit` state-vector simulator and adjoint differentiation. Adjoint differentiation requires access to intermediate quantum states and is therefore unavailable on physical hardware, where parameter-shift differentiation would be required at substantially greater cost. All quantum results are noiseless: expectation values are computed analytically, with no shot noise, decoherence or gate error.

Experiments ran on a consumer laptop (8 physical cores, 16 GB) with four-way run-level parallelism. Circuit-level parallelism offers no benefit in state-vector simulation, where cost scales with gate count × 2ⁿ rather than circuit depth.

**Reproducibility.** Every reported number is regenerable from committed code. This rule is not decorative: while persisting our analysis, we discovered that an earlier ad-hoc aggregation silently pooled two arms sharing a model label, producing errors of 0.25–0.5 percentage points that were invisible until the numbers were regenerated from code.

---

## V. Results

### A. Observed and decomposed degradation

| Model | Observed | Decomposition composition (share) | Decomposition residual | Shared-test training effect | Implied composition under shared-test (share) | Interaction |
|---|---|---|---|---|---|---|
| Logistic regression | 4.66 pp | 3.82 pp (share not reported — below the 2pp floor under the robustness sweep, §III-C, §V-G) | 0.84 pp | −0.26 to −0.06 pp | ≈4.7 pp (share not reported, same reason) | 0.90–1.10 pp |
| MLP | 18.46 pp | 4.99 pp (27%) | 13.47 pp | 4.62–5.00 pp | 13.46–13.84 pp (73–75%) | 8.47–8.85 pp |
| VQC | 7.25 pp | 8.50 pp (117%) | −1.25 pp | −0.81 to 0.35 pp | 6.90–8.06 pp (95–111%) | −1.60 to −0.44 pp |

**[[Corrected from v1 — see Flag 5. The VQC row read 7.30 pp / 8.55 pp (117%) in the draft; the source of record (`docs/decisions.md` D-049, and independently re-derived from committed code, `scripts/composition_summary.py`, this session) gives 7.25 pp / 8.50 pp. The residual (−1.25 pp) and rounded percentage (117%) happen to come out the same either way, which is likely why this went unnoticed — but the two input numbers were off by 0.05 pp each.]]**

Worst-client accuracy declines monotonically with α for all three model families, consistent with prior reports of client-level disparity [3]. The middle three columns reproduce the original single-method decomposition; the right three columns are an independent training-effect estimate obtained by re-evaluating the same trained models against a pooled, α-invariant held-out set (no retraining, following the shared-test evaluation protocol used by [6]), with the worst-group statistic (a minimum over a fixed, α-independent 4-way split of that same held-out set) reported as the range's upper bound so the two methods are compared on matched statistics, not just matched data. **The "Decomposition composition (share)" column is not robust to a minimum-partition-size filter for LR and MLP (Section V-G). The "Implied composition under shared-test" column is more resistant for MLP and VQC, but not for LR — its observed decline (4.66pp) crosses below the 2pp stability floor (§III-C) under the robustness sweep, so it is reported here in absolute terms rather than as a percentage, per that floor, not as a special case.**

**Composition dominates for all three model families under the shared-test estimate, not two of three.** For logistic regression and the quantum classifier, the two methods agree: both put the genuine training effect near zero (LR: 0.84 pp decomposition vs. −0.26 to −0.06 pp shared-test, an interaction of ~1 pp; VQC: −1.25 pp vs. −0.81 to 0.35 pp, an interaction of at most ~1.6 pp, both consistent with this study's 0.3–1.0 pp noise floor). **For logistic regression specifically, we report this in absolute terms rather than as a percentage, per the 2pp floor above: LR shows no measurable training effect (0.84 pp decomposition residual, −0.26 to −0.06 pp shared-test — both indistinguishable from this study's own noise floor); its observed decline is itself small (4.66 pp) and almost entirely composition (≈4.7 pp of it, by either estimator). A percentage built on top of that small a decline is not a stable way to state the same fact, which is why one is not reported here.** For the MLP, the two methods disagree in magnitude while agreeing in direction: both find a real, non-zero training effect (the shared-test estimate is 4.4 standard errors from zero on its own), but the decomposition's residual (13.47 pp) is 8.47–8.85 pp larger than the shared-test estimate (4.6–5.0 pp). This is exactly the pattern predicted by Section III-C's caveat that the decomposition is first-order and does not isolate interaction terms: the two methods track each other closely wherever the training effect is near zero, and diverge only in proportion to how large that effect actually is. We read this divergence as a **measured model-partition interaction term** — the decomposition's composition-only comparison is scored against the real, α-dependent Dirichlet partition, so a model whose parameters genuinely shift with training (the MLP) picks up an interaction with that partition's structure that a two-way composition/training split cannot separate from pure training damage. We do not declare either estimate correct. The MLP's genuine training effect is reported as the range **4.6–13.5 pp**, and the ~8.5 pp gap between its bounds is reported as a limitation of the decomposition, quantified rather than left as the untested caveat it was in Section III-C.

Centralised classical accuracy is 77.50%. Global accuracy is flat across the sweep for logistic regression and declines 4.7 pp for the MLP.

### B. FedProx

FedProx was applied to the MLP only — the model family with a genuine residual for a proximal term to act on; logistic regression's residual (0.84 pp) and the VQC's (−1.25 pp, indistinguishable from zero) leave no meaningful drift for FedProx to recover, so building it for those arms would measure a null effect on top of a null effect (`docs/arm3_report.md`).

Swept across μ ∈ {0.01, 0.05, 0.1} against the same decomposition, FedProx recovers a modest, non-monotonic fraction of the MLP's 13.47 pp genuine training-heterogeneity residual: 12.72 pp at μ=0.01, 11.14 pp at μ=0.05, and 12.80 pp at μ=0.1 — a reduction of 0.67–2.33 pp (5.0–17.3% relative), never eliminating it. The reduction is not monotonic in μ: μ=0.05, the literature-recommended value for this dataset, shows the largest reduction, while μ=0.01 and μ=0.1 show smaller, similar reductions to each other. A paired significance check (same seed/fold pairs, FedProx minus FedAvg worst-client accuracy at α=0.1) finds improvements of +0.87 pp (μ=0.01, SE 1.53), +2.93 pp (μ=0.05, SE 1.91), and +2.05 pp (μ=0.1, SE 1.77) — none reaching conventional significance (~2 SE), with μ=0.05 closest.

FedProx does what it is mechanistically designed to do: client parameter divergence at the final round, α=0.1, decreases monotonically with μ (1.1088 at μ=0 down to 0.9578 at μ=0.1, a 13.6% reduction) — but this clean, monotonic reduction in the targeted mechanism does not translate into a correspondingly large or monotonic recovery of worst-client accuracy. μ was not tuned toward a preferred outcome; all three values are reported as run.

### C. Quantum arms

The quantum classifier's baseline worst-client accuracy is 64.9% at α = 100, against 69.9% for the parameter-matched MLP. Training required approximately 13,300× the wall-clock time of the MLP per run; the full sweep of 250 replicates required 8.97 hours with four-way parallelism.

Circular-mean aggregation [12] is statistically indistinguishable from arithmetic aggregation across the entire sweep, identical to four decimal places at α = 100 and α = 1.0. Inspecting 28,800 trained parameter values, the maximum magnitude is 1.7927 radians, approximately 1.35 radians short of π, with no value exceeding 0.9π and no trend with α. Circular and arithmetic means diverge only near the wraparound boundary, which this training regime does not approach. The aggregation operator is therefore not the determinant of quantum heterogeneity sensitivity at this scale.

### D. Natural partition calibration

**Revised, P-016.** The earlier α ≈ 1.5 (95% CI: 1.0–4.7) estimate (A-003) used a single statistic (total variation distance, client-weighted mean only) on a sparse 4-point Dirichlet grid with 10 seeds, bracketed and log-linearly interpolated between only the two nearest tested conditions. This pass replaces it with a fine-grained calibration: a 25-point log-spaced α grid from 0.05 to 100, 30 seeds per grid point, and **two** distance statistics — total variation (TV) distance and Jensen-Shannon (JS) divergence (base-2, bounded [0,1]) — each reported as both the **client-size-weighted mean** and the **unweighted maximum** across the four real clients, so the calibration does not rest on a single metric or ignore the worst client in favor of the average one. Figure 1 plots both statistics against the full α grid with the natural partition's value and equivalent-α estimate marked.

| statistic | natural partition's value | equivalent α (point estimate) | 95% CI |
|---|---|---|---|
| TV distance, mean | 0.1854 | 1.475 | [1.147, 1.961] |
| TV distance, max | 0.3817 | 0.991 | [0.812, 1.259] |
| JS divergence, mean | 0.0374 | 1.727 | [1.378, 2.180] |
| JS divergence, max | 0.1510 | 1.014 | [0.835, 1.270] |

All four statistics fall inside the tested grid's range — the natural partition maps cleanly onto a single Dirichlet α **region**, not any single tested condition, and all four independent estimates converge to a materially tighter range than the earlier single-statistic calibration (previously CI width 3.7; now every CI here has width ≤0.85). It does not, however, converge to one single α value: the two **mean**-based statistics (client-weighted average across clients) both center near **α ≈ 1.5–1.7**, while the two **max**-based statistics (driven by the single most-skewed client) both center near **α ≈ 1.0**, a small but consistent and reportable difference — the natural partition's *typical* client looks slightly milder (α≈1.5–1.7) than its *worst* client (α≈1.0). **Stated plainly: the natural partition corresponds to Dirichlet α in the range 1.0–1.7 depending on which statistic and which client (mean or worst) is asked, not to one exact value** — narrower and more precisely bounded than the earlier 1.0–4.7 range, but a range on principle, not a point being rounded for convenience. The value α = 0.1, commonly used to represent realistic non-IID conditions, remains more severe than any of the four estimates of the heterogeneity observed in this real archive.

**Figure 1.** `results/figs/alpha_calibration_fine.png` — TV distance and JS divergence vs. Dirichlet α (log scale), client-weighted mean and cross-client maximum, natural partition's value (dashed) and equivalent-α estimate with 95% CI (dotted line, shaded band).

### E. Client partition sizes

| condition | mean n | median n | min n | max n |
|---|---|---|---|---|
| α = 100 | 46.0 | 46.0 | 29 | 63 |
| α = 1.0 | 46.0 | 42.0 | 4 | 119 |
| α = 0.5 | 46.0 | 41.0 | 3 | 121 |
| α = 0.1 | 46.0 | 31.5 | **1** | 147 |
| natural | 46.0 | 48.0 | 18 | 73 |

Mean client test-partition size is invariant across conditions (46.0 — the same pooled test rows divided four ways, on average); only the spread widens with heterogeneity. At α = 0.1, individual client test partitions range from 1 to 147 rows, against a roughly uniform ~46 at α = 100.

**Second dataset, both client counts (P-017).**

| K | α | mean n | median n | min n | max n | zero-row cells |
|---|---|---|---|---|---|---|
| 4 | 100 | 3575.9 | 3568.0 | 3040 | 4449 | 0 / 200 |
| 4 | 1.0 | 3575.9 | 3105.5 | 585 | 10282 | 0 / 200 |
| 4 | 0.5 | 3575.9 | 2798.5 | 21 | 12008 | 0 / 200 |
| 4 | 0.1 | 3575.9 | 1077.0 | **2** | 13934 | 0 / 200 |
| 130 | 100 | 110.0 | 110.0 | 66 | 171 | 0 / 6500 |
| 130 | 1.0 | 110.0 | 80.0 | **0** | 798 | 1 / 6500 |
| 130 | 0.5 | 110.0 | 60.0 | **0** | 1063 | 1 / 6500 |
| 130 | 0.1 | — | — | — | — | *not computed — structurally infeasible, no valid draw exists under the 15-row minimum-client-size floor (P-014)* |

**Mean is again invariant within a client count** (3575.9 at K=4; 110.0 at K=130 — the same pooled test rows divided evenly, on average) — only the spread widens with heterogeneity, exactly as in the first dataset, and it widens further with client count: K=4's worst case is 2 test rows (α=0.1); K=130 produces two individual (seed, fold, client) cells with **zero** test rows at all (1 of 6,500 cells at both α=1.0 and α=0.5 — rare, and correctly excluded rather than scored, since `worst_client_acc` skips empty groups, so the reported worst-client statistic in Section V.F is a minimum over the *non-empty* clients present in that fold, not always literally all K). See Section VII for why this matters beyond a footnote — and see P-019 for a direct robustness check on the composition-share headline against this exact tail risk.

### F. Cross-dataset generalization: does the decomposition itself generalize?

The question this section answers is not "does composition dominate training on a second dataset" (Section V.A already established that for the first) but **whether the two-estimator diagnostic itself — agreement as trust, disagreement as warning — holds up as a method when applied somewhere new.** We test this on a second, independently sourced clinical dataset (Diabetes 130-US Hospitals [17]; 71,518 first-encounters-per-patient after excluding repeat visits, 24 retained features, no per-record hospital identifier in the public release so no natural-partition arm is possible) for the two classical model families only, at both 4 and 130 simulated Dirichlet clients — the latter isolating client count as a variable on its own, holding the α grid and protocol fixed. **As with Section V.A, the "Decomposition composition (share)" column below is not robust to a minimum-partition-size filter for most of these rows (Section V-G, checked directly against this exact K=130/small-partition concern). Three rows in this table (heart disease LR, and dataset 2's LR/MLP at K=4, α=100→1.0) fall below the 2pp observed-decline floor (§III-C) and are reported in absolute terms rather than as a percentage for that reason — the K=4/α=1.0 rows trigger the floor directly from their raw value, without needing the robustness sweep. The shared-test-derived columns this section's own conclusions rely on are stable for every dataset-2 row where a sweep was run.**

| Dataset | Model | K | α pair | Observed | Decomposition composition (share) | Decomposition residual | Shared-test pooled (SE) | Interaction |
|---|---|---|---|---|---|---|---|---|
| Heart Disease (1) | LR | 4 | 100→0.1 | 4.66 pp | 3.82 pp (below 2pp floor — absolute only) | 0.84 pp | −0.26 to −0.06 pp | 0.90–1.10 pp |
| Heart Disease (1) | MLP | 4 | 100→0.1 | 18.46 pp | 4.99 pp (27.0%) | 13.47 pp | 4.62–5.00 pp | 8.47–8.85 pp |
| Heart Disease (1) | VQC | 4 | 100→0.1 | 7.25 pp | 8.50 pp (117.2%) | −1.25 pp | −0.81 to 0.35 pp | −1.60 to −0.44 pp |
| Diabetes 130 (2) | LR | 4 | 100→1.0 | 0.80 pp | 0.65 pp (below 2pp floor — absolute only) | 0.16 pp | 0.14 pp (SE 0.07) | −0.08 to 0.01 pp |
| Diabetes 130 (2) | LR | 4 | 100→0.5 | 3.89 pp | 6.99 pp (179.6%) | **−3.10 pp** | 1.04 pp (SE 0.24) | −4.14 to −3.96 pp |
| Diabetes 130 (2) | LR | 4 | 100→0.1 | 16.14 pp | 21.04 pp (130.4%) | **−4.90 pp** | 3.79 pp (SE 0.25) | −8.69 to −8.43 pp |
| Diabetes 130 (2) | MLP | 4 | 100→1.0 | 0.99 pp | 0.87 pp (below 2pp floor — absolute only) | 0.12 pp | 0.22 pp (SE 0.08) | −0.15 to −0.10 pp |
| Diabetes 130 (2) | MLP | 4 | 100→0.5 | 3.25 pp | 3.92 pp (120.6%) | −0.67 pp | 0.44 pp (SE 0.12) | −1.11 to −0.95 pp |
| Diabetes 130 (2) | MLP | 4 | 100→0.1 | 15.71 pp | 13.71 pp (87.2%) | 2.00 pp | 1.38 pp (SE 0.26) | 0.62–1.21 pp |
| Diabetes 130 (2) | LR | 130 | 100→1.0 | 14.84 pp | 18.94 pp (127.7%) | **−4.11 pp** | 0.95 pp (SE 0.20) | −5.14 to −5.06 pp |
| Diabetes 130 (2) | LR | 130 | 100→0.5 | 29.52 pp | 23.81 pp (80.6%) | 5.72 pp | 4.19 pp (SE 0.16) | 1.53–2.53 pp |
| Diabetes 130 (2) | LR | 130 | 100→0.1 | — | — | — | — | *structurally infeasible — see note below* |
| Diabetes 130 (2) | MLP | 130 | 100→1.0 | 12.23 pp | 13.89 pp (113.6%) | **−1.67 pp** | 0.26 pp (SE 0.12) | −1.92 to −1.81 pp |
| Diabetes 130 (2) | MLP | 130 | 100→0.5 | 26.89 pp | 20.64 pp (76.7%) | 6.26 pp | 1.21 pp (SE 0.26) | 5.05–5.64 pp |
| Diabetes 130 (2) | MLP | 130 | 100→0.1 | — | — | — | — | *structurally infeasible — see note below* |

**K=130, α=0.1 is not omitted; it does not exist under this protocol.** Verified directly (not assumed from a failed run): every Dirichlet(0.1) draw of the second dataset's 71,518 rows into 130 client bins produces at least one client with zero rows — 0 of over 1,000 independent draws succeeded even at a floor of 1 row, let alone the 15-row minimum-client-size guard used throughout this study. This is itself consistent with the section's headline result below, not a separate nuisance: the same widening-spread mechanism that drives the composition confound also makes the partition protocol itself break down at its most extreme setting, once client count is large enough.

**Bolded residuals in the table are negative** — the decomposition claims composition *more than fully* explains the observed decline — at exactly the rows where the shared-test estimator finds a real, non-trivial positive training effect instead (every listed shared-test SE is small relative to its point estimate; several estimates are 4–26 standard errors from zero). This happens for **logistic regression**, which agreed cleanly with the shared-test estimator throughout the first dataset. **The interaction term (final column) is not a fixed property of model family: it changes sign across configurations for both LR and MLP on the second dataset, and its magnitude at several second-dataset rows (up to 8.7 pp) exceeds the entire interaction found for the first dataset's MLP (8.5 pp).** We read this as confirming Section III-C's own caveat more strongly than we expected when it was written: the decomposition's residual absorbs an interaction it cannot isolate, and that interaction is large enough, and unpredictable enough in sign, that the decomposition cannot be trusted as a standalone estimate on a dataset it has not already been checked against.

**The pre-registered client-count prediction is confirmed, on the reliable statistic — cleanly at α=0.5, where both client counts clear the 2pp floor, and directionally but not as a clean percentage-to-percentage claim at α=1.0, where K=4's decline does not.** Comparing the two client counts at the α values available to both (1.0 and 0.5), the decomposition's own share moves inconsistently — up at α=1.0, down at α=0.5, for both models — because the decomposition's own residual is unstable at these configurations (the negative-residual rows above). The **shared-test-implied** composition share (observed decline minus the shared-test estimate, as a fraction of observed decline — the statistic Section V.A already established as the more trustworthy of the two wherever the two estimators disagree) is the primary evidence here, but by §III-C's 2pp floor it is only reportable as a percentage on **both** sides of the comparison at α=0.5: MLP moves from **87–91% at K=4 to 96–98% at K=130**; LR moves from 73–78% to 86–89%. **At α=1.0, K=4's decline is too small to report as a share for either model (MLP: 0.99pp observed, ≈0.77pp of it composition in absolute terms; LR: 0.80pp observed, ≈0.66pp composition) — the comparison against K=130's clean percentages (MLP 98–99%, LR 93–94%) is directionally consistent with the same prediction (composition accounts for nearly all of a small decline at K=4, and a stated 98–99%/93–94% majority of a much larger one at K=130) but is not itself reported as a percentage-to-percentage growth claim, per the same floor.** **The α=0.5 comparison, safely reportable on both sides, is the evidence this prediction rests on; the confound is worst in exactly the many-client regime most federated learning studies actually use** — this dataset's 130-client condition is closer to typical cross-silo/cross-device federated learning study sizes than the first dataset's 4-client natural-site count was.

### G. Minimum-partition-size robustness check (P-019, corrected P-021)

**Why this check was necessary.** Section V-E's K=4 worst case is n=2 test samples, where worst-client accuracy can only take three values (0%, 50%, 100%). Every composition-share number reported above through Section V.F is built on the decomposition's own worst-client statistic — a minimum over exactly these per-client partitions — so a headline resting on it needs a direct robustness check against the tail, not only a caveat noting the tail exists. We re-analysed already-computed predictions (no retraining — see below) excluding any (seed, fold, α, client) cell below a minimum test-partition-size threshold from the worst-client minimum, swept across five thresholds (0, 5, 10, 15, 20 samples) rather than committing to one arbitrary cutoff.

**No retraining.** Per-client accuracy and partition size for every replicate already existed in committed results (`results/diagnostic_results.csv`, `results/arm4_diagnostic_results.csv`, `results/vqc_composition_partial/*.json`) or was obtained by deterministic reproduction of the already-frozen training recipe (same seeds, same protocol, same models — consistent with this project's established "deterministic reproduction is not retraining" precedent, Section V.A). `scripts/partition_size_robustness.py`; raw output in `results/partition_size_robustness.json`.

**An initial version of this section asserted the shared-test-implied share is "structurally immune" to this instability. That assertion was wrong and has been corrected here, not softened.** The share is `(observed − shared-test training effect) / observed`; the shared-test training effect is threshold-invariant (it comes from pooled accuracy on the full held-out set, not a per-client minimum), but the **observed decline in the denominator is the exact same worst-client-minimum statistic that destabilised the raw decomposition share** — a stable numerator term does not rescue an unstable denominator. The correct claim is narrower: the shared-test-implied share is *more resistant* than the raw decomposition share in most configurations, not immune, and it fails outright for one.

**Raw observed worst-client decline, by threshold (the shared denominator):**

| Dataset | Model | K | Observed, t=0 | t=5 | t=10 | t=15 | t=20 |
|---|---|---|---|---|---|---|---|
| Heart Disease (1) | LR | 4 | 4.67 pp | 3.99 pp | 2.70 pp | 1.40 pp | **0.43 pp** |
| Heart Disease (1) | MLP | 4 | 18.46 pp | 17.84 pp | 15.84 pp | 15.52 pp | 14.22 pp |
| Heart Disease (1) | VQC | 4 | 7.25 pp | 7.25 pp | 5.77 pp | 4.00 pp | 2.37 pp |
| Diabetes 130 (2) | LR | 4 | 16.14 pp | 14.97 pp | 13.54 pp | 13.30 pp | 13.30 pp |
| Diabetes 130 (2) | MLP | 4 | 15.71 pp | 15.00 pp | 14.03 pp | 13.19 pp | 13.19 pp |
| Diabetes 130 (2) | LR | 130 | 29.52 pp | 27.93 pp | 27.49 pp | 25.00 pp | 24.80 pp |
| Diabetes 130 (2) | MLP | 130 | 26.89 pp | 23.17 pp | 21.63 pp | 18.35 pp | 17.40 pp |

**Decomposition share vs. shared-test-implied share, by threshold:**

| Dataset | Model | K | Decomp. share t=0→t=20 | Shared-test-implied share t=0→t=20 | Interpretable? |
|---|---|---|---|---|---|
| Heart Disease (1) | LR | 4 | 82.0% → **−174.6%** | 105.6% → **160.3%** | **No** — both estimators destabilise |
| Heart Disease (1) | MLP | 4 | 27.1% → 2.5% | 75.0% → 67.5% | Yes — shared-test-implied stable (−7.5 pts) |
| Heart Disease (1) | VQC | 4 | 117.1% → 151.9% | 95.2% → 85.3% | Yes — shared-test-implied stable (−9.9 pts) |
| Diabetes 130 (2) | LR | 4 | 130.4% → 135.6% | 76.5% → 71.5% | Yes — both estimators stable |
| Diabetes 130 (2) | MLP | 4 | 87.2% → 73.9% | 91.2% → 89.5% | Yes — shared-test-implied stable (−1.7 pts) |
| Diabetes 130 (2) | LR | 130 | 80.6% → 57.4% | 85.8% → 83.1% | Yes — shared-test-implied stable (−2.7 pts) |
| Diabetes 130 (2) | MLP | 130 | 76.7% → 48.8% | 95.5% → 93.0% | Yes — shared-test-implied stable (−2.5 pts) |

**Verdict, stated plainly per instruction, with each configuration named rather than a single blanket claim: the shared-test-implied share destabilises for exactly one configuration — heart disease LR — and it is the one configuration where this mattered most to check, since it is the archive's convex "safe" model. Every other configuration's shared-test-implied share is stable, moving by 10 percentage points or less across the full sweep, in contrast to the raw decomposition share's movement of 20 to over 250 points on the same five configurations that were already flagged as non-robust.**

**Why heart disease LR fails and the others do not.** Algebraically, share = 100 − 100·(TE/observed), where TE is the fixed shared-test training effect. As the threshold rises and the observed decline shrinks toward zero, this ratio is bounded only if TE is exactly zero — which never happens exactly. Heart disease LR's observed decline was already the smallest of any configuration at t=0 (4.67 pp, only ~5× this study's own 0.3–1.0 pp noise floor) and falls fastest in *relative* terms under filtering (a 91% reduction, to 0.43 pp by t=20), while its shared-test training effect (−0.26 pp) is fixed but nonzero — so the ratio is pushed further from 100% as the denominator vanishes, reaching 160.3% at t=20. Every other configuration's observed decline either starts larger or shrinks less aggressively in relative terms (dataset 2's K=4 rows plateau after t=10 because the threshold stops excluding additional clients past that point — Section V-E's K=4 minimum is 2 rows for LR, so a threshold of 15 or 20 removes the same cells as 10 once the tail is exhausted), which keeps their ratios from being driven by a near-vanishing denominator within the range tested here. **This is a property of the ratio near a near-zero training effect, not a property of either estimator specifically** — it would eventually affect any of the currently-stable configurations too, at a large enough threshold; this sweep only tested up to 20 samples and does not rule that out.

**Consequence for the headline, resolved (P-021 → this pass): the 101–160% range was never a finding about LR — it was the signature of an unstable ratio, and is no longer reported as a percentage at all.** §III-C formalizes the general rule this case motivated: composition share is reported only where the observed decline is at least 2 percentage points, both at the unfiltered estimate and across any robustness sweep run for that configuration; heart disease LR's observed decline (4.66pp) crosses below that floor at threshold=15 under the sweep, so it is now reported in absolute terms throughout the paper (Abstract, §I, §V-A): **logistic regression shows no measurable training effect; its observed decline is small (4.66pp) and almost entirely composition (≈4.7pp of it, by either estimator).** Applying the same floor mechanically (not as a special case) also caught two dataset-2 rows whose raw, unfiltered observed decline is already below 2pp without needing a sweep — LR and MLP at K=4, α=100→1.0 (§V.F) — and, downstream of that, the client-count-prediction claim (§V.F, echoed in the Abstract/§I/§VIII) has been restated using the α=0.5 comparison, the only one where both client counts clear the floor on both sides. **The paper's cross-model claim — that composition dominates the observed decline for all three families — is not threatened by any of this**: MLP and VQC (dataset 1) and every dataset-2 configuration above the floor remain stable, all still showing composition shares comfortably ≥67% (or, for LR, an absolute composition of ≈4.7pp against a near-zero training effect, which is the same qualitative claim stated without a fragile percentage).

---

## VI. Discussion

**Convexity does not fully explain the pattern.** The convex model shows a residual near zero and the non-convex MLP a substantial one, consistent with the expectation that parameter averaging approximates the pooled optimum for convex objectives largely independently of partitioning. However, the variational quantum classifier is also non-convex and behaves like the convex model on this axis. We do not resolve this; candidate explanations include the circuit's limited effective capacity placing it in a near-linear regime, but we have not tested them.

**Our results do not contradict prior reports.** Hsu et al. [7] report severe global degradation for convolutional networks on ten-class vision data; our convex model on a six-feature binary tabular task shows none. These are different regimes, and the theoretical result that heterogeneity slows but does not prevent convergence for strongly convex objectives is consistent with our observation.

**Implication for practice.** Studies sweeping heterogeneity under client-local evaluation should report the decomposition, or evaluate on a shared held-out set. The procedure requires no additional training — only re-evaluation of an already-trained model.

---

## VII. Limitations

**Scale.** 920 records, four clients, six features, one binary task. Absolute magnitudes are dataset-specific; only the method transfers.

**Sample size in the tail — worse on the second dataset, and this needs to be read before any worst-client number in this paper, not discovered afterward.** At extreme skew, individual client test partitions are small, and worst-client accuracy is correspondingly noisy. On the first archive, the smallest observed test partition across all replicates at α = 0.1 is **n = 1** (median 31.5, mean 46 — Section V-E). **On the second dataset, it is worse in two distinct ways, at both client counts tested:** at K=4, α=0.1, the smallest test partition is **n = 2**; at K=130 — the client count used for the headline confound-scaling result in Section V.F — **1 of 6,500 (seed, fold, client) cells has zero test rows, at both α=1.0 and α=0.5** (Section V-E). A single- or two-example client's accuracy is necessarily 0%, 50%, or 100%, which inflates variance on individual worst-client measurements sharply at these conditions; a zero-row cell has no accuracy at all and is excluded from the minimum rather than counted as a failure, meaning the reported "worst client" at K=130 is occasionally a minimum over 129 clients, not literally 130, in the affected (seed, fold) replicates. **This directly touches Section V.F's headline claim**, which depends on worst-client and pooled statistics computed at exactly this K=130, high-skew regime — the composition-share-grows-with-K finding is real and measured on real partitions (Section V.F, P-014), and P-019 re-derives it under a minimum-partition-size filter as a direct robustness check rather than leaving the tail-noise concern as an unresolved caveat.

**The decomposition, used alone, cannot be trusted to identify which model families are safe to report without a shared-test check.** Section III-C's caveat that it is first-order and does not isolate interaction terms is not merely theoretical: Section V measures that interaction directly, on two datasets, and finds it is a property of the specific (model, partition, α) configuration rather than of model family. On the first archive, only the non-convex model (MLP) diverged from the shared-test estimate, consistent with a "convex models are safe" rule of thumb; on the second, the convex model (logistic regression) itself diverges at 3 of 5 measured configurations, in both directions and by up to 8.7 percentage points — larger than the first archive's entire MLP interaction. No property of a model checked in advance (convexity, parameter count, which archive performed well on it before) reliably predicted this. This is a bounded limitation of the decomposition as a standalone estimator, not a limitation of the underlying measurement question: the shared-test estimate, which does not share this failure mode, remains available and is what we recommend as the primary evaluation (Section VIII).

**Simulation fidelity.** All quantum results are noiseless and therefore characterise an upper bound; physical hardware would perform no better. Wall-clock timings measure classical simulation cost and are not hardware-transferable.

**Feature restriction.** Cross-site availability excluded `ca`, `thal` and `chol`, all clinically relevant; the cost of this exclusion on classification accuracy was not quantified, since this paper's contribution is an evaluation protocol, not a model comparison, and a feature-ablation study is outside that scope.

**Dataset age.** Collected 1988; diagnostic practice has since changed.

**Site composition.** The Switzerland client combines two institutions.

**Literature coverage.** Our survey is targeted rather than systematic. The gap is stated as not found rather than as absent.

---

## VIII. Conclusion

Client-level degradation under swept heterogeneity conflates training damage with evaluation composition when test data is partitioned by the same client assignment used for training. **We set out to fix this with a decomposition, validated against an independent shared-test estimate. We are reporting instead that the fix does not stand alone.** On our first, four-site clinical archive, the two estimators agreed for a convex classical model and a variational quantum classifier — both near zero genuine training effect — and disagreed only for a small non-convex network, by roughly 8.5 percentage points. That pattern was consistent with a comfortable story: disagreement tracks non-convexity, so a practitioner using a convex model could trust the decomposition alone. **On a second, independently sourced clinical dataset, that story does not survive contact with more data.** The convex model itself — logistic regression, the "safe" family on dataset one — disagrees with the shared-test estimator at 3 of 5 measured configurations on dataset two, by up to 8.7 percentage points, with the decomposition's own residual going *negative* at points where the shared-test estimator finds a real, many-standard-error-from-zero positive training effect. The interaction between model, partition, and heterogeneity level is not predictable from model family alone, and no rule we tested lets a practitioner know in advance whether their model is one the decomposition can be trusted for. A pre-registered prediction that this problem should worsen with client count is confirmed on the more reliable of the two statistics, at the heterogeneity level (α=0.5) where both client counts clear our own 2pp reporting floor: the shared-test-implied composition share for the multilayer perceptron rises from 87–91% at 4 clients to 96–98% at 130 — the confound is largest in exactly the many-client regime most federated learning studies use, not a small-scale artifact of our own experimental convenience. We do not recommend the decomposition as a standalone correction. We recommend the shared-test protocol as the primary evaluation — it is the more stable estimator across both datasets we tested, and it is already standard practice in evaluations such as [6] — and, where a heterogeneity sweep under client-local evaluation cannot be avoided, we recommend reporting both estimators together and treating their divergence as a warning that an unmeasured interaction is present, not as a discrepancy to be resolved by picking whichever number looks more convenient. We further report that the first archive's natural inter-institutional heterogeneity is milder than the synthetic setting commonly used to represent it, and that the quantum classifier's four-order-of-magnitude cost purchased no measurable heterogeneity robustness.

---

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. Agüera y Arcas, "Communication-efficient learning of deep networks from decentralized data," in *Proc. AISTATS*, PMLR, 2017, pp. 1273–1282.

[2] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith, "Federated optimization in heterogeneous networks," in *Proc. MLSys*, vol. 2, 2020, pp. 429–450.

[3] T. Li, M. Sanjabi, A. Beirami, and V. Smith, "Fair resource allocation in federated learning," in *Proc. ICLR*, 2020.

[4] M. Mohri, G. Sivek, and A. T. Suresh, "Agnostic federated learning," in *Proc. ICML*, PMLR vol. 97, 2019, pp. 4615–4625.

[5] T. Li, S. Hu, A. Beirami, and V. Smith, "Ditto: Fair and robust federated learning through personalization," in *Proc. ICML*, PMLR vol. 139, 2021, pp. 6357–6368.

[6] Q. Li, Y. Diao, Q. Chen, and B. He, "Federated learning on non-IID data silos: An experimental study," in *Proc. IEEE ICDE*, 2022, pp. 965–978.

[7] T.-M. H. Hsu, H. Qi, and M. Brown, "Measuring the effects of non-identical data distribution for federated visual classification," NeurIPS Workshop on Federated Learning, 2019, arXiv:1909.06335.

[8] A. Janosi, W. Steinbrunn, M. Pfisterer, and R. Detrano, "Heart Disease," UCI Machine Learning Repository, 1989, doi: 10.24432/C52P4X. Licensed CC BY 4.0.

[9] R. Detrano *et al.*, "International application of a new probability algorithm for the diagnosis of coronary artery disease," 1989.

[10] A. S. Bhatia, M. Saggi, and S. Kais, "Communication-efficient quantum federated learning optimization for multi-center healthcare data," in *Proc. IEEE-EMBS BHI*, 2024.

[11] A. S. Bhatia and S. Kais, "Enhancing quantum federated learning with Fisher information-based optimization," arXiv:2507.17580, 2025.

[12] S. I. Nanayakkara and S. R. Pokhrel, "A2G-QFL: Adaptive aggregation with two gains in quantum federated learning," arXiv:2512.03363, 2025.

[13] R. M. Wichmann, M. A. R. Bigoto, and A. D. P. Chiavegatto Filho, "Federated learning for COVID-19 mortality prediction in a multicentric sample of 21 hospitals," *PLOS Comput. Biol.*, vol. 21, no. 11, e1013695, 2025.

[14] F. Asad, J. S. Khan, M. Tariq, S. Munir, and M. A. Khan, "Federated proximal optimization for privacy-preserving heart disease prediction: A controlled simulation study on non-IID clinical data," arXiv:2601.17183, 2026.

[15] C. Xie, D.-A. Huang, W. Chu, D. Xu, C. Xiao, B. Li, and A. Anandkumar, "PerAda: Parameter-efficient federated learning personalization with generalization guarantees," in *Proc. CVPR*, 2024.

[16] D. Chen, D. Gao, W. Kuang, Y. Li, and B. Ding, "pFL-Bench: A comprehensive benchmark for personalized federated learning," in *Proc. 36th Conf. Neural Inf. Process. Syst. (NeurIPS), Datasets and Benchmarks Track*, 2022.

[17] J. Clore, K. Cios, J. DeShazo, and B. Strack, "Diabetes 130-US Hospitals for Years 1999–2008," UCI Machine Learning Repository, 2014, doi: 10.24432/C5230J. Licensed CC BY 4.0. Associated publication: B. Strack *et al.*, "Impact of HbA1c measurement on hospital readmission rates: analysis of 70,000 clinical database patient records," *BioMed Research International*, vol. 2014, Article ID 781670, 2014.

---

## Flags for Author Review

**Flag 1 — α-calibration number conflicts with language used elsewhere in this project; now superseded twice.** §V-D originally reported α ≈ 1.5 (95% CI 1.0–4.7) via TV distance alone on a 4-point grid (`scripts/alpha_calibration.py`, A-003) — itself milder than the informal "α ≈ 0.5–1.0" language baked into `docs/decisions.md` (D-037), `scripts/plots.py`'s legends, and other project docs, for the reason explained below (label-skew-only vs. downstream-metric-matching are different questions). **P-016 (this pass) superseded the A-003 number with a fine-grid, dual-statistic calibration** (`scripts/alpha_calibration_fine.py`, 25-point grid, 30 seeds/point, TV + JS, mean + max) — see the revised §V-D above. The new estimates (α ≈ 1.0–1.7 depending on statistic) are consistent in direction with A-003's original 1.5 but meaningfully tighter (every new CI has width ≤0.85 vs. A-003's 3.7), and explicitly report a mean-vs-max split A-003 did not have the resolution to see. **The D-037 "~0.5–1.0" reconciliation this flag originally asked for is still open** — it remains a genuinely different question (feature-distribution/downstream-metric matching, not label-skew distance) and was not the scope of either the original or this pass's calibration work; still needs an explicit reconciling sentence in the paper before submission.

**Flag 2 — RESOLVED, per instruction.** The unverifiable "four of six predictions contradicted" claim is removed. §I now cites the two predictions this project actually logged as pre-registered before their outcomes were measured (D-048/D-049, P-009/P-014) and states plainly that both were confirmed, not refuted — the opposite of the original sentence's framing. No other predictions in `docs/decisions.md` are logged with the explicit "predicted before measuring" structure these two have.

**Flag 3 — left open, assigned to Ayuvi, who owns the literature sections (per instruction, not touched this pass).** [12] A2G-QFL's description is simplified: their actual contribution (confirmed via their own abstract) is a *dual*-gain framework — a geometry gain (the rotation-averaging point this paper cites) *and* a QoS gain based on latency/fidelity/instability for client-importance weighting. The draft's sentence describes only the geometry half. Not incorrect, but incomplete — worth a clause acknowledging the QoS dimension exists, or being explicit that this paper adopts only the geometric half as an ablation baseline (which it does, correctly, elsewhere).

**Flag 4 — RESOLVED, per instruction.** Verified [13] (Wichmann et al.) against its actual source: a COVID-19 mortality prediction study across 21 Brazilian hospitals — a real, correctly-formatted citation, just not about "this dataset family" as the original sentence implied. §II-E's sentence rewritten to attribute [13] to multi-institutional clinical FL generally and [14] specifically to this dataset family, matching what each citation actually is rather than lumping them together. Neither citation was removed from the reference list — [13] remains a legitimate reference, now correctly scoped.

**Flag 5 — VQC row in the §V-A table had a real numerical error, now corrected.** v1 read 7.30 pp / 8.55 pp; the source of record (`docs/decisions.md` D-049) and an independent re-derivation from committed code this session (`scripts/composition_summary.py`, which reproduces LR/MLP to within floating-point noise and VQC exactly, since it reads the same static per-replicate files) both give 7.25 pp / 8.50 pp. The error was easy to miss because the residual (−1.25 pp) comes out identical either way — the two errors canceled in the subtraction. Worth double-checking the other two rows against source data before final submission too, though LR and MLP both checked out exactly.

**Flag 6 — RESOLVED, per instruction: cut, not merely stated as open.** The feature-exclusion sensitivity analysis (tracked as D-014, referenced across sessions, never built) is no longer framed as pending work for this paper at all — this paper's contribution is an evaluation protocol, not a model comparison, so a feature-ablation study is out of scope rather than an open item. §VII's "Feature restriction" bullet now states in one sentence that the cost was not quantified and why, with no placeholder implying a number is still coming.

**Flag 7 — RESOLVED this pass.** §II-B rewritten: establishes client-level disparity as prior art, states that our decomposition suggests a substantial composition share across all three model families (not "two of three"), and reframes the q-FFL exposure as structural and identified, not measured as an error on their part.

**Flag 8 — RESOLVED this pass, for §IV.A and §VII specifically.** §IV.A now carries a second-dataset paragraph (provenance, first-encounter-per-patient filtering 101,766→71,518, the near-zero-variance filter, the 24-feature retained set, the class-weighting/balanced-accuracy departure, and the absence of a hospital identifier). §VII's decomposition-limitation bullet now states the (model, partition, α)-not-model-family finding directly, scoped explicitly as a limitation of the decomposition as a standalone estimator rather than of the underlying measurement question. §V.F itself (the table and its surrounding prose) was not touched this pass — it was already consistent with this framing when P-015 wrote it.

**Tone check across the full draft (requested this pass, per instruction):** scanned for language describing the work itself as having failed, as opposed to negative/null results reported as findings in the sections where they belong. Found the existing draft (Abstract, Contributions, Conclusion, §V.A, §V.F, §VI) already consistent with this — the deliberate framing sentence in the Abstract/Conclusion ("we did not end up with a validated correction... reported as the finding, not as a setback") states a null result plainly without characterizing the work as unsuccessful, and P-015's own note confirms this was intentional, not something to soften further. The three sections edited this pass (§II-B, §IV.A, §VII) were written to match: §II-B reports the composition share as something the decomposition "suggests," not as a verdict; §VII scopes its strongest statement explicitly to "the decomposition as a standalone estimator," not to the paper's overall contribution. No instances of the work being described as a failure were found or introduced.

**Flags NOT raised (checked and found correct):** the "28,800 trained parameter values" and "1.7927 radians / 1.35 short of π / no value exceeding 0.9π" claims in §V-C were verified directly against `results/angle_capture/*.npz` (20 files × 20 rounds × 4 clients × 18 params = 28,800 exactly; max computed magnitude 1.79269, π − max = 1.34890, no value exceeds 0.9π) — these are correct as written. The "13,300×" and "8.97 hours" figures in §V-C match `docs/arm4_report.md` exactly. LR and MLP rows in the §V-A table match source data exactly.

**Note on this file's location (P-020):** this draft was moved into the repository from a shared local `Downloads/paper_draft_v2.md` copy on 2026-08-28, after both Prithvi's and Ayuvi's sessions independently edited that shared file in the same window (resolved cleanly this time, since the edits were non-overlapping — see A-004 and P-015 through P-018 in `docs/decisions.md`). This file is now the canonical, version-controlled draft; edit it here, not in Downloads.
