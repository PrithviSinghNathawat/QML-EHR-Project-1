# Federated Learning Fairness / Worst-Client Disparity — Literature

Compiled 2026-08-20 (Ayuvi). Purpose: our diagnostic (`docs/diagnostic_report.md`)
found that global accuracy stays flat under non-IID Dirichlet partitioning while
worst-client accuracy declines monotonically as α falls. This is not a novel
observation — the FL fairness literature has established both the phenomenon and
several ways of measuring/correcting it. We cite this work rather than presenting
worst-client degradation as our discovery; our own contribution is comparing how a
VQC's worst-client degradation compares to a classical model's under identical
conditions, not the existence of the degradation itself.

---

## [1] Mohri, Sivek & Suresh — Agnostic Federated Learning

M. Mohri, G. Sivek, and A. T. Suresh, "Agnostic federated learning," in *Proc.
36th Int. Conf. Mach. Learn. (ICML)*, Long Beach, CA, USA, 2019, pp. 4615–4625.

**What it established:** The standard FedAvg objective — minimizing loss
averaged over the pooled/weighted client distribution — has no guarantee about
the loss on any individual client or sub-population. It proposes optimizing for
the *worst-case* mixture of client distributions instead (a minimax objective),
and shows empirically that models trained to minimize average loss can perform
poorly on specific clients even when the average looks fine.

**Relation to our finding:** This is the foundational reason our result is
expected, not surprising: FedAvg was never optimizing for the metric we later
checked (worst-client accuracy). It gives the theoretical vocabulary — "agnostic"
average-case objective vs. worst-case guarantee — for why a flat global accuracy
number tells us nothing about the worst client.

**Flag:** Does not itself report a worst-client-accuracy-vs-heterogeneity sweep
like ours; it's the conceptual ancestor of the metric distinction, not a direct
replication.

---

## [2] Li, Sanjabi, Beirami & Smith — Fair Resource Allocation in Federated
Learning (q-FFL)

T. Li, M. Sanjabi, A. Beirami, and V. Smith, "Fair resource allocation in
federated learning," in *Proc. Int. Conf. Learn. Represent. (ICLR)*, Addis
Ababa, Ethiopia, 2020.

**What it established:** Directly measures the *variance* of test accuracy
across devices under vanilla FedAvg and shows it is large and uneven — some
clients get much worse accuracy than others even at a fixed global average. It
proposes q-FFL, which reweights the objective toward high-loss (poorly
performing) clients, and shows this narrows the accuracy spread with a small
cost to average accuracy.

**Relation to our finding:** This is the closest match to our *measurement*,
not just our reasoning — they explicitly report accuracy variance across
clients under vanilla FedAvg as the problem worth fixing. Our diagnostic's
worst-client-vs-global gap is the same phenomenon, measured across a Dirichlet α
sweep rather than a fixed non-IID split.

**Flag:** They report the general disparity phenomenon under FedAvg but do not
sweep a controlled heterogeneity parameter (α) the way we do, so our α-sweep
curve itself is not already published by this paper — the qualitative
phenomenon is, though.

---

## [3] Li, Sahu, Zaheer, Sanjabi, Talwalkar & Smith — Federated Optimization in
Heterogeneous Networks (FedProx)

T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith,
"Federated optimization in heterogeneous networks," in *Proc. Mach. Learn.
Syst. (MLSys)*, 2020.

**What it established:** Introduces the proximal term used in our own Arm 3.
Directly relevant beyond fairness framing: shows FedAvg's convergence
degrades under statistical heterogeneity (non-IID, unbalanced client data),
and that a proximal term restraining local updates from drifting too far from
the global model improves both convergence stability and — relevant here —
robustness for the worst-behaved clients under heterogeneous data.

**Relation to our finding:** Establishes the mechanism side of our story:
client drift under heterogeneity is the thing FedProx was built to control, and
our own client-parameter-divergence metric (Section 2 of the diagnostic report)
is measuring exactly that drift. This paper is why we expect Arm 3 (FedProx) to
help worst-client accuracy at low α relative to Arm 2 (FedAvg) — it's the source
of that validation gate in `CLAUDE.md`.

**Flag:** Not primarily a fairness paper — worst-client accuracy is not its
headline metric — but it's the direct precedent for both our Arm 3 design and
our divergence metric.

---

## [4] Liu — FedGA: A Fair Federated Learning Framework Based on the Gini
Coefficient

S. Liu, "FedGA: A fair federated learning framework based on the Gini
coefficient," *arXiv:2507.12983*, Jul. 2025.

**What it established:** Uses bottom-decile ("worst-client-region") accuracy as
an explicit fairness metric — the same quantity we use, generalized from a
single worst client to the worst 10% — and proposes a Gini-coefficient-based
reweighting scheme to raise it. Reports that vanilla FedAvg/FedProx baselines
leave a substantial gap between mean accuracy and bottom-decile accuracy under
non-IID splits.

**Relation to our finding:** A recent, direct confirmation that worst-region
accuracy (not just the mean) is the right lens for measuring FL fairness damage
under heterogeneity, and that FedAvg does not close this gap on its own — same
qualitative pattern we observe (FedAvg leaves worst clients behind; global/mean
metrics understate it).

**Flag:** Reports the mean-vs-worst-region gap phenomenon we found, though as a
motivation for a new fairness algorithm rather than as a controlled α-sweep
diagnostic. Worth citing precisely because it's recent (2025) evidence the
phenomenon is still an open, active problem, not settled by earlier work.

---

## [5] Naseer & Shoaib — When More Parameters Hurt: Foundation Model Priors
Amplify Worst-Client Disparity Under Extreme Federated Heterogeneity

K. Naseer and U. Shoaib, "When more parameters hurt: Foundation model priors
amplify worst-client disparity under extreme federated heterogeneity,"
*arXiv:2605.08992*, 2026.

**What it established:** Runs a controlled non-IID label-skew sweep (explicitly
parameterized, comparable in spirit to a Dirichlet-α design) comparing a small
model (TextCNN) against a large pretrained model (DistilBERT+LoRA) under
FedAvg. Reports worst-client accuracy gaps directly as the headline metric:
50.1% (large model) vs. 32.2% (small model) under extreme label skew, with the
gap nearly closing under moderate heterogeneity (α ≥ 0.5).

**Explicit flag — this paper already reports our core finding pattern.** It is
the closest published match we found to our own result shape: global/mean
accuracy insulated from damage that concentrates in worst-client accuracy,
worsening sharply as heterogeneity increases, moderate at α ≥ 0.5 and severe at
extreme skew. We should **not** present "worst-client accuracy degrades while
global accuracy stays flat under increasing non-IID skew" as a novel
observation of this project — it is established, including with an explicit
α-style sweep, by this 2026 paper on a different domain (text classification,
not EHR tabular data). Our contribution has to be framed as: (a) confirming
this pattern on a healthcare/EHR tabular setting rather than text, and (b)
extending the comparison to a variational quantum classifier against classical
models, which this paper does not touch at all (it compares model *capacity*
within classical architectures, not classical-vs-quantum).

---

## Summary for the paper

For `paper/02_related_work.md`: cite [1] and [2] for the general
average-vs-worst-case fairness framing, [3] for FedProx as our Arm 3 baseline
and the theoretical grounding for our divergence metric, and [4]/[5] as recent
(2025–2026) direct evidence that the worst-client/global-accuracy gap under
non-IID skew is an active, unresolved phenomenon — with [5] flagged explicitly
as reporting the same qualitative pattern we found, so our methodology section
must state our contribution as extending this to EHR data and to a
quantum-classical comparison, not as discovering the gap itself.
