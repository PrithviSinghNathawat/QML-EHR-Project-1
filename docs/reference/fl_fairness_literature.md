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
cost to average accuracy. **Appendix Table 10** ("Effects of data heterogeneity
and the number of devices on unfairness") makes this quantitative: under FedAvg
(q=0), across their synthetic heterogeneity levels (IID -> (1,1) -> (2,2),
100 devices), Average accuracy declines 89.2% -> 83.0% -> 82.6% (a real 6.6pp
drop, not flat) while Worst-10% accuracy collapses 70.9% -> 36.8% -> 25.5%
(44.7pp).

**Relation to our finding, precisely:** the *disparity* between average and
worst-client accuracy under FedAvg is directly established here, quantitatively,
across a controlled heterogeneity sweep — genuine prior art for that part of
our result. **What is not established here:** (a) our specific observation
that *global* accuracy stays essentially flat (LR: 0.7625 to 0.7651 across our
full α sweep, within noise) — their average accuracy actually moves a real
6.6pp, so "flat" is our finding, not theirs, even though "much smaller than the
worst-client drop" is shared; (b) any convexity/model-capacity contrast — their
synthetic dataset and linear/softmax model give no convex-vs-non-convex
comparison axis, so our LR-vs-MLP convexity result (D-028) has no counterpart
here; (c) their device counts (50, 100) are an order of magnitude above our 4
clients, and their own table shows fewer devices -> *more* uniform accuracy
(less disparity) — so extrapolating their trend down to n=4 is not free and
could go either direction, not automatically "worse."

**Flag:** the disparity phenomenon (average vs. worst-client gap, under
FedAvg, across a controlled heterogeneity sweep) is established prior art
here, specifically and quantitatively. The flatness of the global metric
specifically, the convexity contrast, and the low-client-count regime are not
covered by this table and remain ours to establish. (Corrected 2026-08-20 — an
earlier version of this entry read as though the whole pattern, not just the
disparity, was already published here; see D-033.)

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
gap nearly closing under moderate heterogeneity (α ≥ 0.5). **Checked their
Table 3 directly for the global/mean-accuracy trend (2026-08-20, corrected —
see D-033): it is not flat.** TextCNN's average accuracy runs 86.6% (α=0.1) to
97.8% (α=5.0), an 11.2pp range; DistilBERT+LoRA runs 80.8% to 93.6%/91.3%, a
comparably wide range. Both models' *average* accuracy is clearly
heterogeneity-sensitive here, not insulated the way our LR's global accuracy
is (0.7625 to 0.7651 across our entire α sweep).

**Corrected relation to our finding:** the worst-client-accuracy-gap-under-
label-skew phenomenon is real and reported here. But the specific pattern we
find — worst-client accuracy collapsing *while global accuracy stays flat* —
is not what this paper shows: their global accuracy moves substantially with
heterogeneity too, just less than worst-client accuracy does. So this is
evidence that worst-client damage exceeds global damage under skew (same
general shape as q-FFL's Table 10 — see entry [2]), not evidence that global
accuracy specifically stays flat. Citing this for the former, not the latter.

**Flag:** an earlier version of this entry claimed this paper shows "global
accuracy insulated from damage," which overstated what their own Table 3
reports — corrected here after directly checking the actual average-accuracy
numbers rather than relying on the paper's own worst-client-focused framing.
This paper remains useful as evidence the worst-client-gap phenomenon
generalizes to a different domain (text classification) and a different
heterogeneity axis (model capacity, not just Dirichlet α), but not as a second
instance of our specific "global flat, worst-client collapses" shape.

---

## [6] Li, Hu, Beirami & Smith — Ditto: Fair and Robust Federated Learning
Through Personalization

T. Li, S. Hu, A. Beirami, and V. Smith, "Ditto: Fair and robust federated
learning through personalization," in *Proc. 38th Int. Conf. Mach. Learn.
(ICML)*, 2021.

**What it established:** Verified directly (title, authors, venue, abstract),
2026-08-20. Proposes Ditto, a personalization-based method that pursues
fairness (uniform accuracy across clients) and robustness (to poisoning/
adversarial clients) simultaneously, rather than trading one for the other as
prior methods (including q-FFL) tend to. Each client keeps a personalized
model regularized toward the global model, rather than the server reweighting
its objective toward poorly-performing clients (q-FFL's approach).

**Relation to our finding:** A third distinct response to the same underlying
problem q-FFL and Mohri et al. diagnose — client-level accuracy disparity
under FedAvg. Establishes that server-side reweighting (q-FFL) is not the only
fix; personalization is a competing family of solutions. We do not evaluate
personalization in this project (out of scope, per `CLAUDE.md`'s no-scope-
additions guardrail), so this is cited as evidence the disparity problem is an
active area with multiple proposed fixes, not as a method we compare against.

**Flag:** Verified only at the level of title/authors/venue/abstract, not at
the same table-level depth as q-FFL's Table 10 or NIID-Bench's Table III —
appropriate here since Ditto is cited for "this problem has multiple proposed
fixes," not for a specific quantitative claim about our own result shape.

---

## [7] Hsu, Qi & Brown — Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification

T-M. H. Hsu, H. Qi, and M. Brown, "Measuring the effects of non-identical
data distribution for federated visual classification," *arXiv:1909.06335*,
2019.

**What it established:** Introduces Dirichlet(α)-based synthetic label-skew
partitioning as a controllable way to simulate non-IID client data for
federated learning experiments — the specific partitioning method this
project uses (`CLAUDE.md`, `scripts/partitioner.py`). Reports that CNN
classification accuracy degrades as the Dirichlet concentration decreases
(more skew); confirmed directly, 2026-08-20: their most-skewed CIFAR-10
setting shows baseline (non-mitigated) accuracy around 30.1%, well below
their more-IID settings.

**Relation to our finding:** This is the **mandatory attribution** for our
independent variable (α) — every Dirichlet-α sweep in this project's design
traces back to this paper's method, not something we invented. It is also a
useful contrast, not just a source: their CNNs (non-convex) show clear
*global*-accuracy degradation under their heterogeneity sweep, which is the
opposite of what our convex model (LR) shows (global accuracy flat). This
lines up with, and lends outside support to, our own convexity finding
(D-028) — a non-convex image classifier degrading globally under Dirichlet
skew, versus our convex tabular classifier not degrading globally under the
same partitioning method.

**Flag:** Verified via abstract + reported CIFAR-10 numbers, not the full
per-α degradation curve — sufficient for the two claims above (source
attribution; CNNs show global degradation where our LR does not), not
sufficient to claim we've replicated their exact numbers.

---

## [8] Li, Diao, Chen & He — Federated Learning on Non-IID Data Silos: An
Experimental Study (NIID-Bench)

Q. Li, Y. Diao, Q. Chen, and B. He, "Federated learning on non-IID data
silos: An experimental study," in *Proc. IEEE 38th Int. Conf. Data Eng.
(ICDE)*, 2022.

**What it established:** A characterization-study benchmark (closest in spirit
to this project's own goals) comparing FedAvg, FedProx, and other algorithms
across several non-IID partitioning strategies including Dirichlet-α. Checked
directly against both the GitHub repo (Xtra-Computing/NIID-Bench) and the
paper's full text, 2026-08-20 (see D-032): **does not report per-client or
worst-client accuracy anywhere** — only aggregate/global top-1 accuracy. Does
report (Section V-A2, Table III) that no algorithm, including FedProx,
consistently outperforms FedAvg across all settings — e.g. CIFAR-10,
Dirichlet(0.5): FedAvg 68.2% vs. FedProx 67.9%; rcv1: FedAvg 48.2% vs. FedProx
70.3%.

**Relation to our finding:** Not a second precedent for our worst-client
contrast (it doesn't measure per-client accuracy at all) — but it is a direct
precedent for our project's own *methodology* (a multi-algorithm, multi-
heterogeneity-strategy characterization study) and for a specific validation
outcome: if our own Arm 3 (FedProx) does not clearly beat Arm 2 (FedAvg) at
low α, that would match this established result, not indicate a bug.

**Flag:** Cited for characterization-study precedent and FedProx-vs-FedAvg
inconsistency, explicitly not for worst-client accuracy, which it does not
report.

---

## Summary for the paper

For `paper/02_related_work.md`: [1] (Mohri) and [2] (q-FFL) establish the
average-vs-worst-case fairness framing generally, with [2]'s Appendix Table 10
as the specific, quantitatively-verified precedent for the average/worst-
client *disparity* — not for our specific "global flat" finding or our
convexity contrast, both of which remain ours. [6] (Ditto) shows this is an
active problem with multiple proposed fixes. [3] (FedProx) is our Arm 3
baseline and the grounding for our divergence metric; [8] (NIID-Bench) is the
characterization-study precedent and supports FedProx not uniformly beating
FedAvg. [7] (Hsu et al.) is the mandatory source attribution for Dirichlet-α
partitioning itself, and an outside contrast (their non-convex CNNs show
global degradation under skew; our convex LR does not). [4] (FedGA) and [5]
(Naseer & Shoaib) are recent (2025–2026) evidence the worst-client/average gap
generalizes across settings — [5] corrected 2026-08-20 to cite for the gap
existing, not for global accuracy specifically staying flat, which their own
Table 3 contradicts.

**What remains genuinely ours, after all of the above:** the specific
"global accuracy flat, worst-client accuracy collapses" shape (not just "worst-
client damage exceeds global damage," which is now well-precedented); the
convexity contrast (LR vs. MLP) under this exact shape, which none of [1]-[8]
test; confirmation of the disparity phenomenon on EHR tabular data specifically
rather than vision/text/synthetic data; and the extension to a variational
quantum classifier, which none of [1]-[8] address.
