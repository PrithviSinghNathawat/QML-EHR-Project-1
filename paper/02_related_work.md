# 2. Related Work

Full literature detail and verification notes behind every claim in this
section: `docs/reference/fl_fairness_literature.md`.

## 2.1 Federated learning under heterogeneity: FedAvg, FedProx

Federated averaging (FedAvg) [1] established the basic federated training
pattern this project's loop follows: clients train locally on their own data,
and a server aggregates their parameters by a size-weighted average. FedAvg
was not designed with data heterogeneity across clients as a central concern,
and subsequent work identified that non-IID client data degrades its
convergence and final accuracy. FedProx [2] addresses this directly by adding
a proximal term to each client's local objective, penalizing drift away from
the global model before aggregation — the same mechanism our own client
parameter-divergence metric (`docs/diagnostic_report.md`, Section 2) measures
directly. FedProx is our Arm 3 baseline; per `CLAUDE.md`, it reuses FedAvg's
aggregation unchanged, with all of its behavior difference living in the
client's local `fit`.

## 2.2 Fairness and client-level disparity: is the worst client protected?

FedAvg's average-loss objective gives no guarantee about the loss on any
individual client — only about the loss averaged across the (weighted) client
population. Mohri et al. [3] made this precise, proposing an alternative
minimax ("agnostic") objective that optimizes for the worst-case client
distribution rather than the average one. Li et al. [4] (q-FFL) measured the
consequence directly: under vanilla FedAvg, test accuracy varies substantially
across devices, and their Appendix Table 10 gives the specific, quantitative
form of this that our own result echoes — under FedAvg, as their heterogeneity
increases (synthetic IID → moderate → severe, 100 devices), **average accuracy
declines a real but modest 6.6 percentage points (89.2% → 82.6%), while
worst-10%-of-devices accuracy collapses 44.7 points (70.9% → 25.5%)**. This is
the specific precedent for our own contrast between global and worst-client
accuracy: **the disparity between the two — worst-client damage substantially
exceeding global damage under heterogeneity — is established prior art, not a
discovery of this project.**

It is a precedent for the disparity, not for the specific shape of our own
result. Their average accuracy moves a real 6.6 points; our own logistic
regression's global accuracy is flat within measurement noise across our
entire α sweep (0.7625 to 0.7651). Their setup — a synthetic linear/softmax
classification task — offers no convex-versus-non-convex model comparison, so
our finding that a non-convex model (MLP) shows a global-accuracy penalty at
extreme heterogeneity that a convex model (logistic regression) does not
(`docs/decisions.md`, D-028) has no counterpart in their work. Their device
counts (50, 100) are also an order of magnitude above our 4 clients, and their
own results show fewer devices producing *more* uniform accuracy — so their
trend does not extrapolate freely down to our client count. **What we
contribute, against this precedent, is specifically the flatness of the
global metric in a convex model, and the breakdown of that flatness in a
non-convex one, on a new data modality (EHR tabular data) that this literature
does not address.**

Li et al. [5] (Ditto) propose a distinct response to the same underlying
disparity problem: rather than reweighting the server-side objective toward
poorly-performing clients (q-FFL's approach), each client keeps a personalized
model regularized toward the global one, pursuing fairness and robustness to
adversarial clients simultaneously. We do not evaluate personalization in this
project — out of scope per this project's no-scope-additions guardrail — but
Ditto is evidence the client-disparity problem remains an active area with
multiple competing families of proposed fixes, not a solved or narrow concern.

Two more recent works extend this evidence base. Liu [6] (FedGA, 2025) reports
a persistent gap between mean and bottom-decile accuracy under FedAvg and
FedProx baselines, motivating a Gini-coefficient-based reweighting scheme.
Naseer and Shoaib [7] (2026) report worst-client accuracy gaps of 50.1%
(large, pretrained model) versus 32.2% (small model) under extreme label
skew, on a text-classification task — evidence the disparity phenomenon
generalizes across model scale, not just across the specific settings q-FFL
tested. We checked their own accuracy table directly rather than relying on
their framing: their *global* accuracy also varies substantially across their
sweep (86.6%–97.8% for the small model), so — as with q-FFL — we cite this
work for the disparity it reports, not for a global-accuracy-flatness finding
that its own data does not show.

## 2.3 NIID-Bench: a characterization-study precedent

Li et al. [8] (NIID-Bench) benchmark FedAvg, FedProx, and other algorithms
across several non-IID partitioning strategies, including Dirichlet-α — a
characterization study close in spirit to this project's own goal. We checked
directly (their public code and paper text) whether they report per-client or
worst-client accuracy under their sweep: **they do not** — only aggregate,
global top-1 accuracy, in every setting. NIID-Bench is therefore not a second
precedent for our worst-client contrast. It is a precedent for a different,
useful point: their Table III shows no algorithm, including FedProx,
consistently outperforming FedAvg across settings (e.g. CIFAR-10 at
Dirichlet(0.5): FedAvg 68.2% vs. FedProx 67.9%; rcv1: FedAvg 48.2% vs. FedProx
70.3%). If our own Arm 3 does not clearly beat Arm 2 at low α, that would
match this established result rather than indicate an implementation defect.

## 2.4 Hsu et al.: the source of Dirichlet partitioning, and a convexity contrast

Hsu et al. [9] introduced Dirichlet(α)-based synthetic label-skew partitioning
as a controllable way to simulate non-IID client data — this is the specific
method this project's independent variable is built on
(`scripts/partitioner.py`), and this citation is the mandatory attribution for
it, not an optional one. Beyond the method itself, their results offer an
external contrast for our own convexity finding: their CNNs (non-convex) show
clear *global*-accuracy degradation as their Dirichlet concentration
decreases (their most-skewed CIFAR-10 setting reaches a baseline accuracy
around 30.1%, well below their more-IID settings). Our convex model (logistic
regression) shows no such global-accuracy degradation under the same
partitioning method (D-028). This is external, independent support for the
same convexity-mediates-global-visibility pattern our own LR-vs-MLP comparison
finds internally.

## 2.5 Quantum federated learning

*[Placeholder — no verified quantum federated learning citations are available
in this repository as of 2026-08-20. `docs/reference/` contains only the
classical FL fairness literature summarized above; no quantum-FL papers have
been added or verified. This subsection needs Prithvi's quantum-side
literature (or a fresh, independently verified search) before it can be
written — left marked rather than filled with unverified citations.]*

## 2.6 Research gap: calibrating synthetic heterogeneity against real institutional heterogeneity

The federated learning literature that uses Dirichlet-α (or similar)
synthetic partitioning — including the foundational method paper itself [9]
and the characterization study closest to our own goals [8] — does not, as
far as we have found, calibrate that synthetic severity scale against a real,
naturally-occurring institutional partition on the same dataset. We checked
two of the closest candidates directly: a federated learning study across
real educational institutions uses Dirichlet-parameterized partitions to
*simulate* institutional heterogeneity, rather than comparing real
institutional splits against a synthetic scale; a federated EHR heterogeneity
study across seven real hospitals (AKI/sepsis risk prediction) uses only real
institutional splits, with no synthetic Dirichlet comparison at all. **We did
not find a paper that places a real institutional partition on the same
synthetic Dirichlet-α scale it is being compared against** — not a claim that
no such paper exists, only that a reasonable-effort search did not surface
one. This project's own natural-vs-Dirichlet comparison
(`docs/diagnostic_report.md`, Section 5) does exactly this calibration for
the UCI Heart Disease dataset's four real sites, finding the natural partition
sits mid-range on the synthetic scale (comparable to α≈0.5–1.0) rather than
outside it.

## References

[1] B. McMahan, E. Moore, D. Ramage, S. Hampson, and B. A. y Arcas,
    "Communication-efficient learning of deep networks from decentralized
    data," in *Proc. 20th Int. Conf. Artif. Intell. Stat. (AISTATS)*, 2017,
    pp. 1273-1282.

[2] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith,
    "Federated optimization in heterogeneous networks," in *Proc. Mach.
    Learn. Syst. (MLSys)*, 2020.

[3] M. Mohri, G. Sivek, and A. T. Suresh, "Agnostic federated learning," in
    *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, Long Beach, CA, USA, 2019,
    pp. 4615-4625.

[4] T. Li, M. Sanjabi, A. Beirami, and V. Smith, "Fair resource allocation in
    federated learning," in *Proc. Int. Conf. Learn. Represent. (ICLR)*,
    Addis Ababa, Ethiopia, 2020.

[5] T. Li, S. Hu, A. Beirami, and V. Smith, "Ditto: Fair and robust federated
    learning through personalization," in *Proc. 38th Int. Conf. Mach. Learn.
    (ICML)*, 2021.

[6] S. Liu, "FedGA: A fair federated learning framework based on the Gini
    coefficient," *arXiv:2507.12983*, Jul. 2025.

[7] K. Naseer and U. Shoaib, "When more parameters hurt: Foundation model
    priors amplify worst-client disparity under extreme federated
    heterogeneity," *arXiv:2605.08992*, 2026.

[8] Q. Li, Y. Diao, Q. Chen, and B. He, "Federated learning on non-IID data
    silos: An experimental study," in *Proc. IEEE 38th Int. Conf. Data Eng.
    (ICDE)*, 2022.

[9] T-M. H. Hsu, H. Qi, and M. Brown, "Measuring the effects of non-identical
    data distribution for federated visual classification,"
    *arXiv:1909.06335*, 2019.

*Quantum federated learning citations: pending — see Section 2.5.*
