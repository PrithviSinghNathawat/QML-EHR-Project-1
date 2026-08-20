# 2. Related Work

<!-- Federated learning on non-IID data; variational quantum classifiers; quantum federated learning; EHR/healthcare FL. -->

## Federated learning fairness / client-level performance disparity

Full literature summary: `docs/reference/fl_fairness_literature.md`.

The average-vs-worst-case distinction in federated objectives goes back to
Mohri et al. [1], who show FedAvg's average-loss objective gives no guarantee
about performance on any individual client. Li et al. [2] measure this
directly as accuracy variance across devices under vanilla FedAvg and propose
q-FFL to narrow it. Li et al. [3] (FedProx, used as our Arm 3 baseline) target
the same underlying client-drift mechanism our divergence metric measures.
Two recent works, Liu [4] (FedGA, 2025) and Naseer and Shoaib [5] (2026),
report worst-client/bottom-decile accuracy gaps that persist under FedAvg even
as mean accuracy stays flat — [5] in particular reports the same qualitative
pattern we find (global accuracy insulated from damage that concentrates in
worst-client accuracy, worsening with heterogeneity), on a text-classification
task. We do not claim this pattern as a novel finding; our contribution is
confirming it on EHR tabular data and extending the comparison to a
variational quantum classifier, which none of [1]-[5] address.

**References**

[1] M. Mohri, G. Sivek, and A. T. Suresh, "Agnostic federated learning," in
    *Proc. 36th Int. Conf. Mach. Learn. (ICML)*, Long Beach, CA, USA, 2019,
    pp. 4615-4625.

[2] T. Li, M. Sanjabi, A. Beirami, and V. Smith, "Fair resource allocation in
    federated learning," in *Proc. Int. Conf. Learn. Represent. (ICLR)*,
    Addis Ababa, Ethiopia, 2020.

[3] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, and V. Smith,
    "Federated optimization in heterogeneous networks," in *Proc. Mach.
    Learn. Syst. (MLSys)*, 2020.

[4] S. Liu, "FedGA: A fair federated learning framework based on the Gini
    coefficient," *arXiv:2507.12983*, Jul. 2025.

[5] K. Naseer and U. Shoaib, "When more parameters hurt: Foundation model
    priors amplify worst-client disparity under extreme federated
    heterogeneity," *arXiv:2605.08992*, 2026.
