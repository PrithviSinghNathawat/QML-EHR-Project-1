# Evaluation-Protocol Confound: Client-Partitioned Test Data — Literature

Compiled 2026-08-22 (Ayuvi). Purpose: our decomposition analysis
(`docs/decisions.md` D-044-D-052, `docs/arm4_report.md`) found that because the
same client assignment determines both training partition and test slices, a
completely fixed, untrained-further model appears to degrade on worst-client
accuracy as α falls, purely from evaluation-slice composition — separate from
any genuine training-heterogeneity effect. LR's observed decline was 82%
composition, MLP's 27%, VQC's 117% (residual negative). This is now our
strongest candidate contribution. Task: establish whether the FL literature
already identifies and separates this confound. **Same rigor standard as
`fl_fairness_literature.md`: verified citations, explicit confidence per
claim, plain statement if already published — not softened.**

**The confound, precisely, for reference:** any FL evaluation design where
(a) a client's test data is drawn from the same skewed partition as its
training data, and (b) the reported metric involves a *worst-client* or
*per-client* statistic (not a metric computed once against the same test set
regardless of the training partition) is structurally exposed to this
confound — a model that is *fully insensitive* to the training partition can
still show a worst-client-accuracy trend purely because which rows land in
whose test slice changes as α changes.

---

## Does the general FL/pFL literature address this confound?

**Confidence: high that client-local, matching-skew train/test splitting is
standard practice.** A general search across FL evaluation-methodology
literature returned a direct, unambiguous characterization: "each client
typically has a local train set and a corresponding test set with the same
Dirichlet distribution... common for simulating realistic heterogeneous
federated settings," and personalized-FL evaluation standardly computes
"accuracy of the local non-IID test set." This is not one paper's practice —
it is described as the field's common approach.

**Confidence: high that this confound is not commonly named or decomposed.**
No search turned up a named term for this effect ("evaluation-composition
artifact," "test-slice confound," or similar), nor a paper proposing to
separate it from genuine training effects, despite multiple search framings
(evaluation protocol design, personalized-vs-global evaluation, client-
specific test distributions, separating distribution-shift from training
degradation). This is a "did not find," not a "does not exist" — reported at
the confidence a reasonable-effort literature and code search supports, not
higher.

---

## [1] pFL-Bench — a comprehensive personalized-FL benchmark, directly checked

Wang, D. et al., "pFL-Bench: A comprehensive benchmark for personalized
federated learning," *arXiv:2206.03655*, 2022.

**Verified directly (arXiv HTML, quoted text), confidence: high.** States
plainly: *"For all the adopted datasets, the train/val/test splitting is
conducted within the local data of each client."* Every dataset in the
benchmark (FEMNIST 3:1:1, CIFAR10 4:1:1, Twitter ~3:1:1) uses this
within-client splitting.

**Does it address our confound? No evidence that it does.** A client with
skewed training labels gets a correspondingly skewed test slice from the same
local pool. No decomposition of evaluation-composition from training-effect
was found in the accessible text. **This is a comprehensive, purpose-built
personalized-FL benchmark paper — exactly the kind of paper that would be
expected to raise this issue if it were a recognized concern in the field —
and it does not raise it.** This is the single strongest piece of evidence
that our decomposition is not already standard methodology.

---

## [2] q-FFL (Li, Sanjabi, Beirami, Smith, ICLR 2020) — checked at the source-code level

**Their synthetic-data experiments (the ones behind Table 10, our cited
prior-art source for the disparity phenomenon) use client-local, matching-skew
train/test splits — verified directly from their public code, confidence:
high, not an inference.** Fetched `data/synthetic/generate_synthetic.py`
(github.com/litian96/fair_flearn) directly. Each device `i` has its own
generated distribution; the split is:

```python
combined = list(zip(X[i], y[i]))
random.shuffle(combined)
num_samples = len(X[i])
train_len = int(0.9 * num_samples)
train_data['user_data'][uname] = {'x': X[i][:train_len], 'y': y[i][:train_len]}
test_data['user_data'][uname]  = {'x': X[i][train_len:], 'y': y[i][train_len:]}
```

The 90/10 split happens *within* each device's own data — a device with a
more skewed generated distribution gets a correspondingly skewed test slice,
by construction.

**Consequence, stated plainly: q-FFL's own Table 10 — our primary cited
prior-art for the average/worst-decile disparity — is itself potentially
subject to this exact confound, and we found no evidence they decompose it.**
This does not undo q-FFL as prior art for "the disparity exists" (D-031/
D-033 stand: their Average-vs-Worst-10% gap is real and quantified). It does
mean their Table 10 numbers describe *observed* disparity, not necessarily a
disparity attributable to genuine training-heterogeneity sensitivity as
opposed to evaluation composition — the same distinction our own decomposition
draws for LR/MLP/VQC. **We were not able to confirm whether their reported
gap would itself shrink under an analogous decomposition** (out of scope to
re-run their experiments) — flagged as an open question, not claimed as an
answer.

---

## [3] NIID-Bench (Li, Diao, Chen, He, ICDE 2022) — checked at the source-code level

**Uses a single shared/global test set, not partitioned per party — verified
directly from their public code, confidence: high.** Fetched
`utils.py` (github.com/Xtra-Computing/NIID-Bench). `partition_data()` returns
`net_dataidx_map` (per-party training index map) but loads test data once,
unpartitioned:

```python
X_test, y_test = cifar10_test_ds.data, cifar10_test_ds.target
```

`get_dataloader()` applies the per-party `dataidxs` only to the *training*
dataset; the test dataloader is built without any party-specific index
filter. `compute_accuracy()` runs over this single shared test set.

**Our confound does not apply to NIID-Bench, and we should say so precisely:
their evaluation is not client-partitioned, so a fixed model's reported
accuracy cannot shift purely from test-slice composition changing with α — 
there is only one test slice, shared across every party and every α
condition.** This is consistent with, and now explains at the mechanism
level, our earlier finding (D-032) that NIID-Bench never reports per-client
accuracy at all: with a shared test set, there is no meaningful "per-client
test accuracy" to report in the first place — global accuracy on a shared
test set is the only accuracy their design produces.

---

## [4] Ditto (Li, Hu, Beirami, Smith, ICML 2021) — checked at the source-code level, one level less complete than q-FFL

**Evaluates each client on its own local test data — verified directly from
their public code, confidence: medium-high (the trainer-level mechanism is
directly quoted; the specific data-generation script for each dataset was
not individually re-checked the way q-FFL's was).** Fetched
`flearn/trainers_MTL/fedbase.py` (github.com/litian96/ditto):

```python
def test(self, models):
    '''tests self.latest_model on given clients'''
    for idx, c in enumerate(self.clients):
        self.client_model.set_params(models[idx])
        ct, cl, ns = c.test()
```

Each client object is evaluated via its own `c.test()`, against `test_data[u]`
set up per-client at initialization — not a shared test set.

**Ditto shares its codebase lineage and lead author with q-FFL and FedProx**
(same `flearn` framework family), which is suggestive but not itself proof
that its specific data-generation scripts split train/test the same skewed
way q-FFL's `generate_synthetic.py` does — flagged as the one place in this
review where the conclusion rests partly on lineage/pattern inference rather
than a fully independent code check of the exact same file type verified for
q-FFL. **Best-supported reading: Ditto's per-client accuracy numbers are also
exposed to the same confound, at slightly lower confidence than q-FFL's
direct verification.**

---

## Does "personalized vs. global evaluation" already cover this?

**No — different axis, checked and rejected as a match.** Personalized-FL
papers (Ditto included) commonly report two different numbers: "personalized/
local accuracy" (each client's own fine-tuned model on its own local test
data) versus "global/generic accuracy" (one shared model evaluated somewhere,
sometimes on a shared set). This distinguishes *which model* is being
evaluated (personalized vs. shared), not *whether the evaluation composition
itself shifts with heterogeneity for a fixed model*. Our confound is about
the latter: even a single, non-personalized, fixed global model shows the
composition artifact if its worst-client accuracy is computed against
client-local test slices that themselves shift with α. This is a genuinely
different question from the personalized-vs-global framing, not a
relabeling of it.

---

## Summary for the paper

**Plain statement, not softened:** we did not find a paper that decomposes
worst-client (or per-client) accuracy degradation under increasing FL
heterogeneity into an evaluation-composition component and a genuine
training-heterogeneity component. Client-local, matching-skew train/test
splitting is standard practice in this literature (confirmed directly for
q-FFL, Ditto, and pFL-Bench's general design) — meaning most of the
per-client/worst-client numbers this field reports, including our own primary
cited prior-art source (q-FFL's Table 10), are potentially exposed to exactly
this confound, unaddressed. NIID-Bench is the one checked exception, and it
is an exception by construction, not by decomposition: a shared test set
makes the confound structurally impossible, which is also why it never
reports per-client accuracy in the first place. **Phrase in the paper as "we
did not find," not "first" or "novel"** — this is a reasonable-effort search
and code-check across four specific sources plus general literature, not an
exhaustive survey of every FL evaluation-methodology paper in existence.
