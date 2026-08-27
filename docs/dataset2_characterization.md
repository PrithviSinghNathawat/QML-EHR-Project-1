# Dataset 2 Characterization — Diabetes 130-US Hospitals

2026-08-28. Data acquisition and characterisation only, per instruction — no
partitioning, no training, no experiments. Numbers below are produced by
`scripts/dataset2_characterize.py` and saved to
`results/dataset2_characterization.json`. See `data2/SOURCE.md` for
provenance/citation/license.

## Record count

- **101,766 encounters**, 101,766 unique `encounter_id` (one row per encounter,
  not per patient).
- **71,518 unique patients** (`patient_nbr`). **30,248 encounters (29.7%) are
  repeat visits** from a patient who already appears elsewhere in the dataset.
  This is a new leakage class not present in the heart-disease dataset (one row
  per patient there): a naive random split can put two encounters of the same
  patient on both sides of train/test. Flagged for the CV design, not solved
  here — see "Pipeline adaptation" below.

## Feature list, with missingness

50 columns total: `encounter_id`, `patient_nbr` (identifiers, not features),
`readmitted` (target), and 47 candidate features. Missing values are coded as
the literal string `"?"` (not blank) — read with `na_values="?"`.

**There is no per-hospital breakdown of missingness, because there is no
per-record hospital identifier in this dataset at all** (see next section). All
missingness below is pooled across the full 101,766-row population — the only
level at which it can be computed.

| feature | % missing |
|---|---|
| `weight` | 96.86% |
| `max_glu_serum` | 94.75% |
| `A1Cresult` | 83.28% |
| `medical_specialty` | 49.08% |
| `payer_code` | 39.56% |
| `race` | 2.23% |
| `diag_3` | 1.40% |
| `diag_2` | 0.35% |
| `diag_1` | 0.02% |
| all other 38 candidate features | 0.00% |

**Applying this project's ≥85%-present retention rule (D-016/D-019) at the pooled
level** (the only level available — see below): 5 features fail (`weight`,
`max_glu_serum`, `A1Cresult`, `medical_specialty`, `payer_code`), **42 of 47
candidate features pass**. Full retained/dropped lists in
`results/dataset2_characterization.json`.

**A second pathology the first dataset never surfaced: near-zero variance,
independent of missingness.** 15 of the 23 medication columns are >99% a single
value (`nateglinide` 99.3%, `chlorpropamide` 99.9%, `troglitazone` 99.997%, …),
and 2 (`examide`, `citoglipton`) are **100% one value** — zero variance, present
in every row, and would pass the missingness filter cleanly while carrying no
signal at all (and breaking a scaler that assumes non-zero range). The
missingness-only retention rule from dataset 1 is not sufficient here and needs
a variance/entropy floor added alongside it — not applied yet, since this task
is characterisation only, but noted as a required pipeline change.

## Natural client structure from hospital IDs — does not exist in this release

**This is the most consequential finding of this characterisation pass and is
reported plainly, since it changes the premise for client construction.**

All 50 columns were inspected directly (not inferred from documentation): there
is **no hospital, facility, or site identifier field anywhere in
`diabetic_data.csv`**. The closest categorical fields by cardinality —
`discharge_disposition_id` (26 values), `medical_specialty` (72 values, but
49.08% missing), `admission_source_id` (17 values), `payer_code` (17 values,
39.56% missing) — describe *how a patient entered or left care*, not *which of
the 130 institutions treated them*. None is a hospital identifier, and treating
any of them as one would misrepresent the data.

Checked directly against the dataset's own documentation (UCI page + the
source publication, Strack et al. 2014) to make sure this wasn't a parsing
error on our end: the paper describes the data as drawn from "130 US hospitals
and integrated delivery networks" — that is a fact about the **data warehouse
the extract was pulled from** (Cerner's Health Facts, a multi-institution
research database), not a claim that individual encounters carry a hospital
tag in the public release. Per-facility attribution was evidently stripped as
part of de-identification before release; this is consistent with how
multi-site EHR extracts are typically shared publicly. Nothing in this repo's
premise for Task 2 was wrong to expect — the "130 hospitals" fact is real — but
**it is not a usable per-record partition key**, and the second dataset cannot
replicate the heart-disease dataset's 4-real-site natural partition (D-037,
A-003) on this axis. See "Pipeline adaptation" below for what this means for
client construction, and P-009.

## Class balance

`readmitted` is 3-valued in the raw data:

| value | count | % |
|---|---|---|
| `NO` (not readmitted) | 54,864 | 53.91% |
| `>30` (readmitted after 30+ days) | 35,545 | 34.93% |
| `<30` (readmitted within 30 days) | 11,357 | 11.16% |

**Per-hospital class balance cannot be reported, for the same reason as
per-hospital missingness above — there is no hospital field to break it down
by.** Per-Dirichlet-client class balance (once a client scheme is chosen) can be
computed the same way as dataset 1, following the existing "print class balance
per client at each α" gate — not run yet, since this is characterisation only.

## Target variable

**Binary early readmission — `readmitted == "<30"` → 1, else → 0 — is the
standard task for this dataset**, matching the source publication's own framing
(Strack et al. 2014 predicts early readmission specifically, not readmission in
general, because early readmission is the metric tied to CMS penalties). Under
this binarization: **11.16% positive class** — meaningfully more imbalanced
than dataset 1's `num > 0` binarization. Accuracy alone may be a weak metric
here (a constant "never readmitted" predictor scores 88.84%); this is a design
question for the modeling phase, not resolved here, but flagged now so it
isn't rediscovered as a surprise later.

## Pipeline adaptation proposal (not implemented — proposal only)

**Feature retention under the ≥85% rule.** The rule itself (retain only if
≥85% present) transfers cleanly at the pooled level: 42 of 47 features pass
(see above). What does **not** transfer is the rule's *original purpose* —
D-016/D-019 used per-site missingness specifically to catch a feature that
looks fine pooled but is catastrophically missing at one real institution
(Switzerland's 100%-zero `chol`). Since dataset 2 has no real per-record site
field, that structural-missingness check cannot be replicated at all here —
only pooled missingness is checkable, which is a strictly weaker guarantee.
Recommend: (1) apply the pooled ≥85% rule as the primary filter (5 columns
dropped, as above), and (2) add a near-zero-variance floor (e.g. drop if one
value's share exceeds some threshold — 95% or 99%, to be decided) to catch the
15 near-constant medication columns the missingness rule alone would keep.

**Client construction: real hospital IDs are not available, so this cannot
mirror dataset 1's natural-partition arm (D-037/A-003) at all.** Recommended
path: use Dirichlet(α) synthetic partitioning as the *only* client-construction
method for dataset 2, exactly as already implemented in
`scripts/partitioner.py` — no new partitioning logic needed. Concretely, this
means dropping the "natural vs. synthetic" comparison for dataset 2 and
reporting it as a scope difference between the two datasets, not attempting to
manufacture a fake hospital key (e.g. from `admission_source_id`) to preserve
narrative symmetry with dataset 1.

**What changes at 130 clients instead of 4.** `scripts/partitioner.py`'s
Dirichlet draw already accepts an arbitrary client count `K` — running it with
`K=130` requires no new code, only a different call. At 130 clients, average
client size is 101,766/130 ≈ 783 rows (before Dirichlet skew), comfortably
above D-022's minimum-client-size floor (15 rows) even at low α, so the
existing reject-and-redraw guard should still behave sensibly, though this has
not been checked empirically (no experiment has run). The federated loop
(`scripts/federated_loop.py`) is compute-cost-agnostic to client count for the
classical arms in scope here (LR/MLP; no quantum arm requested for this
dataset yet) — 130 local `fit()` calls per round instead of 4 is still cheap on
CPU. The one genuinely new consideration is statistical, not computational —
see the prediction below.

## Pre-registered prediction — client count and the composition confound

**Logged before running anything, per instruction, so it can be checked rather
than rationalized after the fact.** P-007/P-008 traced dataset 1's
composition-decomposition gap to the minimum operator's sensitivity to the
Dirichlet partition's structure. If that sensitivity is real and generalizes,
it should scale with client count: **the minimum of more groups drawn from
similar underlying distributions is stochastically smaller than the minimum of
few, independent of any genuine training effect** (an order-statistics
argument, not specific to this project's models). Dataset 1 tested this
mechanism at 4-5 clients. Dataset 2's 130 clients is a genuine scale change on
the exact axis the mechanism depends on.

**Prediction: at 130 Dirichlet clients, the composition-only share of the
observed worst-client decline should be substantially larger than at 4-5
clients, for the same model family, even holding α fixed.** If this holds, it
strengthens the case that the composition confound is at least partly a
generic property of the minimum-over-many-groups statistic rather than
something specific to this project's 4-client setup — which would argue for
reporting worst-*decile* or worst-*quantile* accuracy instead of worst-client
minimum in future work, not just for this dataset but reopening the framing
question for dataset 1 too. If it does not hold — if the composition share
stays roughly flat from 4 to 130 clients — that would argue the mechanism is
more specific to the small-N regime and the 4-client framing was not
misleading. Either outcome is informative; not run yet.

## Summary of what's decided vs. open

**Decided by this characterisation pass:** record count, feature list and
missingness, that no hospital ID field exists, overall class balance, target
binarization convention.

**Explicitly not decided here (Prithvi's call):** the near-zero-variance
threshold, whether to run at K=130 or some smaller synthetic client count,
whether/when to bring a quantum arm to this dataset, and whether the
composition-confound prediction above should be tested before or after the
classical arms are built.
