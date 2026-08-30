"""Composition-Controlled Evaluation (CCE).

CCE is an evaluation protocol for federated-learning studies that sweep
data heterogeneity as an independent variable and report client-local
metrics (worst-client accuracy, per-client accuracy variance, and similar
statistics computed over each client's own held-out partition).

**The problem it corrects for.** When client-local test data is
partitioned by the same assignment used to partition training data,
sweeping heterogeneity changes two things at once: the model (because
training-data composition changes) and the evaluation (because each
client's test-slice composition changes too, becoming more skewed as
heterogeneity increases). A client-local metric computed this way
conflates a genuine training effect with an artifact of what is being
measured — the two cannot be told apart from the metric's value alone,
and this is invisible at any single heterogeneity level, since a
confound in a *trend* requires a trend to exist.

**What CCE does.** It controls for the evaluation-composition half of
that conflation using two independent estimators, applied to the SAME
already-trained models with no retraining required:

1. **Pooled accuracy** (`cce_pooled_accuracy`) on a fixed,
   heterogeneity-invariant held-out set — a mean-based statistic that
   does not depend on any per-client partition, so it is structurally
   unaffected by the confound.
2. **Worst-group accuracy** (`cce_worst_group_accuracy`, using
   `cce_fixed_partition` for the group assignment) on a FIXED random
   partition of that same held-out set — a minimum-based statistic,
   matching the *form* of a client-local "worst-client" claim, but
   computed on a partition that does not vary with heterogeneity, so any
   change in it is training-driven, not composition-driven.

Comparing either estimator's decline against the observed decline under
the study's original (heterogeneity-dependent) client-local partition —
`cce_paired_estimate` — isolates the training effect from the
composition effect and reports both, plus the gap between them.

**How to apply this to a new study.** Hold out a heterogeneity-invariant
test set *before* partitioning by heterogeneity condition (the same test
data must be usable regardless of which heterogeneity condition trained
the model being scored). Evaluate every already-trained model against it
with `cce_pooled_accuracy` and `cce_worst_group_accuracy`. Feed the
resulting declines, alongside whatever client-local metric the study
already reports, to `cce_paired_estimate`. Agreement between the two CCE
estimators and the client-local metric indicates the client-local trend
is training-driven; disagreement indicates a composition confound, an
unisolated model-partition interaction, or both — CCE detects this, it
does not resolve it on its own.

**Reporting-floor rule** (`cce_paired_estimate`'s `floor_pp` argument,
default 2.0): a composition/training *share*, expressed as a percentage
of the observed decline, is a ratio with the observed decline as its
denominator. As the observed decline shrinks toward zero, this ratio
becomes numerically unstable for any nonzero training-effect estimate —
this is a property of the ratio itself, not of either estimator. Below
the floor, `cce_paired_estimate` omits the percentage fields and reports
absolute values only.
"""
import numpy as np

DEFAULT_N_GROUPS = 4


def _metric_fn(metric: str):
    """Returns a (y_true, y_pred) -> float scorer. 'accuracy' is plain
    accuracy. 'balanced' is balanced accuracy (mean per-class recall),
    falling back to plain accuracy when only one class is present in
    y_true, since balanced accuracy is undefined there."""
    from sklearn.metrics import accuracy_score, balanced_accuracy_score

    if metric == "accuracy":
        return lambda y_true, y_pred: float(accuracy_score(y_true, y_pred))
    if metric == "balanced":
        def _scorer(y_true, y_pred):
            if len(np.unique(y_true)) < 2:
                return float((y_true == y_pred).mean())
            return float(balanced_accuracy_score(y_true, y_pred))
        return _scorer
    raise ValueError(f"unknown metric {metric!r}, expected 'accuracy' or 'balanced'")


def cce_pooled_accuracy(model, X, y, metric: str = "accuracy") -> float:
    """Shared-test pooled accuracy: the model's accuracy on the full,
    heterogeneity-invariant held-out set X, y. A mean-based statistic --
    unaffected by client-partition skew, since there is no per-client
    split here at all."""
    score = _metric_fn(metric)
    pred = (model.predict_proba(X) >= 0.5).astype(int)
    return score(y, pred)


def cce_fixed_partition(n_rows: int, seed: int, fold: int, n_groups: int = DEFAULT_N_GROUPS,
                         base_seed: int = 100_000) -> np.ndarray:
    """Deterministic, heterogeneity-independent assignment of n_rows test
    rows into n_groups groups. Constructed once per (seed, fold) and
    reused identically across every heterogeneity condition evaluated for
    that (seed, fold), so the partition itself never varies with
    heterogeneity -- any change in worst-group accuracy computed on it is
    therefore training-driven, not composition-driven.

    base_seed disambiguates independent uses of this function within the
    same (seed, fold) -- e.g. two different studies, or two different
    metric choices, evaluated on data that happens to share seed/fold
    numbering should not silently reuse each other's "random" partition.
    Existing callers each keep their own historical base_seed (see the
    call sites in dataset1_reeval_balanced.py, dataset2_decomposition.py,
    shared_test_worst_group.py) so that already-published numbers stay
    exactly reproducible; a new caller should pick an unused base_seed."""
    rng = np.random.default_rng(base_seed + seed * 100 + fold)
    return rng.integers(0, n_groups, size=n_rows)


def cce_worst_group_accuracy(model, X, y, partition: np.ndarray, n_groups: int = DEFAULT_N_GROUPS,
                              metric: str = "accuracy") -> float:
    """Minimum accuracy across the n_groups fixed groups in `partition`
    (from cce_fixed_partition). A minimum-based statistic, matching the
    form of a worst-client claim, but computed on a partition that is
    heterogeneity-invariant. Groups with zero rows are skipped, not
    scored as a failure; returns NaN if every group is empty."""
    score = _metric_fn(metric)
    accs = []
    for g in range(n_groups):
        mask = partition == g
        if mask.sum() > 0:
            pred = (model.predict_proba(X[mask]) >= 0.5).astype(int)
            accs.append(score(y[mask], pred))
    return min(accs) if accs else float("nan")


def cce_paired_estimate(observed_decline_pp: float, composition_only_decline_pp: float,
                         shared_test_decline_pp: float, floor_pp: float = 2.0) -> dict:
    """Combines a client-local decomposition (observed_decline_pp,
    composition_only_decline_pp -- both computed by the caller's own
    heterogeneity-dependent client partition) with a CCE shared-test
    decline (shared_test_decline_pp, from cce_pooled_accuracy or
    cce_worst_group_accuracy, computed against a heterogeneity-invariant
    partition) into one paired estimate.

    Returns absolute pp values always; percentage-share fields are None
    when observed_decline_pp is below floor_pp, per this module's
    reporting-floor rule (see module docstring).

    Why the floor exists: a composition/training share is a percentage of
    observed_decline_pp -- algebraically, share = 100 - 100*(TE/observed),
    where TE is whichever training-effect estimate (decomposition residual
    or shared-test decline) sits in the numerator. As observed_decline_pp
    -> 0, this ratio is unbounded for any fixed nonzero TE: a small,
    noisy denominator can turn an unremarkable training effect into an
    arbitrarily large or negative-looking percentage, and can flip sign
    entirely, without the underlying quantities having changed in any
    interesting way. This is a property of the ratio itself, not of
    which estimator produced TE or of the specific study it's applied
    to -- it will happen in any adopting study that reports a
    composition/training share for a configuration whose observed
    decline is small. Below floor_pp, this function reports the absolute
    pp values instead of a percentage for that reason, not as a
    stylistic preference."""
    decomposition_residual_pp = observed_decline_pp - composition_only_decline_pp
    implied_composition_pp = observed_decline_pp - shared_test_decline_pp
    interaction_pp = decomposition_residual_pp - shared_test_decline_pp
    clears_floor = abs(observed_decline_pp) >= floor_pp

    def _pct(numerator):
        if not clears_floor:
            return None
        return round(100 * numerator / observed_decline_pp, 1)

    return {
        "observed_decline_pp": round(observed_decline_pp, 3),
        "composition_only_decline_pp": round(composition_only_decline_pp, 3),
        "decomposition_residual_pp": round(decomposition_residual_pp, 3),
        "decomposition_share_pct": _pct(composition_only_decline_pp),
        "shared_test_training_effect_pp": round(shared_test_decline_pp, 3),
        "implied_composition_pp": round(implied_composition_pp, 3),
        "implied_composition_share_pct": _pct(implied_composition_pp),
        "interaction_pp": round(interaction_pp, 3),
        "clears_reporting_floor": clears_floor,
    }


if __name__ == "__main__":
    # Smoke test / usage example, not a unit test suite.
    example = cce_paired_estimate(observed_decline_pp=18.46, composition_only_decline_pp=4.99,
                                   shared_test_decline_pp=4.62)
    import json
    print(json.dumps(example, indent=2))
