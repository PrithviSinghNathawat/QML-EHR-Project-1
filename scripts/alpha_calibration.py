"""Calibrates the natural 4-site partition against the synthetic Dirichlet
alpha scale, using total variation (TV) distance between each client's local
label distribution and the pooled label distribution -- a formal distance
metric, not the informal "sits between alpha=0.5 and 1.0" comparison used
elsewhere (D-037), which was based on comparing divergence/worst-client
accuracy magnitudes, not a distance-metric fit.

For a binary label, TV distance between a client's P(y=1)=p_k and the
pooled P(y=1)=p_global reduces to |p_k - p_global| (standard identity for
two-point distributions). Client-size-weighted mean across clients gives one
scalar per (condition, seed): the natural partition is deterministic (single
value); each Dirichlet alpha is stochastic, so this is repeated across the
same 10 seeds used everywhere else in this project's protocol.

Equivalent alpha for the natural partition: log-linear interpolation between
the two Dirichlet alpha values whose mean TV distance bracket the natural
partition's TV distance. CI via percentile bootstrap: resample each
bracketing alpha's 10 seed-level TV values with replacement, recompute the
interpolated equivalent alpha, repeat 10,000 times.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from cv_protocol import client_assignment, load_pool  # noqa: E402

SEEDS = list(range(10))
ALPHAS = [100, 1.0, 0.5, 0.1]
N_BOOTSTRAP = 10_000
RNG_SEED = 0


def weighted_tv_distance(df, assign) -> float:
    p_global = df["target"].mean()
    total = len(df)
    dist = 0.0
    for c in sorted(assign.unique()):
        client_rows = df.loc[assign == c]
        if len(client_rows) == 0:
            continue
        p_k = client_rows["target"].mean()
        weight = len(client_rows) / total
        dist += weight * abs(p_k - p_global)
    return dist


def natural_tv(df) -> float:
    assign = client_assignment(df, "natural", seed=0)  # natural doesn't depend on seed
    return weighted_tv_distance(df, assign)


def dirichlet_tv_by_seed(df, alpha) -> np.ndarray:
    vals = []
    for seed in SEEDS:
        assign = client_assignment(df, alpha, seed=seed)
        vals.append(weighted_tv_distance(df, assign))
    return np.array(vals)


def interpolate_equivalent_alpha(alpha_lo, tv_lo, alpha_hi, tv_hi, tv_natural) -> float:
    """Log-linear interpolation in alpha, linear in TV distance between the
    two bracketing points. alpha_lo < alpha_hi in value; TV distance is
    monotonically decreasing in alpha, so tv_lo > tv_hi."""
    log_lo, log_hi = np.log(alpha_lo), np.log(alpha_hi)
    frac = (tv_lo - tv_natural) / (tv_lo - tv_hi)
    frac = np.clip(frac, 0.0, 1.0)
    log_interp = log_lo + frac * (log_hi - log_lo)
    return float(np.exp(log_interp))


if __name__ == "__main__":
    df = load_pool()
    tv_natural = natural_tv(df)
    print(f"Natural partition TV distance: {tv_natural:.4f}")

    tv_by_alpha = {a: dirichlet_tv_by_seed(df, a) for a in ALPHAS}
    for a in ALPHAS:
        vals = tv_by_alpha[a]
        print(f"alpha={a}: TV distance mean={vals.mean():.4f} std={vals.std():.4f} (n={len(vals)} seeds)")

    # Find the bracketing pair (alphas sorted ascending; TV distance descending in alpha)
    sorted_alphas = sorted(ALPHAS)
    means = {a: tv_by_alpha[a].mean() for a in sorted_alphas}
    bracket = None
    for lo, hi in zip(sorted_alphas, sorted_alphas[1:]):
        if means[hi] <= tv_natural <= means[lo]:
            bracket = (lo, hi)
    if bracket is None:
        print("Natural TV distance falls outside the tested alpha range -- no interpolation possible.")
        sys.exit(0)

    alpha_lo, alpha_hi = bracket
    point_estimate = interpolate_equivalent_alpha(
        alpha_lo, means[alpha_lo], alpha_hi, means[alpha_hi], tv_natural
    )
    print(f"\nBracketing alphas: {alpha_lo} (TV={means[alpha_lo]:.4f}) -- {alpha_hi} (TV={means[alpha_hi]:.4f})")
    print(f"Point estimate, equivalent alpha: {point_estimate:.4f}")

    rng = np.random.default_rng(RNG_SEED)
    boot_estimates = []
    lo_vals, hi_vals = tv_by_alpha[alpha_lo], tv_by_alpha[alpha_hi]
    for _ in range(N_BOOTSTRAP):
        lo_resample = rng.choice(lo_vals, size=len(lo_vals), replace=True).mean()
        hi_resample = rng.choice(hi_vals, size=len(hi_vals), replace=True).mean()
        if lo_resample == hi_resample:
            continue
        boot_estimates.append(
            interpolate_equivalent_alpha(alpha_lo, lo_resample, alpha_hi, hi_resample, tv_natural)
        )
    boot_estimates = np.array(boot_estimates)
    ci_lo, ci_hi = np.percentile(boot_estimates, [2.5, 97.5])
    print(f"95% bootstrap CI (resampling seed-level TV distances, {len(boot_estimates)} draws): [{ci_lo:.4f}, {ci_hi:.4f}]")
