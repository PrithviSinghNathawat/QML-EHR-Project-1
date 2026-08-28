"""P-016: fine-grid, dual-statistic alpha calibration for the natural 4-site
partition, superseding A-003's 4-point/10-seed/TV-only estimate.

A-003 bracketed the natural partition's TV distance between only two of the
four alphas already used elsewhere in this project (100, 1.0, 0.5, 0.1) --
adequate for a rough check, not for a precise equivalent-alpha claim. This
script:

  - adds Jensen-Shannon divergence (log base 2, bounded [0,1] like TV
    distance) alongside TV distance, so the calibration isn't a single-metric
    artifact;
  - reports both the client-size-weighted MEAN and the unweighted MAX across
    clients (A-003 only had the mean) -- the max is the more relevant
    statistic for a "worst client" framing, since one very skewed client
    changes the risk profile even if the average client looks mild;
  - sweeps a fine log-spaced alpha grid (25 points, 0.05 to 100) with 30
    seeds per point (A-003 used the original 4-point grid at 10 seeds) so the
    equivalent-alpha estimate is a genuine curve fit, not a two-point
    bracket-and-interpolate.

For a binary label, TV distance between a client's P(y=1)=p_k and the pooled
P(y=1)=p_global reduces to |p_k - p_global| (two-point distribution
identity). JS divergence between two Bernoulli(p_k), Bernoulli(p_global) is
computed directly from the standard KL-based definition.
"""
import sys

import numpy as np

sys.path.insert(0, "scripts")
from cv_protocol import client_assignment, load_pool  # noqa: E402

SEEDS = list(range(30))
ALPHA_GRID = np.geomspace(0.05, 100, 25)
N_BOOTSTRAP = 10_000
RNG_SEED = 0


def _bernoulli_kl(p, q):
    """KL(Bernoulli(p) || Bernoulli(q)), base-2, with 0*log(0)=0 convention."""
    terms = []
    for a, b in [(p, q), (1 - p, 1 - q)]:
        if a == 0:
            terms.append(0.0)
        elif b == 0:
            terms.append(np.inf)
        else:
            terms.append(a * np.log2(a / b))
    return sum(terms)


def _js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * _bernoulli_kl(p, m) + 0.5 * _bernoulli_kl(q, m)


def per_client_distances(df, assign):
    """Returns (weights, tv_per_client, js_per_client) for one partition draw."""
    p_global = df["target"].mean()
    total = len(df)
    weights, tvs, jss = [], [], []
    for c in sorted(assign.unique()):
        client_rows = df.loc[assign == c]
        if len(client_rows) == 0:
            continue
        p_k = client_rows["target"].mean()
        weights.append(len(client_rows) / total)
        tvs.append(abs(p_k - p_global))
        jss.append(_js_divergence(p_k, p_global))
    return np.array(weights), np.array(tvs), np.array(jss)


def summarize_partition(df, assign):
    w, tv, js = per_client_distances(df, assign)
    return {
        "tv_mean": float(np.sum(w * tv)),
        "tv_max": float(np.max(tv)),
        "js_mean": float(np.sum(w * js)),
        "js_max": float(np.max(js)),
    }


def natural_stats(df):
    assign = client_assignment(df, "natural", seed=0)
    return summarize_partition(df, assign)


def dirichlet_stats_by_seed(df, alpha):
    rows = []
    for seed in SEEDS:
        assign = client_assignment(df, alpha, seed=seed)
        rows.append(summarize_partition(df, assign))
    return {k: np.array([r[k] for r in rows]) for k in rows[0]}


def equivalent_alpha(grid_alphas, grid_means, target_value):
    """Interpolate the fine grid (assumed monotonically decreasing in alpha,
    typical for these divergence statistics) to find the alpha at which the
    mean statistic equals target_value. np.interp needs ascending x, and the
    statistic is descending in alpha, so interpolate on the reversed curve in
    log-alpha space. Returns None if target_value falls outside the grid's
    observed range (extrapolation refused, not silently produced)."""
    order = np.argsort(grid_means)  # ascending statistic -> ascending log-alpha reversed
    stat_sorted = grid_means[order]
    log_alpha_sorted = np.log(grid_alphas)[order]
    if not (stat_sorted.min() <= target_value <= stat_sorted.max()):
        return None
    log_eq = np.interp(target_value, stat_sorted, log_alpha_sorted)
    return float(np.exp(log_eq))


if __name__ == "__main__":
    import json

    df = load_pool()
    nat = natural_stats(df)
    print("Natural partition:", {k: round(v, 4) for k, v in nat.items()})

    grid_results = {}
    for alpha in ALPHA_GRID:
        grid_results[float(alpha)] = dirichlet_stats_by_seed(df, alpha)
        means = {k: v.mean() for k, v in grid_results[float(alpha)].items()}
        print(f"alpha={alpha:.3f}: " + " ".join(f"{k}={v:.4f}" for k, v in means.items()))

    grid_alphas = np.array(list(grid_results.keys()))
    output = {"natural": nat, "grid_alphas": grid_alphas.tolist(), "per_alpha": {}}

    rng = np.random.default_rng(RNG_SEED)
    for stat in ["tv_mean", "tv_max", "js_mean", "js_max"]:
        grid_means = np.array([grid_results[a][stat].mean() for a in grid_alphas])
        point_est = equivalent_alpha(grid_alphas, grid_means, nat[stat])

        boot_estimates = []
        for _ in range(N_BOOTSTRAP):
            resampled_means = np.array([
                rng.choice(grid_results[a][stat], size=len(grid_results[a][stat]), replace=True).mean()
                for a in grid_alphas
            ])
            est = equivalent_alpha(grid_alphas, resampled_means, nat[stat])
            if est is not None:
                boot_estimates.append(est)
        boot_estimates = np.array(boot_estimates)

        if point_est is None:
            print(f"\n{stat}: natural value {nat[stat]:.4f} falls OUTSIDE the "
                  f"grid's range [{grid_means.min():.4f}, {grid_means.max():.4f}] "
                  f"-- does not map onto any tested alpha.")
            output["per_alpha"][stat] = {
                "natural_value": nat[stat],
                "grid_range": [float(grid_means.min()), float(grid_means.max())],
                "equivalent_alpha": None,
                "ci95": None,
                "in_range": False,
            }
        else:
            ci_lo, ci_hi = np.percentile(boot_estimates, [2.5, 97.5]) if len(boot_estimates) else (None, None)
            print(f"\n{stat}: natural value {nat[stat]:.4f} -> equivalent alpha "
                  f"{point_est:.3f} (95% CI [{ci_lo:.3f}, {ci_hi:.3f}], "
                  f"n_boot_valid={len(boot_estimates)})")
            output["per_alpha"][stat] = {
                "natural_value": nat[stat],
                "equivalent_alpha": point_est,
                "ci95": [float(ci_lo), float(ci_hi)],
                "in_range": True,
            }
        output["per_alpha"][stat]["grid_means"] = grid_means.tolist()
        output["per_alpha"][stat]["grid_std"] = np.array(
            [grid_results[a][stat].std() for a in grid_alphas]
        ).tolist()

    with open("results/alpha_calibration_fine.json", "w") as f:
        json.dump(output, f, indent=2)
    print("\nwrote results/alpha_calibration_fine.json")
