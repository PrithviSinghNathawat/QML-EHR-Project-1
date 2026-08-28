"""Figure for P-016: divergence statistic vs. alpha, natural partition marked.
Reads results/alpha_calibration_fine.json (scripts/alpha_calibration_fine.py)."""
import json

import matplotlib.pyplot as plt
import numpy as np

with open("results/alpha_calibration_fine.json") as f:
    data = json.load(f)

grid_alphas = np.array(data["grid_alphas"])
natural = data["natural"]
per_alpha = data["per_alpha"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

panels = [
    (axes[0], "tv_mean", "tv_max", "Total variation distance", "TV distance"),
    (axes[1], "js_mean", "js_max", "Jensen-Shannon divergence (bits)", "JS divergence"),
]

for ax, mean_key, max_key, title, ylabel in panels:
    mean_curve = np.array(per_alpha[mean_key]["grid_means"])
    max_curve = np.array(per_alpha[max_key]["grid_means"])
    mean_std = np.array(per_alpha[mean_key]["grid_std"])
    max_std = np.array(per_alpha[max_key]["grid_std"])

    ax.plot(grid_alphas, mean_curve, color=cycle[0], marker="o", ms=3, label="Dirichlet, client-weighted mean")
    ax.fill_between(grid_alphas, mean_curve - mean_std, mean_curve + mean_std, color=cycle[0], alpha=0.15)
    ax.plot(grid_alphas, max_curve, color=cycle[1], marker="s", ms=3, label="Dirichlet, max across clients")
    ax.fill_between(grid_alphas, max_curve - max_std, max_curve + max_std, color=cycle[1], alpha=0.15)

    ax.axhline(natural[mean_key], color=cycle[0], ls="--", lw=1.2)
    ax.axhline(natural[max_key], color=cycle[1], ls="--", lw=1.2)

    for key, color in [(mean_key, cycle[0]), (max_key, cycle[1])]:
        info = per_alpha[key]
        if info["in_range"]:
            eq = info["equivalent_alpha"]
            ci_lo, ci_hi = info["ci95"]
            ax.axvspan(ci_lo, ci_hi, color=color, alpha=0.08)
            ax.axvline(eq, color=color, ls=":", lw=1.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Dirichlet $\alpha$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8, loc="upper right")

fig.suptitle("Natural 4-site partition calibrated against synthetic Dirichlet $\\alpha$\n"
             "(dashed = natural partition's value; dotted + shaded band = equivalent $\\alpha$ point estimate + 95% CI)")
fig.tight_layout()
fig.savefig("results/figs/alpha_calibration_fine.png", dpi=200)
print("wrote results/figs/alpha_calibration_fine.png")
