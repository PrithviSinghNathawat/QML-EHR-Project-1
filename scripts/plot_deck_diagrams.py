"""Two schematic diagrams for the Review-2 deck only (not a pipeline output,
no experimental data -- pure illustration of the federated loop and the
diagnostic-pair method). Saved to results/figs/.
"""
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

OUT_DIR = "results/figs"


def box(ax, xy, w, h, text, fc="#e8eef7", ec="#1E2761", fontsize=11, weight="normal"):
    rect = mpatches.FancyBboxPatch(
        xy, w, h, boxstyle="round,pad=0.02,rounding_size=0.05",
        linewidth=1.5, edgecolor=ec, facecolor=fc,
    )
    ax.add_patch(rect)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center",
             fontsize=fontsize, weight=weight, wrap=True)


def arrow(ax, xy1, xy2, **kw):
    ax.annotate("", xy=xy2, xytext=xy1,
                arrowprops=dict(arrowstyle="-|>", color="#1E2761", lw=1.8, **kw))


def federated_loop_diagram():
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    ax.set_xlim(0, 10.2)
    ax.set_ylim(0, 6.3)
    ax.axis("off")

    box(ax, (3.7, 4.3), 2.6, 1.0, "Global model\n(server)", fc="#1E2761", ec="#1E2761", fontsize=12, weight="bold")
    ax.texts[-1].set_color("white")

    client_x = [0.3, 2.9, 5.5, 8.1]
    for i, x in enumerate(client_x):
        box(ax, (x, 2.0), 1.8, 1.1, f"Client {i}\nlocal fit()", fc="#e8eef7")
        arrow(ax, (5.0, 4.3), (x + 0.9, 3.1))  # broadcast down
        arrow(ax, (x + 0.9, 2.0), (5.0, 4.3))  # params up

    box(ax, (3.4, 0.2), 3.2, 1.1, "aggregate(params, sizes)\n-> new global model", fc="#fff3e0", ec="#c77800")
    for x in client_x:
        arrow(ax, (x + 0.9, 2.0), (5.0, 1.3))
    arrow(ax, (5.0, 1.3), (5.0, 4.3), connectionstyle="arc3,rad=-0.3")

    ax.text(5.0, 6.0, "One federated round, repeated for R rounds", ha="center", fontsize=13, weight="bold")
    ax.text(5.0, 5.55, "Loop never inspects which model class it holds", ha="center", fontsize=9.5, style="italic", color="#555")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "deck_architecture_loop.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved: deck_architecture_loop.png")


def diagnostic_pair_diagram():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    box(ax, (3.2, 4.9), 3.6, 0.9, "Per-α trained models\nM(100), M(1.0), M(0.5), M(0.1)", fc="#e8eef7")

    box(ax, (0.4, 2.9), 4.1, 1.4,
        "Estimator A: composition\ndecomposition\n\nFreeze M(100), re-score against\nevery α's client-local test slice",
        fc="#fff3e0", ec="#c77800", fontsize=10)
    box(ax, (5.5, 2.9), 4.1, 1.4,
        "Estimator B: shared-test\nre-evaluation\n\nScore each M(α) on one pooled,\nα-invariant held-out set",
        fc="#e6f4ea", ec="#1b7a3d", fontsize=10)

    arrow(ax, (5.0, 4.9), (2.45, 4.3))
    arrow(ax, (5.0, 4.9), (7.55, 4.3))

    box(ax, (1.7, 0.9), 2.6, 1.0, "Agree\n-> trust the estimate\n(no real training effect)", fc="#e6f4ea", ec="#1b7a3d", fontsize=10)
    box(ax, (5.7, 0.9), 3.0, 1.0, "Diverge\n-> model-partition\ninteraction term (a finding)", fc="#fde8e8", ec="#b3261e", fontsize=10)
    arrow(ax, (2.5, 2.9), (3.0, 1.9))
    arrow(ax, (7.5, 2.9), (7.2, 1.9))

    ax.text(5.0, 5.85, "The diagnostic pair", ha="center", fontsize=14, weight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "deck_diagnostic_pair.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved: deck_diagnostic_pair.png")


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    federated_loop_diagram()
    diagnostic_pair_diagram()
