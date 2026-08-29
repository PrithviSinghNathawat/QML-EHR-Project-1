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
    # Orthogonal ("bus") routing throughout -- every wire is a sequence of
    # horizontal/vertical segments aligned with a box's own center, so no
    # line ever crosses through a neighbouring box. The original version
    # used straight diagonal lines converging on single center points, which
    # visibly cut through Client 1 and Client 2's text (found on inspection
    # after it shipped in the deck) -- this replaces that approach entirely
    # rather than nudging the same diagonal lines.
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.set_xlim(-1.0, 10.2)
    ax.set_ylim(0, 6.6)
    ax.axis("off")

    LINE = dict(color="#1E2761", lw=1.8, solid_capstyle="round")
    global_cx = 5.0
    global_top, global_bottom = 5.5, 4.5
    box(ax, (3.7, global_bottom), 2.6, global_top - global_bottom,
        "Global model\n(server)", fc="#1E2761", ec="#1E2761", fontsize=12, weight="bold")
    ax.texts[-1].set_color("white")

    client_x = [0.3, 2.9, 5.5, 8.1]
    client_w = 1.8
    client_top, client_bottom = 3.1, 2.0
    client_centers = [x + client_w / 2 for x in client_x]
    for i, x in enumerate(client_x):
        box(ax, (x, client_bottom), client_w, client_top - client_bottom,
            f"Client {i}\nlocal fit()", fc="#e8eef7")

    agg_cx = 5.0
    agg_top, agg_bottom = 1.3, 0.3
    box(ax, (3.4, agg_bottom), 3.2, agg_top - agg_bottom,
        "aggregate(params, sizes)\n-> new global model", fc="#fff3e0", ec="#c77800")

    # Broadcast bus: global -> horizontal bus -> straight down into each client
    bus1_y = 3.85
    ax.plot([global_cx, global_cx], [global_bottom, bus1_y], **LINE)
    ax.plot([client_centers[0], client_centers[-1]], [bus1_y, bus1_y], **LINE)
    for cx in client_centers:
        ax.plot([cx, cx], [bus1_y, client_top + 0.22], **LINE)
        arrow(ax, (cx, client_top + 0.22), (cx, client_top))

    # Upload bus: each client -> straight up into a horizontal bus -> aggregate
    bus2_y = 1.65
    ax.plot([client_centers[0], client_centers[-1]], [bus2_y, bus2_y], **LINE)
    for cx in client_centers:
        ax.plot([cx, cx], [client_bottom, bus2_y], **LINE)
    ax.plot([agg_cx, agg_cx], [bus2_y, agg_top + 0.22], **LINE)
    arrow(ax, (agg_cx, agg_top + 0.22), (agg_cx, agg_top))

    # Round loop-back: routed out to the far left, clear of every box, so it
    # cannot cross the bus lines or Client 0.
    loop_x = -0.55
    ax.plot([3.4, loop_x], [agg_bottom + 0.5, agg_bottom + 0.5], **LINE)
    ax.plot([loop_x, loop_x], [agg_bottom + 0.5, global_bottom + 0.5], **LINE)
    ax.plot([loop_x, 3.7 - 0.22], [global_bottom + 0.5, global_bottom + 0.5], **LINE)
    arrow(ax, (3.7 - 0.22, global_bottom + 0.5), (3.7, global_bottom + 0.5))

    ax.text(5.0, 6.25, "One federated round, repeated for R rounds", ha="center", fontsize=13, weight="bold")
    ax.text(5.0, 5.8, "Loop never inspects which model class it holds", ha="center", fontsize=9.5, style="italic", color="#555")

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
