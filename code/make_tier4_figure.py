"""Forest plot: every Tier 4 configuration against the pre-registered floor.

    python code/make_tier4_figure.py   -> figures/fig_tier4.png

Values are the seed-paired differences in sub-12-pixel recall against the Tier 1
baseline, read from results/protocol/results/tier4/TIER4_VS_BASELINE.csv (written
by code/make_tier4_table.py): a bias-corrected bootstrap interval where all three
training runs completed, and a point estimate without an interval for the
configurations the pre-registration classes as underpowered.
"""
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")

INK = "#1F2937"
GREY = "#6B7280"
GOOD = "#15803D"
BAD = "#B91C1C"

CSV = os.path.join(ROOT, "results", "protocol", "results", "tier4", "TIER4_VS_BASELINE.csv")


def load_arms():
    """-> [(label, mean paired difference, ci low or None, ci high or None)]"""
    out = []
    for r in csv.DictReader(open(CSV, encoding="utf-8")):
        lo = float(r["ci_lo"]) if r["ci_lo"] else None
        hi = float(r["ci_hi"]) if r["ci_hi"] else None
        out.append((r["arm"], float(r["diff"]), lo, hi))
    return out


def main():
    ARMS = load_arms()
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 3.9), dpi=200)
    ys = np.arange(len(ARMS))[::-1]

    ax.axvspan(-0.02, 0.02, color="#F3F4F6", zorder=0)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    for x in (-0.02, 0.02):
        ax.axvline(x, color=GREY, lw=1.0, ls="--", zorder=2)
    ax.text(0.0215, len(ARMS) - 0.30, "pre-registered floor,  $\\pm$0.02",
            fontsize=9.4, color=GREY, va="center")

    for y, (name, m, lo, hi) in zip(ys, ARMS):
        if lo is None:
            # underpowered: the point estimate only, hollow and grey, no whiskers
            ax.plot([m], [y], "o", ms=8, color=GREY, mfc="white", mew=1.4,
                    ls="none", zorder=4)
            ax.text(0.995, y, "underpowered, no interval", fontsize=9.6,
                    va="center", ha="right", color=GREY,
                    transform=ax.get_yaxis_transform(which="grid"))
            continue
        crosses = lo <= 0 <= hi
        clears = m > 0.02 and not crosses
        col = BAD if (m < 0 and not crosses) else (GOOD if clears else GREY)
        ax.plot([lo, hi], [y, y], color=col, lw=2.0, solid_capstyle="round",
                zorder=3)
        for e in (lo, hi):
            ax.plot([e, e], [y - .12, y + .12], color=col, lw=1.6, zorder=3)
        ax.plot([m], [y], "o", ms=8, color=col, mfc="white", mew=2.0, zorder=4)
        verdict = ("worse than baseline" if (m < 0 and not crosses)
                   else "clears the floor" if clears
                   else "no effect" if crosses else "below the floor")
        ax.text(0.995, y, verdict, fontsize=9.6, va="center", ha="right",
                color=col, transform=ax.get_yaxis_transform(which="grid"))

    ax.set_yticks(ys)
    labels = [n for n, _, _, _ in ARMS]
    ax.set_yticklabels(labels, fontsize=10.4)
    # The last configuration listed is the one that clears the floor; it is named
    # in bold. Tick labels follow the order given to set_yticks, and ys counts down
    # from the top, so index 0 is the first configuration listed.
    ax.get_yticklabels()[len(ARMS) - 1].set_fontweight("bold")
    ax.set_xlabel("seed-paired change in sub-12-pixel recall",
                  fontsize=11)
    ax.set_xlim(-0.058, 0.098)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#F3F4F6", lw=0.8)
    ax.set_axisbelow(True)

    fig.tight_layout()
    p = os.path.join(OUT, "fig_tier4.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote figures/fig_tier4.png")


if __name__ == "__main__":
    main()
