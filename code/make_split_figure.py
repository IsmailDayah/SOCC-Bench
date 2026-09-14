"""The split design, with its counts read from the Study M4 partition.

    python code/make_split_figure.py   -> figures/fig_split_design.png

A split is decided per flight episode, never per frame: consecutive frames of one
orbit are near duplicates, so assigning at frame level would place near-identical
images on both sides of the partition and report memorisation as generalisation.
Image and box counts come from results/protocol/study_M4/assignment.csv.
"""
import collections
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
ASSIGNMENT = os.path.join(ROOT, "results", "protocol", "study_M4", "assignment.csv")

INK = "#111827"
MUTE = "#6B7280"

# (split, title, role, fill, edge)
SPLITS = [
    ("train", "TRAIN", "model fitting", "#DBEAFE", "#1D4ED8"),
    ("val", "VALIDATION", "checkpoint selection", "#FEF3C7", "#B45309"),
    ("test", "TEST", "never seen in training", "#EDE9FE", "#6D28D9"),
]

SESSIONS = ["S1  rooftop futsal court", "S2  graded occluder rig",
            "S3  covered walkway", "S4  park deck"]


def counts():
    """split -> (images, boxes) from the partition file."""
    imgs, boxes = collections.Counter(), collections.Counter()
    for r in csv.DictReader(open(ASSIGNMENT, encoding="utf-8")):
        imgs[r["split"]] += 1
        boxes[r["split"]] += int(r["boxes"])
    return imgs, boxes


def card(ax, x, y, w, h, fill, edge, lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        facecolor=fill, edgecolor=edge, linewidth=lw, zorder=2))


def main():
    imgs, boxes = counts()
    os.makedirs(OUT, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.6, 4.6), dpi=200)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    # ---- the four capture sessions feeding the corpus --------------------
    ax.text(0.5, 0.965, "four capture sessions", ha="center", va="center",
            fontsize=10.5, color=MUTE)
    sw, gap = 0.212, 0.018
    x0 = (1.0 - (4 * sw + 3 * gap)) / 2
    for i, name in enumerate(SESSIONS):
        x = x0 + i * (sw + gap)
        card(ax, x, 0.845, sw, 0.078, "#F3F4F6", "#9CA3AF", lw=1.2)
        ax.text(x + sw / 2, 0.884, name, ha="center", va="center",
                fontsize=9.2, color=INK)

    # ---- the unit of assignment -----------------------------------------
    ax.add_patch(FancyArrowPatch((0.5, 0.838), (0.5, 0.775),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#9CA3AF", lw=1.4, zorder=1))
    card(ax, 0.145, 0.655, 0.71, 0.105, "#FFFFFF", INK, lw=1.6)
    ax.text(0.5, 0.728, "unit of assignment:  one whole flight episode",
            ha="center", va="center", fontsize=11.5, color=INK,
            fontweight="bold")
    ax.text(0.5, 0.688,
            "every frame of a flight goes to the same split, never to two",
            ha="center", va="center", fontsize=9.6, color=MUTE)

    ax.add_patch(FancyArrowPatch((0.5, 0.648), (0.5, 0.585),
                                 arrowstyle="-|>", mutation_scale=13,
                                 color="#9CA3AF", lw=1.4, zorder=1))

    # ---- the three splits ------------------------------------------------
    cw, cgap = 0.29, 0.035
    cx0 = (1.0 - (3 * cw + 2 * cgap)) / 2
    for i, (split, title, role, fill, edge) in enumerate(SPLITS):
        x = cx0 + i * (cw + cgap)
        card(ax, x, 0.255, cw, 0.315, fill, edge)
        card(ax, x + 0.022, 0.487, cw - 0.044, 0.062, edge, edge)
        ax.text(x + cw / 2, 0.518, title, ha="center", va="center",
                fontsize=11.5, color="white", fontweight="bold")
        ax.text(x + cw / 2, 0.428, f"{imgs[split]:,} images", ha="center",
                va="center", fontsize=11.0, color=INK)
        ax.text(x + cw / 2, 0.375, f"{boxes[split]:,} boxes", ha="center",
                va="center", fontsize=11.0, color=INK)
        ax.text(x + cw / 2, 0.302, role, ha="center", va="center",
                fontsize=9.4, color=MUTE, style="italic")

    # ---- what the design buys, stated once -------------------------------
    ax.text(0.5, 0.158,
            f"{sum(imgs.values()):,} images and {sum(boxes.values()):,} boxes in "
            "total; the test split is held byte-identical across every stage",
            ha="center", va="center", fontsize=9.6, color=INK)
    ax.text(0.5, 0.085,
            "consecutive frames of one orbit are near-duplicates, so a "
            "frame-level split would report memorisation as generalisation",
            ha="center", va="center", fontsize=9.6, color=MUTE, style="italic")

    p = os.path.join(OUT, "fig_split_design.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  wrote figures/fig_split_design.png")


if __name__ == "__main__":
    main()
