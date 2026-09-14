"""Recall by size band for every Tier 1-3 arm, from the merged band ledgers.

    python code/make_band_figure.py   -> figures/fig_band_recall.png

Each line is the mean over the seeds scored; error bars are the seed range
(half-spread), not a confidence interval. RT-DETR-L has two seeds.
"""
import collections
import csv
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "protocol", "results")
OUT = os.path.join(ROOT, "figures")
INK = "#0E1B2E"
ACC = "#C2410C"
GREY = "#64748B"


def bands(path):
    out = collections.defaultdict(dict)
    if not os.path.exists(path):
        return out
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try:
            out[r["tag"]][r["slice"]] = float(r["recall"])
        except (KeyError, ValueError):
            continue
    return out


def fig_bands():
    b1 = bands(os.path.join(RES, "tier1_backfill", "BANDS_tier1_backfill.csv"))
    b2 = bands(os.path.join(RES, "tier2", "BANDS_tier2.csv"))
    b3 = bands(os.path.join(RES, "tier3", "BANDS_tier3.csv"))
    allb = {**b1, **b2, **b3}
    arms = [("yolov8n baseline", ["M_v8n_baseline_s%s" % s for s in (42, 123, 456)],
             INK, "o", "-", 2.4),
            ("+ P2", ["M_yolov8n-p2_baseline_s%s" % s for s in (42, 123, 456)],
             "#64748B", "s", "--", 1.5),
            ("+ CBAM", ["M_yolov8n-cbam_baseline_s%s" % s for s in (42, 123, 456)],
             "#94A3B8", "^", ":", 1.5),
            ("+ P2 + CBAM", ["M_yolov8n-p2-cbam_baseline_s%s" % s for s in (42, 123, 456)],
             "#CBD5E1", "D", "-.", 1.5),
            ("Faster R-CNN", ["M_fasterrcnn_baseline_s%s" % s for s in (42, 123, 456)],
             "#0369A1", "v", "--", 1.7),
            ("RT-DETR-L (n=2)", ["M_rtdetr-l_baseline_s%s" % s for s in (42, 456)],
             ACC, "P", "-", 1.7)]
    order = ["size:<12", "size:12-20", "size:20-32", "size:32+"]
    xs = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(9.6, 5.0), dpi=200)
    for name, tags, col, mk, ls, lw in arms:
        ys, es = [], []
        for slc in order:
            v = [allb[t][slc] for t in tags if t in allb and slc in allb[t]]
            ys.append(st.mean(v) if v else np.nan)
            es.append((max(v) - min(v)) / 2 if len(v) > 1 else 0)
        ax.errorbar(xs, ys, yerr=es, label=name, color=col, marker=mk,
                    linestyle=ls, linewidth=lw, markersize=6, capsize=3,
                    markeredgecolor="white", markeredgewidth=0.6)
    ax.axvspan(-0.35, 0.35, color=ACC, alpha=0.07)
    ax.text(0, 1.045, "the band this project is about", ha="center",
            fontsize=9.5, color=ACC, fontweight="bold")
    ax.set_xticks(xs)
    ax.set_xticklabels(["< 12 px\n(993 boxes)", "12-20 px\n(1149)",
                        "20-32 px\n(1748)", "32+ px\n(277)"], fontsize=10)
    ax.set_ylabel("recall on the test split", fontsize=11, color=INK)
    ax.set_ylim(0, 1.12)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(axis="y", linestyle=":", color="#CBD5E1", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(fontsize=9, frameon=False, loc="lower right", ncol=2)
    ax.tick_params(colors=INK)
    fig.text(0.5, -0.035,
             "Error bars are the seed range (half-spread), not a confidence interval. "
             "Every arm collapses in the same place.",
             ha="center", fontsize=9.5, color=GREY)
    fig.tight_layout()
    p = os.path.join(OUT, "fig_band_recall.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("  wrote figures/fig_band_recall.png")


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    fig_bands()
