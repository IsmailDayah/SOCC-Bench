"""Corpus and result figures, computed from the released labels and the result ledgers.

    python code/make_result_figures.py [--only NAME]

    size           figures/fig_size_histogram.png   size distribution per session
    class          figures/fig_class_balance.png    class counts and sizes
    collapse       figures/fig_size_collapse.png    Tier 1 recall by size band, three seeds
    interventions  figures/fig_interventions.png    Tier 2 and Tier 3 against the floor
    budget         figures/fig_box_budget.png       recall, box budget, ungrounded detections

Every number is read at run time from dataset/*/release/labels or from
results/protocol/results, so a figure cannot drift from the ledgers it plots.
Bands and arms are separated by position and direct labels as well as colour, so
every figure also reads in greyscale.
"""
import argparse
import glob
import os
import statistics as st
import sys
from collections import Counter, defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures")
RES = os.path.join(ROOT, "results", "protocol", "results")
sys.path.insert(0, os.path.join(ROOT, "code"))

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "axes.edgecolor": "#333333",
    "axes.linewidth": 0.9,
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#1a1a1a",
    "axes.labelcolor": "#1a1a1a",
})

INK, GREY = "#1F2937", "#6B7280"
# a sequential ramp that also reads as light-to-dark in greyscale
BAND_C = ["#0B2545", "#134074", "#3E7CB1", "#A8C6E0"]
CLASSES = ["rc_car", "atv_sherp", "truck_peterbilt"]
NICE = {"rc_car": "compact car", "atv_sherp": "all-terrain vehicle",
        "truck_peterbilt": "truck"}

# session -> (label dir, display name)
SCENES = [
    ("S1", "dataset/01_Dataset_SOCC/release/labels", "S1  rooftop court"),
    ("S2", "dataset/01_Dataset_SOCC_S2/release/labels", "S2  graded occluder rig"),
    ("S3", "dataset/01_Dataset_SOCC_S3/release/labels", "S3  covered walkway"),
    ("S4", "dataset/01_Dataset_SOCC_S4/release/labels", "S4  park deck"),
]
IMG_W, IMG_H = 960, 720
SEEDS = ["42", "123", "456"]
SIZE_BANDS = ["size:<12", "size:12-20", "size:20-32", "size:32+"]


def read_scene(rel):
    """Every box in one session as sqrt(area) in pixels, plus its class id."""
    d = os.path.join(ROOT, rel)
    sizes, cls = [], []
    for lp in glob.glob(os.path.join(d, "*.txt")):
        try:
            with open(lp) as fh:
                for ln in fh:
                    p = ln.split()
                    if len(p) < 5:
                        continue
                    w, h = float(p[3]) * IMG_W, float(p[4]) * IMG_H
                    if w > 0 and h > 0:
                        sizes.append((w * h) ** 0.5)
                        cls.append(int(p[0]))
        except OSError:
            continue
    return np.array(sizes), np.array(cls)


def load_corpus():
    out = {}
    for tag, rel, nice in SCENES:
        s, c = read_scene(rel)
        out[tag] = {"sizes": s, "cls": c, "nice": nice}
        print(f"    {tag}: {len(s)} boxes, median {np.median(s):.1f} px"
              if len(s) else f"    {tag}: NO LABELS FOUND at {rel}")
    return out


# --------------------------------------------------------------- figure 1
def fig_size_distribution(corpus):
    """Where the corpus sits on the size axis, per session.

    The extreme tail is real but scarce, and it belongs almost entirely to one
    venue - which is why no result ranks the size and occlusion axes against each
    other. Thresholds are labelled in a band reserved above the histogram, and the
    legend sits below both panels, so nothing is drawn over the data.
    """
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.9, 3.4), dpi=200,
        gridspec_kw={"width_ratios": [1.5, 1.0], "wspace": 0.42})

    allsz = np.concatenate([d["sizes"] for d in corpus.values()])
    bins = np.logspace(np.log10(4), np.log10(300), 46)

    # ---- left: stacked histogram, one colour per session ----------------
    data = [corpus[t]["sizes"] for t, _, _ in SCENES]
    labs = [corpus[t]["nice"] for t, _, _ in SCENES]
    axL.hist(data, bins=bins, stacked=True, color=BAND_C,
             edgecolor="white", linewidth=0.35, label=labs)
    axL.set_xscale("log")
    axL.set_xlim(4, 300)
    axL.set_xlabel("target size, $\\sqrt{\\mathrm{box\\ area}}$ (pixels)", fontsize=11)
    axL.set_ylabel("instances", fontsize=11)
    top = axL.get_ylim()[1]
    axL.set_ylim(0, top * 1.46)          # reserved band for the threshold labels

    for x, txt in ((12, "< 12 px"), (32, "32 px")):
        axL.axvline(x, color=INK, lw=1.3, ls="--", zorder=5)
        axL.text(x, top * 1.38, txt, ha="center", va="center",
                 fontsize=9.0, color=INK, zorder=6,
                 bbox=dict(fc="white", ec="none", pad=1.5))

    n_small = int((allsz < 32).sum())
    n_tail = int((allsz < 12).sum())
    axL.text(0.985, 0.865,
             f"{len(allsz):,} instances\n"
             f"{n_small:,} below 32 px  ({100*n_small/len(allsz):.1f}%)\n"
             f"{n_tail:,} below 12 px  ({100*n_tail/len(allsz):.1f}%)",
             transform=axL.transAxes, ha="right", va="top", fontsize=9.0,
             color=INK, linespacing=1.5, zorder=7,
             bbox=dict(fc="white", ec="#D1D5DB", lw=0.8, pad=5))
    axL.spines[["top", "right"]].set_visible(False)

    # ---- right: who owns the extreme tail --------------------------------
    tails = [int((corpus[t]["sizes"] < 12).sum()) for t, _, _ in SCENES]
    ypos = np.arange(len(SCENES))[::-1]
    axR.barh(ypos, tails, height=0.58, color=BAND_C, edgecolor="white", lw=1.0)
    for y, v in zip(ypos, tails):
        axR.text(v + max(tails) * 0.025, y, f"{v:,}", va="center",
                 fontsize=10, color=INK)
    axR.set_yticks(ypos)
    # only the session code here; the legend below gives each code its name
    axR.set_yticklabels([corpus[t]["nice"].split()[0] for t, _, _ in SCENES],
                        fontsize=10)
    axR.set_xlabel("instances below 12 px", fontsize=11)
    axR.set_xlim(0, max(tails) * 1.30)
    axR.set_ylim(-1.15, len(SCENES) - 0.4)
    share = 100 * max(tails) / max(sum(tails), 1)
    axR.text(0.5, -0.92, f"one venue contributes {share:.0f}% of the extreme tail",
             fontsize=9.8, color=INK, style="italic", ha="left",
             transform=axR.get_yaxis_transform(which="grid"))
    axR.spines[["top", "right"]].set_visible(False)
    axR.tick_params(axis="y", length=0)

    handles, labels = axL.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               fontsize=9.0, bbox_to_anchor=(0.5, 0.045))
    fig.subplots_adjust(bottom=0.28)

    p = os.path.join(OUT, "fig_size_histogram.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


# --------------------------------------------------------------- figure 2
def fig_class_balance(corpus):
    """Class frequency and the size each class actually presents at."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(7.6, 2.8), dpi=200,
                                   gridspec_kw={"wspace": 0.26})

    counts = Counter()
    per_cls = defaultdict(list)
    for t, _, _ in SCENES:
        d = corpus[t]
        for s, c in zip(d["sizes"], d["cls"]):
            if 0 <= c < len(CLASSES):
                counts[c] += 1
                per_cls[c].append(s)

    ids = sorted(counts, key=lambda k: -counts[k])
    names = [NICE[CLASSES[i]] for i in ids]
    vals = [counts[i] for i in ids]
    xs = np.arange(len(ids))
    axL.bar(xs, vals, width=0.58, color=BAND_C[:len(ids)], edgecolor="white", lw=1.0)
    for x, v in zip(xs, vals):
        axL.text(x, v + max(vals) * 0.02, f"{v:,}", ha="center", fontsize=10, color=INK)
    axL.set_xticks(xs); axL.set_xticklabels(names, fontsize=10)
    axL.set_ylabel("instances", fontsize=11)
    axL.set_ylim(0, max(vals) * 1.15)
    axL.spines[["top", "right"]].set_visible(False)

    parts = [np.array(per_cls[i]) for i in ids]
    vp = axR.violinplot(parts, positions=xs, widths=0.72, showextrema=False,
                        showmedians=True)
    for b, c in zip(vp["bodies"], BAND_C[:len(ids)]):
        b.set_facecolor(c); b.set_alpha(0.85); b.set_edgecolor("white")
    vp["cmedians"].set_color(INK); vp["cmedians"].set_linewidth(1.4)
    axR.axhline(32, color=INK, lw=1.2, ls="--")
    axR.set_yscale("log")
    # bands are reserved above and below the violins so no label crosses one
    lo = min(float(a.min()) for a in parts)
    hi = max(float(a.max()) for a in parts)
    axR.set_ylim(lo * 0.34, hi * 2.6)
    axR.text(len(ids) - 0.5, hi * 1.75, "COCO small-object threshold", ha="right",
             va="center", fontsize=9.2, color=INK)
    axR.set_xticks(xs); axR.set_xticklabels(names, fontsize=10)
    axR.set_ylabel("target size (pixels)", fontsize=11)
    axR.spines[["top", "right"]].set_visible(False)
    for x, i in zip(xs, ids):
        axR.text(x, lo * 0.46, f"median {np.median(per_cls[i]):.0f} px",
                 ha="center", va="center", fontsize=9.0, color=GREY)

    p = os.path.join(OUT, "fig_class_balance.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


# --------------------------------------------------------------- figure 3
def fig_size_collapse():
    """Tier 1: recall against target size, three seeds, from BANDS.csv."""
    from analyse_tier2 import load_bands
    b = load_bands(os.path.join(RES, "BANDS.csv"))
    labels = ["< 12 px", "12-20 px", "20-32 px", "32+ px"]
    gt = [b[f"M_v8n_baseline_s{SEEDS[0]}"][sl][0] for sl in SIZE_BANDS]
    seeds = {f"seed {s}": [b[f"M_v8n_baseline_s{s}"][sl][2] for sl in SIZE_BANDS]
             for s in SEEDS}

    fig, ax = plt.subplots(figsize=(7.4, 3.5), dpi=200)
    xs = np.arange(len(labels))
    arr = np.array(list(seeds.values()))
    mean = arr.mean(axis=0)

    ax.fill_between(xs, arr.min(axis=0), arr.max(axis=0), color=BAND_C[3],
                    alpha=0.45, zorder=1, lw=0)
    for (lab, v), mk in zip(seeds.items(), ["o", "s", "^"]):
        ax.plot(xs, v, marker=mk, ms=6.5, lw=1.1, color=BAND_C[1], zorder=3,
                mfc="white", mew=1.4)
    ax.plot(xs, mean, lw=2.6, color=BAND_C[0], zorder=4)

    for x, m, n in zip(xs, mean, gt):
        ax.annotate(f"{m:.3f}", (x, m), textcoords="offset points",
                    xytext=(0, 13), ha="center", fontsize=10.5, color=INK)
        ax.annotate(f"n={n:,}", (x, 0.02), ha="center", fontsize=9.0, color=GREY)

    # the arrow runs in the margin left of the first band, clear of its label
    ax.annotate("", xy=(-0.27, mean[0]), xytext=(-0.27, mean[3]),
                arrowprops=dict(arrowstyle="<->", color="#B91C1C", lw=1.6))
    # the empty quadrant is upper-left; a note near the vertical centre would
    # land on the curve
    ax.text(0.30, 0.86,
            f"recall falls by a factor of roughly "
            f"{mean[3]/mean[0]:.0f}\nacross the size range",
            fontsize=10.6, color="#B91C1C", va="center", linespacing=1.5)
    ax.set_xticks(xs); ax.set_xticklabels(labels, fontsize=11)
    ax.set_xlabel("ground-truth target size", fontsize=11)
    ax.set_ylabel("recall", fontsize=11)
    ax.set_ylim(0, 1.06); ax.set_xlim(-0.42, len(labels) - 0.42)
    ax.grid(axis="y", color="#EEEEEE", lw=0.8); ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.text(0.985, 0.06,
            "shaded band spans the three seeds;\nthe collapse reproduces independently in each",
            transform=ax.transAxes, ha="right", va="bottom", fontsize=9.4,
            color=GREY, style="italic", linespacing=1.5)

    p = os.path.join(OUT, "fig_size_collapse.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


def tier23_arms():
    """[(label, mean paired diff, lo, hi, n)] for the Tier 2 and Tier 3 arms,
    computed by the same code and ledgers as analyse_tier2.py and analyse_tier3.py."""
    import analyse_tier2 as t2
    import analyse_tier3 as t3
    arms = []
    b2 = t2.load_bands(os.path.join(RES, "tier2", "BANDS_tier2_plus_baseline.csv"))
    tags = list(b2)
    for arm, label in (("yolov8n-p2", "P2 detection level"),
                       ("yolov8n-cbam", "CBAM attention"),
                       ("yolov8n-p2-cbam", "P2 + CBAM")):
        d = [b2[t2.tag_for(arm, s, tags)]["size:<12"][2]
             - b2[t2.tag_for("yolov8n", s, tags)]["size:<12"][2] for s in SEEDS]
        lo, hi = t2.bc_ci(d)
        arms.append((label, st.mean(d), lo, hi, len(d), "Tier 2"))
    b3 = t3.load_bands([os.path.join(RES, "tier3", "BANDS_tier3.csv"),
                        os.path.join(RES, "tier1_backfill", "BANDS_tier1_backfill.csv")])
    for arm, label in (("fasterrcnn", "Faster R-CNN (two-stage)"),
                       ("rtdetr-l", "RT-DETR-L (transformer)")):
        d = [av - bv for _, av, bv in t3.paired(b3, arm, "size:<12")]
        lo, hi = t3.bc_ci(d)
        arms.append((label, st.mean(d), lo, hi, len(d), "Tier 3"))
    return arms, b3, t3


# --------------------------------------------------------------- figure 4
def fig_interventions():
    """Tier 2 and Tier 3 against the pre-registered floor, on one axis."""
    arms, b3, t3 = tier23_arms()
    base_budget = t3.budget(b3, "yolov8n")[0]
    fig, ax = plt.subplots(figsize=(7.6, 3.2), dpi=200)
    ys = np.arange(len(arms))[::-1]

    ax.axvspan(-0.02, 0.02, color="#F3F4F6", zorder=0)
    ax.axvline(0, color=INK, lw=1.2, zorder=2)
    for x in (-0.02, 0.02):
        ax.axvline(x, color=GREY, lw=1.0, ls="--", zorder=2)
    ax.text(0.021, len(arms) - 0.35, "pre-registered floor,  $\\pm$0.02", fontsize=9.4,
            color=GREY, va="center")

    for y, (name, m, lo, hi, n, tier) in zip(ys, arms):
        if lo is None:
            # fewer than three seeds: the point estimate only, no interval
            ax.plot([m], [y], "o", ms=8, color=GREY, mfc="white", mew=1.4, zorder=4)
            arm = "rtdetr-l" if "RT-DETR" in name else "fasterrcnn"
            ratio = t3.budget(b3, arm)[0] / base_budget
            ax.text(0.995, y, f"{n} seeds, no interval; {ratio:.1f}x box budget",
                    fontsize=9.6, va="center", ha="right", color=GREY,
                    transform=ax.get_yaxis_transform(which="grid"))
            continue
        crosses = lo <= 0 <= hi
        clears = abs(m) > 0.02 and not crosses
        col = "#B91C1C" if (m < 0 and not crosses) else (BAND_C[0] if clears else GREY)
        ax.plot([lo, hi], [y, y], color=col, lw=2.0, solid_capstyle="round", zorder=3)
        ax.plot([lo, lo], [y - .12, y + .12], color=col, lw=1.6, zorder=3)
        ax.plot([hi, hi], [y - .12, y + .12], color=col, lw=1.6, zorder=3)
        ax.plot([m], [y], "o", ms=8, color=col, mfc="white", mew=2.0, zorder=4)
        verdict = ("worse than baseline" if (m < 0 and not crosses)
                   else "no effect" if crosses
                   else "clears the floor")
        # verdicts sit in their own right-hand column, clear of every interval
        ax.text(0.995, y, verdict, fontsize=9.6, va="center", ha="right",
                color=col, transform=ax.get_yaxis_transform(which="grid"))

    ax.set_yticks(ys)
    ax.set_yticklabels([f"{n}   ({t})" for n, _, _, _, _, t in arms], fontsize=10.4)
    ax.set_xlabel("change in sub-12-pixel recall against the seed-paired baseline",
                  fontsize=11)
    ax.set_xlim(-0.045, 0.115)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.grid(axis="x", color="#F3F4F6", lw=0.8); ax.set_axisbelow(True)

    p = os.path.join(OUT, "fig_interventions.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


# --------------------------------------------------------------- figure 5
def fig_box_budget():
    """Why recall alone cannot judge the Tier 3 arms."""
    import analyse_tier3 as t3
    b3 = t3.load_bands([os.path.join(RES, "tier3", "BANDS_tier3.csv"),
                        os.path.join(RES, "tier1_backfill", "BANDS_tier1_backfill.csv")])
    keys = ["yolov8n", "rtdetr-l", "fasterrcnn"]
    arms = ["YOLOv8n\nbaseline", "RT-DETR-L", "Faster\nR-CNN"]
    recall = [st.mean(b3[(k, s)]["size:<12"][2] for s in SEEDS if (k, s) in b3)
              for k in keys]
    budget = [t3.budget(b3, k)[0] for k in keys]
    ungr = [t3.halluc(b3, k)[0] for k in keys]

    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.8), dpi=200,
                             gridspec_kw={"wspace": 0.40})
    xs = np.arange(3)
    for ax, vals, ylab, fmt, note in (
            (axes[0], recall, "recall below 12 px", "{:.4f}", None),
            (axes[1], budget, "boxes emitted per image", "{:.2f}",
             "2.0x guard"),
            (axes[2], ungr, "ungrounded detections per frame", "{:.3f}", None)):
        cols = [BAND_C[0], BAND_C[2], BAND_C[2]]
        ax.bar(xs, vals, width=0.56, color=cols, edgecolor="white", lw=1.0)
        for x, v in zip(xs, vals):
            ax.text(x, v + max(vals) * 0.03, fmt.format(v), ha="center",
                    fontsize=10, color=INK)
        ax.set_xticks(xs); ax.set_xticklabels(arms, fontsize=9.0)
        ax.set_ylabel(ylab, fontsize=10.4)
        ax.set_ylim(0, max(vals) * 1.24)
        ax.spines[["top", "right"]].set_visible(False)
        if note:
            g = budget[0] * 2
            ax.axhline(g, color="#B91C1C", lw=1.3, ls="--")
            # right-anchored above the third bar, the one stretch of the line
            # with no bar or value label above it
            ax.text(2.45, g * 1.03, note, fontsize=9.0, color="#B91C1C",
                    ha="right", va="bottom")

    p = os.path.join(OUT, "fig_box_budget.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  wrote {os.path.basename(p)}")


FIGS = {
    "size": ("corpus", fig_size_distribution),
    "class": ("corpus", fig_class_balance),
    "collapse": (None, fig_size_collapse),
    "interventions": (None, fig_interventions),
    "budget": (None, fig_box_budget),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None, help="one of: " + ", ".join(FIGS))
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    want = [a.only] if a.only else list(FIGS)
    corpus = None
    if any(FIGS[w][0] == "corpus" for w in want):
        print("  reading the released label sets ...")
        corpus = load_corpus()
    for w in want:
        need, fn = FIGS[w]
        fn(corpus) if need else fn()
    return 0


if __name__ == "__main__":
    sys.exit(main())
