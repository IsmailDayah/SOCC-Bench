"""Tier 2 (architecture): seed-paired differences against the pre-registered plan.

    python code/analyse_tier2.py --bands results/protocol/results/tier2/BANDS_tier2_plus_baseline.csv
    python code/analyse_tier2.py --bands ... --out tier2_analysis.md

The analysis follows the plan fixed before any Tier 2 run:

    PRIMARY ENDPOINT   recall in the `< 12 px` size band, Study M test split
    STATISTIC          mean paired difference against the baseline on the same seed
    INTERVAL           bias-corrected bootstrap 95% CI over the seed pairs,
                       10,000 resamples, RNG seed 0
    DECISION           supported only if the CI excludes 0 AND the mean paired
                       difference exceeds +0.02 absolute recall
    EVERYTHING ELSE    secondary and exploratory, labelled as such

Pairing removes seed variance, which dominates at n = 3. mAP is reported elsewhere
for comparability and carries no inferential weight: an aggregate can stay high
while the smallest targets are missed entirely.

993 ground-truth boxes sit below 12 px, so the +0.02 floor is about 20 additional
detections. A missing arm or seed is reported as missing; an arm with fewer than
three seeds is reported with its actual n and no confidence interval. Nothing is
imputed.
"""
import argparse
import collections
import csv
import math
import os
import random
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results", "protocol", "results")

BASELINE = "yolov8n"
ARMS = ["yolov8n-p2", "yolov8n-cbam", "yolov8n-p2-cbam"]
SEEDS = ["42", "123", "456"]
PRIMARY = "size:<12"
FLOOR = 0.02                 # pre-specified, absolute recall
NBOOT = 10000
BANDS_ORDER = ["size:<12", "size:12-20", "size:20-32", "size:32+",
               "occ:OCCLUDED", "occ:LIGHT", "occ:BYPASSED", "occ:AMBIGUOUS"]

HYP = {
    "yolov8n-p2":      ("H1", "P2 RAISES sub-12px recall", "supported"),
    "yolov8n-cbam":    ("H2", "CBAM does NOT move sub-12px recall", "null"),
    "yolov8n-p2-cbam": ("H3", "P2+CBAM tracks P2", "supported"),
}


def load_bands(path):
    """tag -> slice -> (n, found, recall). Tags look like M_<arm>_baseline_s<seed>."""
    if not os.path.exists(path):
        sys.exit(f"missing {path}")
    out = collections.defaultdict(dict)
    for r in csv.DictReader(open(path, encoding="utf-8")):
        try:
            out[r["tag"]][r["slice"]] = (int(r["n"]), int(r["found"]),
                                         float(r["recall"]))
        except (KeyError, ValueError):
            continue
    return out


# The Tier 1 baseline is tagged `M_v8n_baseline_s42`; Tier 2 arms carry their full
# name, `M_yolov8n-p2_baseline_s42`. Substring matching is unsafe here:
# "M_yolov8n-p2_baseline_s42" contains "v8n", so a naive baseline match would
# classify the P2 arm as the baseline and compare it against itself. The tag is
# therefore parsed, and the arm field matched exactly through an alias table.
ALIASES = {"v8n": "yolov8n", "yolov8n": "yolov8n",
           "yolov8n-p2": "yolov8n-p2",
           "yolov8n-cbam": "yolov8n-cbam",
           "yolov8n-p2-cbam": "yolov8n-p2-cbam"}
TAG_RE = re.compile(r"^M_(?P<arm>.+)_(?P<preset>[a-z]+)_s(?P<seed>\d+)$")


def parse_tag(tag):
    """`M_<arm>_baseline_s<seed>` -> (canonical_arm, seed) or (None, None)."""
    m = TAG_RE.match(tag)
    if not m:
        return None, None
    return ALIASES.get(m.group("arm")), m.group("seed")


def tag_for(arm, seed, tags):
    """Exact arm+seed match via parsed tags. Returns None if absent or ambiguous."""
    hits = [t for t in tags if parse_tag(t) == (arm, str(seed))]
    if len(hits) > 1:
        sys.exit(f"AMBIGUOUS: {len(hits)} tags claim arm={arm} seed={seed}: {hits}")
    return hits[0] if hits else None


def bc_ci(vals, nboot=NBOOT, alpha=0.05, rng_seed=0):
    """Bias-corrected bootstrap CI of the mean. Returns (lo, hi) or (None, None)
    when n < 2 - an interval from one observation is not an interval."""
    n = len(vals)
    if n < 2:
        return None, None
    rng = random.Random(rng_seed)
    obs = st.mean(vals)
    boots = []
    for _ in range(nboot):
        s = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(st.mean(s))
    boots.sort()
    # bias correction: the share of bootstrap means that fall below the observed mean
    below = sum(1 for b in boots if b < obs)
    p0 = below / nboot
    if p0 <= 0 or p0 >= 1:                    # degenerate - fall back to percentile
        lo_i = int((alpha / 2) * nboot)
        hi_i = int((1 - alpha / 2) * nboot) - 1
        return boots[lo_i], boots[hi_i]
    z0 = _ppf(p0)
    zl, zh = _ppf(alpha / 2), _ppf(1 - alpha / 2)
    lo_p = _cdf(2 * z0 + zl)
    hi_p = _cdf(2 * z0 + zh)
    lo_i = min(nboot - 1, max(0, int(lo_p * nboot)))
    hi_i = min(nboot - 1, max(0, int(hi_p * nboot)))
    if lo_i > hi_i:
        lo_i, hi_i = hi_i, lo_i
    return boots[lo_i], boots[hi_i]


def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ppf(p):
    """Inverse normal CDF by bisection - avoids a scipy dependency."""
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default=os.path.join(RES, "tier2",
                                                    "BANDS_tier2_plus_baseline.csv"))
    ap.add_argument("--out", default=None,
                    help="also write the report to this markdown file")
    a = ap.parse_args()

    bands = load_bands(a.bands)
    tags = list(bands)
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("# Tier 2 - architecture, judged against the pre-registered plan")
    say()
    say("Primary endpoint: **recall in the < 12 px band**, seed-paired,")
    say(f"bias-corrected bootstrap 95% CI over seed pairs, pre-specified floor **+{FLOOR}**.")
    say()

    # ---- coverage first ------------------------------------------------------
    say("## Coverage")
    say()
    say("| arm | seeds found | missing |")
    say("|---|---|---|")
    have = {}
    for arm in [BASELINE] + ARMS:
        found, miss = [], []
        for s in SEEDS:
            t = tag_for(arm, s, tags)
            if t and PRIMARY in bands[t]:
                found.append(s); have[(arm, s)] = t
            else:
                miss.append(s)
        say(f"| `{arm}` | {', '.join(found) or 'NONE'} | {', '.join(miss) or '-'} |")
    say()

    base_ok = [s for s in SEEDS if (BASELINE, s) in have]
    if not base_ok:
        say("**Cannot analyse: no baseline seeds present.** The ablation is a paired")
        say("comparison; without the baseline there is nothing to pair against.")
        _write(a.out, L); return

    # ---- primary endpoint ---------------------------------------------------
    say(f"## Primary endpoint - recall at {PRIMARY}")
    say()
    say("| arm | n pairs | baseline | arm | mean paired diff | 95% CI | verdict |")
    say("|---|---|---|---|---|---|---|")
    verdicts = {}
    diffs = {}
    for arm in ARMS:
        pairs, b_vals, a_vals = [], [], []
        for s in SEEDS:
            if (arm, s) in have and (BASELINE, s) in have:
                bv = bands[have[(BASELINE, s)]][PRIMARY][2]
                av = bands[have[(arm, s)]][PRIMARY][2]
                pairs.append(av - bv); b_vals.append(bv); a_vals.append(av)
        if not pairs:
            say(f"| `{arm}` | 0 | - | - | - | - | **NOT RUN** |")
            verdicts[arm] = "not run"
            continue
        d = st.mean(pairs)
        lo, hi = bc_ci(pairs) if len(pairs) >= 3 else (None, None)
        if lo is None:
            ci = "no interval"
            v = f"**UNDERPOWERED (n={len(pairs)})**"
        elif lo > 0 and d > FLOOR:
            ci = f"[{lo:+.4f}, {hi:+.4f}]"; v = "**SUPPORTED**"
        elif hi < 0:
            ci = f"[{lo:+.4f}, {hi:+.4f}]"; v = "**REFUTED (worse)**"
        elif lo > 0:
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
            v = f"detectable but **below the +{FLOOR} floor**"
        else:
            ci = f"[{lo:+.4f}, {hi:+.4f}]"; v = "**NULL** (CI includes 0)"
        verdicts[arm] = v
        diffs[arm] = d
        say(f"| `{arm}` | {len(pairs)} | {st.mean(b_vals):.4f} | {st.mean(a_vals):.4f} "
            f"| {d:+.4f} | {ci} | {v} |")
    say()

    # ---- hypotheses, judged against the pre-registration --------------------
    say("## Hypotheses, judged against the pre-registration")
    say()
    for arm in ARMS:
        h, text, expect = HYP[arm]
        got = verdicts.get(arm, "not run")
        # An arm that was not run, or has too few seeds, is reported as such and
        # never as a refutation: an unknown is not a negative result.
        if "NOT RUN" in got.upper() or "UNDERPOWERED" in got.upper():
            say(f"- **{h}** - {text}. Predicted: *{expect}*. "
                f"**NOT MEASURED** ({got}) - no conclusion is drawn.")
            continue
        # H3 compares two arms with each other, not one arm with the baseline:
        # "P2+CBAM tracks P2" holds when both arms land on the same side of the
        # floor and their difference from each other is itself inside the floor.
        if arm == "yolov8n-p2-cbam" and "yolov8n-p2" in diffs and arm in diffs:
            d_p2, d_both = diffs["yolov8n-p2"], diffs[arm]
            gap = abs(d_both - d_p2)
            tracks = gap < FLOOR and (d_p2 > FLOOR) == (d_both > FLOOR)
            say(f"- **{h}** - {text}. Predicted: *{expect}*. "
                f"P2 {d_p2:+.4f} vs P2+CBAM {d_both:+.4f}, gap {gap:.4f} "
                f"(floor {FLOOR}). "
                f"**{'As predicted - P2+CBAM tracks P2' if tracks else 'CONTRADICTED - the arms diverge'}.**")
            continue
        met = ("SUPPORTED" in got) if expect == "supported" \
            else ("NULL" in got or "floor" in got)
        say(f"- **{h}** - {text}. Predicted: *{expect}*. Observed: {got}. "
            f"**{'As predicted' if met else 'CONTRADICTED'}.**")
    say()

    # ---- secondary, explicitly exploratory ----------------------------------
    say("## Secondary (exploratory - no inferential weight)")
    say()
    say("| band | " + " | ".join(f"`{x}`" for x in [BASELINE] + ARMS) + " |")
    say("|---|" + "---|" * (len(ARMS) + 1))
    for sl in BANDS_ORDER:
        row = [sl]
        for arm in [BASELINE] + ARMS:
            vs = [bands[have[(arm, s)]][sl][2] for s in SEEDS
                  if (arm, s) in have and sl in bands[have[(arm, s)]]]
            row.append(f"{st.mean(vs):.4f}" if vs else "-")
        say("| " + " | ".join(row) + " |")
    say()
    say("The occlusion and size axes are **confounded by venue**: occlusion was")
    say("staged where targets are larger, and the extreme small tail belongs to a venue")
    say("with no occluder. No claim ranks the two axes against each other.")

    _write(a.out, L)


def _write(path, lines):
    if not path:
        return
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
