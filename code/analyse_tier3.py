"""Tier 3 (detection paradigm): H4, H5 and H6, judged against the pre-registered plan.

    python code/analyse_tier3.py
    python code/analyse_tier3.py --bands <BANDS.csv> --bands <BANDS.csv> [--out report.txt]

H4, H5 and H6 were fixed before any Tier 3 run. This script computes them and
nothing besides them; every threshold below is the pre-registered one.

Rules shared with analyse_tier2.py:
  1. An absent arm is reported as not measured, never as a refutation.
  2. Run tags are parsed, never substring-matched: "M_yolov8n-p2_baseline_s42"
     contains "yolov8n", so `arm in tag` would compare P2 against itself.
  3. A difference below the +0.02 floor (about 20 of 993 boxes) is no effect.
  4. An arm with fewer than three seeds is underpowered and gets no interval.

And one rule this tier adds:
  5. Recall is blind to false positives, so it can be won by emitting more boxes.
     RT-DETR applies no NMS - it emits a fixed set of queries filtered by score -
     so a cross-paradigm recall comparison carries the box budget with it, and an
     arm that beats the baseline while emitting more than 2x its boxes per image
     is reported as confounded, not as better.
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

# ---- pre-registered values ---------------------------------------------------
BASELINE = "yolov8n"                 # the Tier 1 comparator, all three seeds
ARMS = ["rtdetr-l", "fasterrcnn"]
SEEDS = ["42", "123", "456"]
SMALL = "size:<12"                   # primary endpoint
OCCL = "occ:OCCLUDED"                # H5's second axis
FLOOR = 0.02                         # absolute recall; ~20 boxes of 993
H4_MAX = 0.20                        # "recovers < 0.20 in the <12 px band"
NBOOT = 10000
RNG = 0
MIN_SEEDS = 3                        # fewer -> underpowered, no interval
BUDGET_RATIO = 2.0                   # box-budget guard, fixed in advance

ALIASES = {"v8n": "yolov8n", "yolov8n": "yolov8n",
           "yolov8n-p2": "yolov8n-p2", "yolov8n-cbam": "yolov8n-cbam",
           "yolov8n-p2-cbam": "yolov8n-p2-cbam",
           "rtdetr-l": "rtdetr-l", "rtdetr": "rtdetr-l",
           "fasterrcnn": "fasterrcnn", "faster-rcnn": "fasterrcnn"}
TAG_RE = re.compile(r"^M_(?P<arm>.+)_(?P<preset>[a-z]+)_s(?P<seed>\d+)$")


def parse_tag(tag):
    """`M_<arm>_<preset>_s<seed>` -> (canonical_arm, seed), else (None, None)."""
    m = TAG_RE.match(tag.strip())
    if not m:
        return None, None
    return ALIASES.get(m.group("arm")), m.group("seed")


def load_bands(paths):
    """{(arm, seed): {slice: (n, found, recall)}} merged over several CSVs.

    A duplicate (arm, seed, slice) across files is fatal: a run scored twice would
    otherwise be resolved silently by file order."""
    out = collections.defaultdict(dict)
    seen = {}
    for p in paths:
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8")):
            tag = (r.get("tag") or "").strip()
            arm, seed = parse_tag(tag)
            if not arm:
                continue
            sl = (r.get("slice") or "").strip()
            key = (arm, seed, sl)
            if key in seen and seen[key] != p:
                sys.exit(f"DUPLICATE {key} in both {seen[key]} and {p}\n"
                         "  refusing to guess which scoring run is authoritative")
            seen[key] = p
            try:
                n = int(r["n"]); found = int(r["found"])
                rec = float(r["recall"])
            except (ValueError, KeyError, TypeError):
                continue
            out[(arm, seed)][sl] = (n, found, rec)
    return out


def bc_ci(vals, nboot=NBOOT, alpha=0.05, rng_seed=RNG):
    """Bias-corrected bootstrap CI of the mean; (None, None) below MIN_SEEDS."""
    n = len(vals)
    if n < MIN_SEEDS:
        return None, None
    rng = random.Random(rng_seed)
    obs = st.mean(vals)
    boots = []
    for _ in range(nboot):
        s = [vals[rng.randrange(n)] for _ in range(n)]
        boots.append(st.mean(s))
    boots.sort()
    n_less = sum(1 for b in boots if b < obs)
    if n_less in (0, len(boots)):
        return boots[0], boots[-1]
    z0 = _ppf(n_less / len(boots))
    za = _ppf(alpha / 2)
    lo_p = _cdf(2 * z0 + za)
    hi_p = _cdf(2 * z0 - za)
    lo = boots[min(len(boots) - 1, max(0, int(lo_p * len(boots))))]
    hi = boots[min(len(boots) - 1, max(0, int(hi_p * len(boots))))]
    return lo, hi


def _cdf(x):
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))


def _ppf(p):
    if p <= 0:
        return -8.0
    if p >= 1:
        return 8.0
    lo, hi = -8.0, 8.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if _cdf(mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def paired(bands, arm, slice_name):
    """Seed-paired (arm, baseline) recalls. Only seeds where BOTH exist."""
    pairs = []
    for s in SEEDS:
        a = bands.get((arm, s), {}).get(slice_name)
        b = bands.get((BASELINE, s), {}).get(slice_name)
        if a and b:
            pairs.append((s, a[2], b[2]))
    return pairs


def budget(bands, arm):
    """Mean predictions/image over available seeds, or None if unmeasured."""
    vals = [bands[(arm, s)]["pred_budget"][2]
            for s in SEEDS
            if "pred_budget" in bands.get((arm, s), {})]
    return (st.mean(vals), len(vals)) if vals else (None, 0)


def halluc(bands, arm):
    """Mean ungrounded detections per 100%-occluded frame, or None."""
    vals = [bands[(arm, s)]["halluc_fullyocc"][2]
            for s in SEEDS
            if "halluc_fullyocc" in bands.get((arm, s), {})]
    return (st.mean(vals), len(vals)) if vals else (None, 0)


def fmt_ci(lo, hi, n):
    return f"no interval (n={n})" if lo is None else f"[{lo:+.4f}, {hi:+.4f}]"


def confound_note(bands, arm):
    """Lines printed whenever an arm beats the baseline on recall. Never silent:
    if either budget is unmeasured, the check is reported as impossible."""
    ab, an = budget(bands, arm)
    bb, bn = budget(bands, BASELINE)
    if ab is None or bb is None:
        miss = " and ".join(
            [x for x, v in ((arm, ab), (BASELINE, bb)) if v is None])
        return [f"  *** CONFOUND CHECK IMPOSSIBLE: no pred_budget row for {miss}.",
                "      Recall is blind to false positives, so a gain cannot be",
                "      attributed to detection quality without the box budget.",
                "      Re-score that arm with score_bands.py, which writes",
                "      pred_budget. Until then this result is UNVERIFIED."]
    if ab > bb * BUDGET_RATIO:
        return [f"  *** CONFOUNDED BY BOX BUDGET: {arm} emits {ab:.2f} boxes/image",
                f"      against the baseline's {bb:.2f} ({ab/bb:.1f}x, threshold "
                f"{BUDGET_RATIO:.1f}x).",
                "      Report as confounded, not as a win."]
    return [f"  budget check OK: {arm} {ab:.2f} vs baseline {bb:.2f} boxes/image "
            f"({ab/bb:.1f}x, under the {BUDGET_RATIO:.1f}x threshold)"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", action="append", default=[],
                    help="BANDS csv (repeatable). Must include the Tier 1 "
                         "baseline rows or there is nothing to compare against.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    paths = a.bands or [os.path.join(RES, "tier3", "BANDS_tier3.csv"),
                        os.path.join(RES, "tier1_backfill", "BANDS_tier1_backfill.csv")]
    bands = load_bands(paths)
    L = []

    def say(s=""):
        print(s)
        L.append(s)

    say("=" * 74)
    say("TIER 3 - detection paradigm, judged against the pre-registered plan")
    say("=" * 74)
    say(f"  sources: {', '.join(os.path.relpath(p, ROOT) for p in paths)}")

    base_seeds = [s for s in SEEDS if (BASELINE, s) in bands]
    say(f"  baseline {BASELINE}: seeds {base_seeds or 'NONE'}")
    if not base_seeds:
        say("\n  NO BASELINE ROWS. Every hypothesis is a comparison against the")
        say("  Tier 1 baseline, so nothing can be judged. Add the Tier 1 bands")
        say("  and re-run.")
        return _write(a.out, L)
    for arm in ARMS:
        got = [s for s in SEEDS if (arm, s) in bands]
        say(f"  {arm:<12}: seeds {got or 'NONE - NOT MEASURED'}")

    bb, bn = budget(bands, BASELINE)
    say(f"\n  baseline box budget: "
        f"{'%.2f/image (%d seeds)' % (bb, bn) if bb else 'NOT MEASURED'}")

    # ---- H4 ----------------------------------------------------------------
    say("\n" + "-" * 74)
    say("H4 (primary): the sub-12px collapse is PARADIGM-INVARIANT")
    say(f"  pre-registered: every Tier 3 arm recovers < {H4_MAX:.2f} in {SMALL}")
    say("-" * 74)
    h4 = {}
    thin = []
    for arm in ARMS:
        vals = [(s, bands[(arm, s)][SMALL][2]) for s in SEEDS
                if SMALL in bands.get((arm, s), {})]
        if not vals:
            say(f"  {arm:<12} NOT MEASURED")
            h4[arm] = None
            continue
        rs = [v for _, v in vals]
        m = st.mean(rs)
        lo, hi = bc_ci(rs)
        per = "  ".join(f"s{s}={v:.4f}" for s, v in vals)
        say(f"  {arm:<12} mean {m:.4f}  n={len(rs)}  {per}")
        if lo is not None:
            say(f"               95% CI [{lo:.4f}, {hi:.4f}]")
        else:
            thin.append(f"{arm} on {len(rs)} of {len(SEEDS)} seeds")
            say(f"               n<{MIN_SEEDS} - UNDERPOWERED, reported without an interval")
        h4[arm] = m < H4_MAX
        say(f"               {'below' if h4[arm] else 'ABOVE'} the {H4_MAX:.2f} "
            f"threshold -> {'consistent with H4' if h4[arm] else 'H4 FALSIFIED'}")
        if not h4[arm]:
            # an arm above the threshold claims to find small objects the others
            # miss - the claim most exposed to the box budget
            for ln in confound_note(bands, arm):
                say(ln)
    done = [v for v in h4.values() if v is not None]
    if not done:
        say("\n  VERDICT H4: NOT MEASURED")
    elif len(done) < len(ARMS):
        say(f"\n  VERDICT H4: PARTIAL - {len(done)}/{len(ARMS)} arms measured, "
            "no verdict until both are in")
    else:
        note = f" ({'; '.join(thin)})" if thin else ""
        say(f"\n  VERDICT H4: {'SUPPORTED' if all(done) else 'FALSIFIED'}{note}")

    # ---- H5 ----------------------------------------------------------------
    say("\n" + "-" * 74)
    say("H5 (primary): the two axes favour OPPOSITE architectures")
    say("  pre-registered: Faster R-CNN is WORSE than baseline in <12px")
    say("                  and BETTER than baseline in OCCLUDED")
    say("-" * 74)
    h5 = {}
    for slice_name, want, label in ((SMALL, "worse", "size <12px"),
                                    (OCCL, "better", "OCCLUDED")):
        pr = paired(bands, "fasterrcnn", slice_name)
        if not pr:
            say(f"  {label:<12} NOT MEASURED")
            h5[slice_name] = None
            continue
        d = [arm_v - base_v for _, arm_v, base_v in pr]
        m = st.mean(d)
        lo, hi = bc_ci(d)
        say(f"  {label:<12} n={len(d)} paired seeds")
        for s, av, bv in pr:
            say(f"     s{s}: frcnn {av:.4f}  baseline {bv:.4f}  diff {av-bv:+.4f}")
        say(f"     mean diff {m:+.4f}   95% CI {fmt_ci(lo, hi, len(d))}")
        if len(d) < MIN_SEEDS:
            say(f"     UNDERPOWERED - no verdict on this axis")
            h5[slice_name] = None
        elif abs(m) < FLOOR:
            say(f"     |diff| < floor {FLOOR} -> NO EFFECT (direction not called)")
            h5[slice_name] = False
        else:
            got = "better" if m > 0 else "worse"
            h5[slice_name] = (got == want)
            say(f"     {got} than baseline; predicted {want} -> "
                f"{'as predicted' if h5[slice_name] else 'SIGN REVERSED'}")
            if got == "better":
                for ln in confound_note(bands, "fasterrcnn"):
                    say(ln)
    vals5 = [v for v in h5.values() if v is not None]
    if len(vals5) < 2:
        say("\n  VERDICT H5: NOT MEASURED (needs both axes)")
    else:
        say(f"\n  VERDICT H5: {'SUPPORTED' if all(vals5) else 'FALSIFIED'}"
            " - H5 is falsified if EITHER prediction fails")

    # ---- H6 ----------------------------------------------------------------
    say("\n" + "-" * 74)
    say("H6 (secondary): transformer attention does not rescue small objects")
    say(f"  pre-registered: RT-DETR-L does not exceed baseline in {SMALL} by >{FLOOR}")
    say("-" * 74)
    pr = paired(bands, "rtdetr-l", SMALL)
    if not pr:
        say("  NOT MEASURED")
    else:
        d = [av - bv for _, av, bv in pr]
        m = st.mean(d)
        lo, hi = bc_ci(d)
        for s, av, bv in pr:
            say(f"     s{s}: rtdetr {av:.4f}  baseline {bv:.4f}  diff {av-bv:+.4f}")
        say(f"     mean diff {m:+.4f}   95% CI {fmt_ci(lo, hi, len(d))}")
        if len(d) < MIN_SEEDS:
            say(f"\n  VERDICT H6: UNDERPOWERED - {len(d)} of {len(SEEDS)} seeds, "
                "no verdict")
            if m > FLOOR:
                for ln in confound_note(bands, "rtdetr-l"):
                    say(ln)
        else:
            exceeds = m > FLOOR
            say(f"\n  VERDICT H6: {'FALSIFIED' if exceeds else 'SUPPORTED'}"
                f" ({'exceeds' if exceeds else 'does not exceed'} the floor)")
            if exceeds:
                for ln in confound_note(bands, "rtdetr-l"):
                    say(ln)

    # ---- the box budget and the ungrounded-detection probe ------------------
    say("\n" + "-" * 74)
    say("BOX BUDGET AND UNGROUNDED DETECTIONS (reported for every arm)")
    say("  Recall cannot see a false positive. These two can.")
    say("-" * 74)
    say(f"  {'arm':<12}{'boxes/image':>14}{'ungrounded/frame':>20}")
    for arm in [BASELINE] + ARMS:
        b, nb = budget(bands, arm)
        h, nh = halluc(bands, arm)
        bs = f"{b:.2f} (n={nb})" if b is not None else "not measured"
        hs = f"{h:.3f} (n={nh})" if h is not None else "not measured"
        say(f"  {arm:<12}{bs:>14}{hs:>20}")
    say("\n  'ungrounded' = detections fired on the 159 test frames where the target")
    say("  is present but 100% occluded, and therefore unlabelled. Those frames")
    say("  hold no ground-truth box, so they cannot appear in any recall band; a")
    say("  detection there has no visible evidence behind it.")
    say("\n" + "=" * 74)
    return _write(a.out, L)


def _write(path, lines):
    if path:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        open(path, "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"\n  wrote {path}")


if __name__ == "__main__":
    main()
