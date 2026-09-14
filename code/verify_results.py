"""Re-derive every number quoted in README.md from the files in this repository.

    python code/verify_results.py

Each check computes a value from the labels, manifests, flight logs or result
ledgers, formats it the way README.md prints it, and confirms that exact text
appears in README.md. A number that is not re-derivable is a claim, not a result.
Runs from a fresh clone; the imagery is not needed.
"""
import collections
import csv
import glob
import json
import math
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
import analyse_tier2 as t2  # noqa: E402
import analyse_tier3 as t3  # noqa: E402
import analyse_tier4 as t4  # noqa: E402
from tile_probe import tiles_for  # noqa: E402

# README text contains typographic characters; never let a legacy console
# encoding turn a verification run into a crash
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RES = os.path.join(ROOT, "results", "protocol", "results")
README = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
FLAT = re.sub(r"\s+", " ", README)
W, H = 960, 720
SEEDS = ["42", "123", "456"]
SESS = {"S1": "dataset/01_Dataset_SOCC", "S2": "dataset/01_Dataset_SOCC_S2",
        "S3": "dataset/01_Dataset_SOCC_S3", "S4": "dataset/01_Dataset_SOCC_S4"}
MAN = {"S1": "socc_frames_manifest.csv", "S2": "socc_s2_frames_manifest.csv",
       "S3": "socc_s3_frames_manifest.csv", "S4": "socc_s4_frames_manifest.csv"}
NAMES = ["rc_car", "atv_sherp", "truck_peterbilt"]

OK = FAIL = 0


def chk(label, text, ok=True):
    """`text` must appear in README.md (whitespace-insensitive) and `ok` must hold."""
    global OK, FAIL
    found = re.sub(r"\s+", " ", text) in FLAT
    good = found and bool(ok)
    why = "" if good else ("  <- not in README" if not found else "  <- data disagrees")
    print(f"  {'ok  ' if good else 'FAIL'}  {label:<58} {text!r}{why}")
    if good:
        OK += 1
    else:
        FAIL += 1


def eq(label, text, computed):
    """README text must appear, and equal the computed, formatted value."""
    chk(label, text, text == computed or (isinstance(computed, str) and computed in text))
    if text != computed and not (isinstance(computed, str) and computed in text):
        print(f"        computed: {computed!r}")


def rows(path):
    return list(csv.DictReader(open(os.path.join(ROOT, path), encoding="utf-8")))


def fmt(n):
    return f"{n:,}"


def sqrt_area(q):
    return ((float(q[3]) * W) * (float(q[4]) * H)) ** 0.5


def main():
    # ================================================================ corpus
    print("-- corpus --")
    labels, per = {}, {}
    for s, d in SESS.items():
        files = glob.glob(os.path.join(ROOT, d, "release", "labels", "*.txt"))
        boxes = []
        for p in files:
            q = [l.split() for l in open(p) if len(l.split()) == 5]
            labels[os.path.basename(p)[:-4] + ".jpg"] = (s, q)
            boxes += q
        sizes = [sqrt_area(q) for q in boxes]
        per[s] = dict(frames=len(files), boxes=len(boxes), sizes=sizes,
                      empty=sum(1 for p in files if not labels[os.path.basename(p)[:-4] + ".jpg"][1]),
                      cls=collections.Counter(NAMES[int(q[0])] for q in boxes))
    venue = {"S1": "rooftop futsal court", "S2": "same court, graded occluder rig",
             "S3": "covered walkway", "S4": "park deck"}
    heights = {}
    for s, d in SESS.items():
        rel = {k for k, v in labels.items() if v[0] == s}
        t = []
        for r in rows(os.path.join(d, MAN[s])):
            img = f"{r['stem']}_f{int(r['frame_idx']):06d}.jpg" if s == "S1" else r["image"]
            if img in rel:
                v = float(r["tof_cm"] or 0)
                if 20 < v < 2000:
                    t.append(v)
        t.sort()
        heights[s] = (st.median(t) / 100, t[len(t) * 19 // 20] / 100)
    classes = {"S1": "car", "S2": "car", "S3": "all three", "S4": "all three"}
    for s in SESS:
        p = per[s]
        present = "car" if set(p["cls"]) == {"rc_car"} else "all three" if len(p["cls"]) == 3 else "?"
        row = (f"| {s} | {venue[s]} | {fmt(p['frames'])} | {fmt(p['boxes'])} | {present} | "
               f"{st.median(p['sizes']):.1f} px | {fmt(sum(x < 12 for x in p['sizes']))} | "
               f"{heights[s][0]:.1f} m ({heights[s][1]:.1f} m) |")
        chk(f"session table row {s}", row, present == classes[s])
    allsz = [x for p in per.values() for x in p["sizes"]]
    nfr = sum(p["frames"] for p in per.values())
    total = (f"| **Total** | | **{fmt(nfr)}** | **{fmt(len(allsz))}** | | "
             f"**{st.median(allsz):.1f} px** | **{fmt(sum(x < 12 for x in allsz))}** | |")
    chk("session table total", total)
    chk("badge frames", f"released%20frames-{nfr:,}".replace(",", "%2C"))
    chk("badge instances", f"instances-{len(allsz):,}".replace(",", "%2C"))
    chk("corpus line", f"**{fmt(nfr)} released frames · {fmt(len(allsz))} instances")
    cls = sum((p["cls"] for p in per.values()), collections.Counter())
    print(f"        classes: {dict(cls)}")
    chk("largest box under 100 px", "from nearly 100 px down to under 3 px",
        90 < max(allsz) < 100 and min(allsz) < 3)
    n32, n12 = sum(x < 32 for x in allsz), sum(x < 12 for x in allsz)
    eq("share below 32 px", f"{100 * n32 / len(allsz):.1f} % of all instances are COCO-small",
       f"{100 * n32 / len(allsz):.1f} % of all instances are COCO-small")
    eq("share below 12 px", f"{100 * n12 / len(allsz):.1f} % fall below 12 px",
       f"{100 * n12 / len(allsz):.1f} % fall below 12 px")
    s4tail = sum(x < 12 for x in per["S4"]["sizes"])
    chk("park deck holds the tail", f"holds {fmt(s4tail)} of the {fmt(n12)} sub-12-pixel instances")
    chk("frames without a box", f"{fmt(sum(p['empty'] for p in per.values()))} frames carry no box")

    a4 = rows("results/protocol/study_M4/assignment.csv")
    aM = rows("results/protocol/study_M/assignment.csv")
    withheld = [r for r in a4 if r["image"] not in labels]
    by = collections.Counter(r["scene"] for r in withheld)
    chk("withheld frames", f"34 further frames ({by['S1']} in S1, {by['S2']} in S2)",
        len(withheld) == 34 and sum(int(r["boxes"]) for r in withheld) == 0)
    chk("withheld frames in test", f"{sum(r['split'] == 'test' for r in withheld)} of them in test")

    recs = sorted({os.path.basename(p).split(".")[0] for p in glob.glob(os.path.join(ROOT, "capture", "captures", "*.csv"))})
    chk("flight recordings", f"logs of all {len(recs)} recordings")
    rates, nopr = [], set()
    for p in glob.glob(os.path.join(ROOT, "capture", "captures", "*.telemetry.csv")):
        rr = list(csv.DictReader(open(p, encoding="utf-8")))
        tt = [float(r["t_rel"]) for r in rr]
        rates.append(1 / st.median([b - a for a, b in zip(tt, tt[1:]) if b > a]))
        if "pitch_deg" not in rr[0]:
            nopr.add(os.path.basename(p)[:2])
    chk("telemetry rate", "about 9 Hz (pitch and roll as well from S2 on)",
        all(8.5 < r < 9.6 for r in rates) and nopr == {"S1"})
    fps, live_n, live_s = [], 0, 0.0
    for stem in sorted({r["stem"] for r in rows(os.path.join(SESS["S4"], MAN["S4"])) if r["capture_class"] == "benchmark"}):
        tt = [float(r["t_rel"]) for r in rows(os.path.join("capture", "captures", f"{stem}.frames.csv"))]
        gaps = [b - a for a, b in zip(tt, tt[1:])]
        n, live = sum(1 for g in gaps if g <= 0.30), sum(g for g in gaps if g <= 0.30)
        fps.append(n / live)
        live_n += n
        live_s += live
    chk("S4 delivered frame rate", f"recorder delivered {min(fps):.1f}\u2013{max(fps):.1f} fps", len(fps) == 7)
    chk("S4 exposures lost", f"with about {100 * (1 - (live_n / live_s) / 30):.0f} % of exposures lost")
    s4m = rows(os.path.join(SESS["S4"], MAN["S4"]))
    chk("S4 resample", f"its {fmt(per['S4']['frames'])} frames were selected from {fmt(len(s4m))} extracted ones")
    prov = collections.Counter(r["review_provenance"] for r in rows(os.path.join(SESS["S3"], "release", "socc_s3_release_manifest.csv")))
    chk("S3 review provenance", f"proposals on {fmt(prov['model_accepted_bulk'])} frames were accepted in bulk and {fmt(prov['human_confirmed'])} frames")
    inst = rows(os.path.join(SESS["S1"], "release", "socc_instances.csv"))
    cards = {r["flight"] for r in inst if r["role"] not in ("C3_occ_setup_stills", "C5_negatives")}
    est = sorted(float(r["est_length_mm_at_mat_plane"]) for r in inst
                 if r["flight"] in cards and r["est_length_mm_at_mat_plane"])
    chk("marker-scale calibration", f"reads a median of {st.median(est):.0f} mm over the {len(est)} frames of the S1 flight cards",
        len(cards) == 8)
    chk("marker-scale bias", f"{100 * (st.median(est) / 216 - 1):.0f} % long, because the car sits nearer the camera")
    chk("S1 backpack instances", f"natural occluder ({sum(r['occluder'] == 'object' for r in inst)} instances)")

    # ============================================================= occlusion
    print("\n-- occlusion --")
    m2 = rows(os.path.join(SESS["S2"], MAN["S2"]))
    m3 = rows(os.path.join(SESS["S3"], MAN["S3"]))
    strips2 = sorted({float(r["occ_strip_mm"]) for r in m2 if r["occ_strip_mm"] not in ("", "0")})
    strips3 = sorted({float(r["occ_strip_mm"]) for r in m3 if r["occ_strip_mm"] not in ("", "0", "0.0")})
    chk("S2 strips", "54, 108 or 162 mm", strips2 == [54.0, 108.0, 162.0])
    chk("S3 strips", "37.5, 75 and 112.5 mm", strips3 == [37.5, 75.0, 112.5])
    rel2 = rows(os.path.join(SESS["S2"], "release", "socc_s2_release_manifest.csv"))
    band = collections.Counter((r["occ_deployed_pct"], r["occ_realised_band"]) for r in rel2
                               if r["flight_key"] in ("S2_PAIR_occ25", "S2_PAIR_occ50", "S2_PAIR_occ75"))
    for lvl in ("25", "50", "75"):
        chk(f"realised bands at {lvl} %",
            f"| {lvl} % | {band[(lvl, 'heavy')]} | {band[(lvl, 'partial')]} | {band[(lvl, 'indeterminate')]} |")
    n75 = sum(v for (l, b), v in band.items() if l == "75")
    chk("heavy share at 75 %", f"only {band[('75', 'heavy')]} of {n75} frames measure as heavily occluded")
    full1 = rows(os.path.join(SESS["S1"], "release", "FULLY_OCCLUDED_S1.csv"))
    runs1 = sorted(collections.Counter(r["run_start"] for r in full1).values(), reverse=True)
    full2 = [r for r in rel2 if r["fully_occluded"] == "1"]
    chk("fully hidden S1", f"{len(full1)} in S1, in two unbroken runs of {runs1[0]} and {runs1[1]} frames",
        len(runs1) == 2)
    chk("fully hidden S2", f"{len(full2)} in S2, {sum(r['occ_deployed_pct'] == '75' for r in full2)} of them on the 75 % lap")
    fullset = {r["image"] for r in full1} | {r["image"] for r in full2}
    split4 = {r["image"]: r["split"] for r in a4}
    chk("all fully hidden frames in test", f"{len(fullset)} frames show the target's position",
        all(split4.get(x) == "test" for x in fullset))
    lab2 = {k: v[1] for k, v in labels.items() if v[0] == "S2"}
    lv = collections.defaultdict(list)
    for r in rel2:
        q = lab2.get(r["image"], [])
        if r["flight_key"] in ("S2_PAIR_clean", "S2_PAIR_occ25", "S2_PAIR_occ50", "S2_PAIR_occ75") and len(q) == 1:
            lv[r["flight_key"]].append(sqrt_area(q[0]))
    clean = sorted(lv["S2_PAIR_clean"])
    pos = 0.05 * (len(clean) - 1)
    p5 = clean[int(pos)] + (clean[min(int(pos) + 1, len(clean) - 1)] - clean[int(pos)]) * (pos - int(pos))
    share = {k: 100 * sum(x < p5 for x in lv[k]) / len(lv[k]) for k in ("S2_PAIR_occ25", "S2_PAIR_occ50", "S2_PAIR_occ75")}
    chk("extinction continuum",
        f"clean lap's 5th percentile ({p5:.1f} px) rises from {share['S2_PAIR_occ25']:.1f} % at 25 % deployed "
        f"to {share['S2_PAIR_occ50']:.1f} % at 50 % and {share['S2_PAIR_occ75']:.1f} % at 75 %, where the smallest box "
        f"is {min(lv['S2_PAIR_occ75']):.1f} px")

    # ================================================================= split
    print("\n-- split --")
    sp = collections.Counter(r["split"] for r in a4)
    bx = collections.Counter()
    for r in a4:
        bx[r["split"]] += int(r["boxes"])
    for s, name in (("test", "test"), ("val", "validation"), ("train", "train")):
        chk(f"M4 {s} counts", f"| {fmt(sp[s])} | {fmt(bx[s])} |")
    testM = {r["image"] for r in aM if r["split"] == "test"}
    test4 = {r["image"] for r in a4 if r["split"] == "test"}
    chk("identical test split", "Two partitions share this **identical test split**", testM == test4)
    b1 = rows("results/protocol/results/BANDS.csv")
    gt = [int(next(r["n"] for r in b1 if r["slice"] == f"size:{b}")) for b in ("<12", "12-20", "20-32", "32+")]
    chk("test boxes per band", f"The test split holds {' / '.join(fmt(x) for x in gt)} boxes", sum(gt) == bx["test"])
    occn = int(next(r["n"] for r in b1 if r["slice"] == "occ:OCCLUDED"))
    chk("OCCLUDED test boxes", f"{occn} OCCLUDED boxes")

    def small_share(assign, split):
        tot = small = 0
        for r in assign:
            if r["split"] != split or r["image"] not in labels:
                continue
            for q in labels[r["image"]][1]:
                tot += 1
                small += sqrt_area(q) < 12
        return small, tot
    vm, _ = small_share(aM, "val")
    v4, _ = small_share(a4, "val")
    trs, trt = small_share(aM, "train")
    tes, tet = small_share(aM, "test")
    chk("Study M validation sub-12 boxes", f"held only **{vm}** sub-12-pixel boxes")
    chk("Study M4 validation sub-12 boxes", f"raising that to **{v4}**")
    chk("train share (Study M)", f"is **{100 * trs / trt:.1f} %** sub-12-pixel boxes")
    chk("test share", f"the test split is **{100 * tes / tet:.1f} %**")
    chk("validation share", f"holds six such boxes (**{100 * vm / small_share(aM, 'val')[1]:.1f} %**)", vm == 6)
    ground = 0
    for s, d in SESS.items():
        rel = {k for k, v in labels.items() if v[0] == s}
        for r in rows(os.path.join(d, MAN[s])):
            img = f"{r['stem']}_f{int(r['frame_idx']):06d}.jpg" if s == "S1" else r["image"]
            if img in rel and img not in split4 and 0 < float(r["tof_cm"] or 0) <= 20:
                ground += 1
    unexplained = len(set(labels) - set(split4)) - ground
    chk("on-ground frames left out", f"leaves {fmt(ground)} released frames outside the partition", unexplained == 0)
    kept = sum(1 for r in a4 if r["split"] == "test" and 0 < float(r["tof_cm"] or 0) <= 20)
    chk("on-ground frames kept in test", f"the test families keep theirs ({kept} frames)",
        all(not (0 < float(r["tof_cm"] or 0) <= 20) for r in a4 if r["split"] != "test"))

    # ================================================================ tier 1
    print("\n-- Tier 1 --")
    bb = t2.load_bands(os.path.join(RES, "BANDS.csv"))
    for b, lab in (("size:<12", "< 12 px"), ("size:12-20", "12–20 px"), ("size:20-32", "20–32 px"), ("size:32+", "32+ px")):
        v = [bb[f"M_v8n_baseline_s{s}"][b][2] for s in SEEDS]
        mean = f"{st.mean(v):.4f}"
        if b in ("size:<12", "size:32+"):
            mean = f"**{mean}**"
        chk(f"Tier 1 row {lab}", f"| {lab} | {fmt(bb['M_v8n_baseline_s42'][b][0])} | "
            + " | ".join(f"{x:.4f}" for x in v) + f" | {mean} |")
    lo = [bb[f"M_v8n_baseline_s{s}"]["size:<12"][2] for s in SEEDS]
    hi = [bb[f"M_v8n_baseline_s{s}"]["size:32+"][2] for s in SEEDS]
    chk("at-a-glance 32+", f"finds **{100 * st.mean(hi):.1f} %** of targets larger than 32 px")
    chk("at-a-glance <12", f"only **{100 * st.mean(lo):.1f} %** of targets smaller than 12 px")
    chk("collapse factor", "falls by a factor of about **eleven**", round(st.mean(hi) / st.mean(lo)) == 11)
    chk("seed spread", f"spread {max(lo) - min(lo):.3f} in the smallest band against an effect of {st.mean(hi) - st.mean(lo):.2f}")
    res = [r for r in rows("results/protocol/results/RESULTS.csv") if r["slice"] == "ALL" and r["study"] == "M"]
    m50 = [float(r["mAP50"]) for r in res]
    chk("mAP50", f"**mAP@0.5 of {st.mean(m50):.3f}** (sd {st.stdev(m50):.3f}", len(res) == 3)
    chk("mAP50-95 and recall", f"mAP@0.5:0.95 {st.mean(float(r['mAP50_95']) for r in res):.3f}; overall recall {st.mean(float(r['recall']) for r in res):.3f}")
    anyc = {r["tag"]: r for r in b1 if r["slice"] == "size:<12"}
    s123 = anyc["M_v8n_baseline_s123"]
    chk("class-agnostic seed 123", f"(seed 123: {float(s123['recall']):.4f} → {float(s123['recall_any_class']):.4f})")
    miss = 100 * (1 - st.mean(float(anyc[f"M_v8n_baseline_s{s}"]["recall_any_class"]) for s in SEEDS))
    chk("no box of any class", f"about **{miss:.0f} % of sub-12-pixel targets produce no box of any class**")
    chk("at-a-glance no box", f"About **{miss:.0f} %** of sub-12-pixel targets produce no box of any class")
    bb1 = t3.load_bands([os.path.join(RES, "tier1_backfill", "BANDS_tier1_backfill.csv")])
    occ = {}
    for v in ("BYPASSED", "AMBIGUOUS", "LIGHT", "OCCLUDED"):
        cells = [bb1[("yolov8n", s)][f"occ:{v}"] for s in SEEDS]
        occ[v] = (st.mean(c[2] for c in cells), {c[0] for c in cells})
    chk("baseline recall by occlusion verdict",
        f"three-seed recall is {occ['BYPASSED'][0]:.3f} on BYPASSED boxes ({fmt(min(occ['BYPASSED'][1]))}), "
        f"{occ['AMBIGUOUS'][0]:.3f} on AMBIGUOUS ({fmt(min(occ['AMBIGUOUS'][1]))}), "
        f"{occ['LIGHT'][0]:.3f} on LIGHT ({fmt(min(occ['LIGHT'][1]))}) and "
        f"{occ['OCCLUDED'][0]:.3f} on OCCLUDED ({fmt(min(occ['OCCLUDED'][1]))})",
        all(len(v[1]) == 1 for v in occ.values()))
    tiles = tiles_for(W, H, 2, 0.15)
    tw = tiles[0][2] - tiles[0][0]
    chk("letterbox scale", "at 1280 px, 1.33 times its native width", abs(1280 / W - 1.333) < 0.001)
    chk("tiled scale", f"a 12 px target reaches the network at about {12 * 1280 / tw:.0f} px instead of {12 * 1280 / W:.0f} px")

    # ================================================================ tier 2
    print("\n-- Tier 2 --")
    b2 = t2.load_bands(os.path.join(RES, "tier2", "BANDS_tier2_plus_baseline.csv"))
    tags2 = list(b2)
    speed = collections.defaultdict(list)
    for r in rows("results/protocol/results/INFERENCE_SPEED.csv"):
        if r["inference_ms"] not in ("", "NOT_MEASURED"):
            speed[r["arm"]].append(float(r["inference_ms"]))
    base_ms = st.mean(speed["yolov8n"])
    chk("baseline speed", f"| Baseline | 0.0853 | reference | {base_ms:.1f} ms | — |")
    same_sign = []
    for arm, lab, verdict in (("yolov8n-p2", "P2 detection level", "null"),
                              ("yolov8n-cbam", "CBAM attention", "**worse than baseline**"),
                              ("yolov8n-p2-cbam", "P2 + CBAM", "null")):
        d = [b2[t2.tag_for(arm, s, tags2)]["size:<12"][2] - b2[t2.tag_for("yolov8n", s, tags2)]["size:<12"][2] for s in SEEDS]
        rec = st.mean(b2[t2.tag_for(arm, s, tags2)]["size:<12"][2] for s in SEEDS)
        lo_, hi_ = t2.bc_ci(d)
        ms = st.mean(speed[arm])
        chk(f"Tier 2 row {lab}", f"| {lab} | {rec:.4f} | {st.mean(d):+.4f} [{lo_:+.4f}, {hi_:+.4f}] | {ms:.1f} ms | {verdict} |".replace("-", "−").replace("P2 + CBAM", "P2 + CBAM").replace("yolov8n−", "yolov8n-"))
        if lo_ > 0 or hi_ < 0:
            same_sign.append((lab, d))

    # ================================================================ tier 3
    print("\n-- Tier 3 --")
    b3 = t3.load_bands([os.path.join(RES, "tier3", "BANDS_tier3.csv"),
                        os.path.join(RES, "tier1_backfill", "BANDS_tier1_backfill.csv")])
    base_b, _ = t3.budget(b3, "yolov8n")
    base_h, _ = t3.halluc(b3, "yolov8n")
    chk("Tier 3 baseline row", f"| YOLOv8n baseline | 0.0853 | reference | {base_b:.2f} | {base_h:.3f} | — |")
    fr = t3.paired(b3, "fasterrcnn", "size:<12")
    dfr = [a - b for _, a, b in fr]
    lo_, hi_ = t3.bc_ci(dfr)
    frb, _ = t3.budget(b3, "fasterrcnn")
    frh, _ = t3.halluc(b3, "fasterrcnn")
    chk("Faster R-CNN row", (f"| Faster R-CNN | {st.mean(a for _, a, _ in fr):.4f} | {st.mean(dfr):+.4f} [{lo_:+.4f}, {hi_:+.4f}] | "
                             f"{frb:.2f} | **{frh:.3f}** | null |").replace("-0", "−0"))
    rt = t3.paired(b3, "rtdetr-l", "size:<12")
    drt = [a - b for _, a, b in rt]
    rtb, _ = t3.budget(b3, "rtdetr-l")
    rth, _ = t3.halluc(b3, "rtdetr-l")
    chk("RT-DETR-L row", f"| RT-DETR-L ({len(rt)} seeds) | {st.mean(a for _, a, _ in rt):.4f} | {st.mean(drt):+.4f} (no interval) | "
                         f"**{rtb:.2f}** | {rth:.3f} | underpowered; **{rtb / base_b:.1f}× box budget** |",
        t3.bc_ci(drt) == (None, None))
    chk("eighteen times", "eighteen times the baseline's rate", round(frh / base_h) == 18)
    spread = dict((s, a - b) for s, a, b in fr)
    chk("Faster R-CNN seed spread",
        (f"seed pairs alone span {spread['42']:+.4f}, {spread['123']:+.4f} and {spread['456']:+.4f}"
         ).replace("-0", "−0"))
    oc = t3.paired(b3, "fasterrcnn", "occ:OCCLUDED")
    doc = [a - b for _, a, b in oc]
    lo_, hi_ = t3.bc_ci(doc)
    chk("Faster R-CNN occluded", f"({st.mean(doc):+.4f} [{lo_:+.4f}, {hi_:+.4f}] on the OCCLUDED boxes)".replace("-0", "−0"))
    same_sign.append(("Faster R-CNN occluded", doc))
    chk("RT-DETR budget", f"emits {rtb:.2f} boxes per image against the baseline's {base_b:.2f}")
    h4 = max(st.mean(a for _, a, _ in fr), st.mean(a for _, a, _ in rt))
    chk("H4 below 0.20", "Both stay far below 0.20", h4 < 0.20)

    # ================================================================ tier 4
    print("\n-- Tier 4 --")
    tab = rows("results/protocol/results/tier4/TIER4_VS_BASELINE.csv")
    labels4 = {"Small-object augmentation": "Small-object augmentation",
               "Small-object-aware selection": "Small-object-aware selection (Study M4)",
               "Larger backbone + augmentation": "Larger backbone (YOLOv8s) + augmentation",
               "Tiled inference alone": "Tiled inference alone",
               "P2 + augmentation": "P2 + augmentation",
               "Augmentation + tiled inference": "Augmentation + tiled inference",
               "P2 + augmentation + tiled inference": "**P2 + augmentation + tiled inference**"}
    for r in tab:
        d = float(r["diff"])
        if r["ci_lo"]:
            ci = f"{d:+.4f} [{float(r['ci_lo']):+.4f}, {float(r['ci_hi']):+.4f}]"
            verdict = {"worse than baseline": "**worse than baseline**", "null, CI includes zero": "null",
                       "clears the floor": "**clears the floor**"}[r["verdict"]]
        else:
            ci = f"{d:+.4f} (no interval)"
            verdict = "underpowered"
        name = labels4[r["arm"]]
        recall = r["recall"]
        if name.startswith("**"):
            ci, recall = f"**{ci}**", f"**{recall}**"
        line = f"| {name} | {r['completed']} of 3 | {recall} | {ci} | {verdict} |".replace("-0.", "−0.")
        chk(f"Tier 4 row {r['arm']}", line)
        if r["ci_lo"] and (float(r["ci_lo"]) > 0 or float(r["ci_hi"]) < 0):
            base = t2.load_bands(os.path.join(RES, "tier4", "BANDS_tier4.csv"))
            pat = dict(zip(labels4, ["M4_yolov8n_augmented_s{s}", "M4_yolov8n_baseline_s{s}", "M4_yolov8s_augmented_s{s}",
                                     "M4_yolov8n_baseline_s{s}_tiled2", "M4_yolov8n-p2_augmented_s{s}",
                                     "M4_yolov8n_augmented_s{s}_tiled2", "M4_yolov8n-p2_augmented_s{s}_tiled2"]))[r["arm"]]
            same_sign.append((r["arm"], [base[pat.format(s=s)]["size:<12"][2] - base[f"M_v8n_baseline_s{s}"]["size:<12"][2] for s in SEEDS]))
    win = next(r for r in tab if r["arm"] == "P2 + augmentation + tiled inference")
    chk("at-a-glance winner", f"0.085 to **{float(win['recall']):.3f}** (+{float(win['diff']):.3f}, 95 % CI [+{float(win['ci_lo']):.3f}, +{float(win['ci_hi']):.3f}])")
    status = rows("results/protocol/runs_t4/RUN_STATUS.csv")
    trunc = sorted((int(r["epochs_run"]), r["scored"]) for r in status if not r["status"].startswith("completed"))
    chk("truncated runs", "crashed while saving a checkpoint at epoch 20, one stopped at epoch 36, one at epoch 85, and one crashed at epoch 33 before it could be scored",
        trunc == [(20, "1"), (33, "0"), (36, "1"), (85, "1")])
    # hypotheses, recomputed with analyse_tier4's own cells and rules
    b4 = t2.load_bands(os.path.join(RES, "tier4", "BANDS_tier4.csv"))
    tags4 = list(b4)
    st4 = t4.load_run_status()

    def contrast(a_cell, b_cell):
        pairs = []
        done = 3
        for s in SEEDS:
            ta, tb = t4.tag_for(a_cell, s, tags4), t4.tag_for(b_cell, s, tags4)
            if ta and tb:
                pairs.append(b4[ta]["size:<12"][2] - b4[tb]["size:<12"][2])
        for c in (a_cell, b_cell):
            done = min(done, sum(1 for s in SEEDS if t4.tag_for(c, s, tags4) and t4.run_complete(t4.tag_for(c, s, tags4), st4)))
        return pairs, done
    p7, d7 = contrast("M4/augmented", "M4/baseline")
    chk("H7", f"**underpowered** ({st.mean(p7):+.4f}; {d7} of 3 augmented runs completed)".replace("-0", "−0"), d7 < 3)
    p8, d8 = contrast("M4/baseline", "M/baseline")
    lo_, hi_ = t2.bc_ci(p8)
    chk("H8", f"**refuted: worse** ({st.mean(p8):+.4f} [{lo_:+.4f}, {hi_:+.4f}])".replace("-0", "−0"), hi_ < 0 and d8 == 3)
    p9, _ = contrast("M4/augmented", "M/baseline")
    chk("H9", f"underpowered ({st.mean(p9):+.4f})".replace("-0", "−0"))
    p10, d10 = contrast("M4/baseline TILED", "M4/baseline")
    lo_, hi_ = t2.bc_ci(p10)
    chk("H10", f"null ({st.mean(p10):+.4f} [{lo_:+.4f}, {hi_:+.4f}])".replace("-0", "−0"), lo_ <= 0 <= hi_ and d10 == 3)
    p10b, _ = contrast("M4/augmented TILED", "M4/augmented")
    chk("H10b", f"underpowered ({st.mean(p10b):+.4f})")
    p11, _ = contrast("M4/p2+augmented", "M4/augmented")
    chk("H11", f"underpowered ({st.mean(p11):+.4f})")
    p12, _ = contrast("M4/yolov8s+aug", "M4/augmented")
    chk("H12", f"underpowered ({st.mean(p12):+.4f}, {len(p12)} seeds)")

    # the best configuration against its own whole-frame weights
    wf = [b4[f"M4_yolov8n-p2_augmented_s{s}"] for s in SEEDS]
    tl = [b4[f"M4_yolov8n-p2_augmented_s{s}_tiled2"] for s in SEEDS]
    bw, bt = st.mean(x["pred_budget"][2] for x in wf), st.mean(x["pred_budget"][2] for x in tl)
    hw, ht = st.mean(x["halluc_fullyocc"][2] for x in wf), st.mean(x["halluc_fullyocc"][2] for x in tl)
    chk("tiled budget", f"{bt:.2f} boxes per image tiled against {bw:.2f} whole-frame ({bt / bw:.2f}×)")
    chk("tiled ungrounded", f"and {ht:.4f} ungrounded detections per frame against {hw:.4f} whole-frame")
    r4 = [r for p in glob.glob(os.path.join(ROOT, "results", "protocol", "runs_t4", "M4_yolov8n-p2_augmented_s*", "RESULTS.csv"))
          for r in csv.DictReader(open(p, encoding="utf-8")) if r["slice"] == "ALL"]
    base_rec = st.mean(lo)
    chk("best: recall <12", f"| Recall < 12 px | {base_rec:.4f} | **{float(win['recall']):.4f}** (+{100 * (float(win['recall']) / base_rec - 1):.0f} %) |")
    chk("best: recall 32+", f"| Recall 32+ px | {st.mean(hi):.3f} | {st.mean(x['size:32+'][2] for x in tl):.3f} |")
    chk("best: mAP50", f"| mAP@0.5 (both whole-frame) | {st.mean(m50):.3f} | {st.mean(float(r['mAP50']) for r in r4):.3f} |", len(r4) == 3)
    chk("best: mAP50-95", f"| mAP@0.5:0.95 (both whole-frame) | {st.mean(float(r['mAP50_95']) for r in res):.3f} | {st.mean(float(r['mAP50_95']) for r in r4):.3f} |")
    chk("best: overall recall", f"| Overall recall (both whole-frame) | {st.mean(float(r['recall']) for r in res):.3f} | {st.mean(float(r['recall']) for r in r4):.3f} |")
    chk("best: boxes", f"| Boxes per image | {base_b:.2f} | {bt:.2f} |")
    p2ms = st.mean(speed["yolov8n-p2"])
    chk("best: speed", f"≈ {4 * p2ms:.0f} ms, *estimated* as four passes at the measured {p2ms:.1f} ms")
    probe = open(os.path.join(RES, "tier4", "TILE_PROBE.txt"), encoding="utf-8").read()
    chk("tiling pilot", "a 120-frame, single-seed pilot", "120 frames | 2x2 tiles" in probe)
    fs = json.load(open(os.path.join(RES, "FRCNN_SPEED.json"), encoding="utf-8"))
    chk("Faster R-CNN CPU timing", f"timed only on CPU ({fs['ratio_frcnn_over_yolov8n']:.1f}× the baseline)")

    # ================================================== protocol statements
    print("\n-- protocol --")
    chk("same-sign claim", "every interval that excludes zero comes from three paired differences of the same sign",
        all(len(d) == 3 and (all(x > 0 for x in d) or all(x < 0 for x in d)) for _, d in same_sign))
    ba = {r["tag"] + r["slice"]: int(r["found"]) for r in rows("results/protocol/results/tier1_backfill/BANDS_tier1_backfill.csv")}
    bf = {r["tag"] + r["slice"]: int(r["found"]) for r in b1}
    chk("duplicate scoring, 32+ s123",
        f"32+ px, seed 123: {bf['M_v8n_baseline_s123size:32+']} or {ba['M_v8n_baseline_s123size:32+']} of 277")
    chk("duplicate scoring, 20-32 s456",
        f"20–32 px, seed 456: {fmt(bf['M_v8n_baseline_s456size:20-32'])} or {fmt(ba['M_v8n_baseline_s456size:20-32'])} of 1,748")
    gpus = {r["study"]: r["gpu"] for f in ("RESULTS.csv", "tier2/RESULTS_tier2.csv") for r in rows(f"results/protocol/results/{f}")}
    gpu3 = {r["gpu"] for r in rows("results/protocol/results/tier3/RESULTS_tier3.csv")}
    gpu4 = {r["gpu"] for p in glob.glob(os.path.join(ROOT, "results", "protocol", "runs_t4", "*", "RESULTS.csv"))
            for r in csv.DictReader(open(p, encoding="utf-8"))}
    chk("GPUs", "Tiers 1, 2 and 4 were trained on NVIDIA A40 GPUs and Tier 3 on H100s",
        set(gpus.values()) == {"NVIDIA A40"} and gpu4 == {"NVIDIA A40"} and all("H100" in g for g in gpu3))
    args = [open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(ROOT, "results", "protocol", "runs_t4", "*", "args.yaml"))]
    chk("held constant", "input size 1280, batch 24, 8 dataloader workers, at most 100 epochs with early-stopping patience 30",
        all("imgsz: 1280" in a and "batch: 24" in a and "workers: 8" in a and "epochs: 100" in a and "patience: 30" in a and "hsv_h: 0.005" in a for a in args) and len(args) == 12)

    print(f"\n  {OK} passed, {FAIL} failed")
    if FAIL:
        sys.exit("README.md AND THE DATA DISAGREE - fix the document, never the data.")
    print("Every checked number in README.md matches the repository's data.")


if __name__ == "__main__":
    main()
