"""Per-size-band and per-occlusion-band recall for a trained model.

    python code/score_bands.py --weights <best.pt> --study M
    python code/score_bands.py --weights <best.pt> --data <yaml> --tag M_v8n_baseline_s42

Appends one row per slice to a band ledger (default results/protocol/BANDS.csv):

    size:<12, size:12-20, size:20-32, size:32+   recall by sqrt(box area) in px
    occ:OCCLUDED / LIGHT / BYPASSED / AMBIGUOUS  recall by the MEASURED occlusion
                                                 partition (occlusion_partition_s2/s3.csv),
                                                 never by flight name
    class:<name>                                 recall by class
    pred_budget                                  predictions per image (box budget)
    halluc_fullyocc                              detections per 100%-occluded frame

Recall, not AP, per band: AP needs a precision axis, and a false positive has no
ground-truth size to be binned by. Every ground-truth box has a size and an
occlusion verdict, and it was either found or not.

A ground-truth box is FOUND if a prediction of the same class overlaps it at
IoU >= --iou. Class-agnostic recall is reported beside it for the size bands, so
"found but misclassified" can be told apart from "not detected at all".
"""
import argparse
import collections
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
W, H = 960, 720


def iou_xyxy(a, b):
    ax0, ay0, ax1, ay1 = a
    bx0, by0, bx1, by1 = b
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax1 - ax0) * (ay1 - ay0) + (bx1 - bx0) * (by1 - by0) - inter
    return inter / ua if ua > 0 else 0.0


def band_of(sqrt_area):
    return ("<12" if sqrt_area < 12 else "12-20" if sqrt_area < 20
            else "20-32" if sqrt_area < 32 else "32+")


def occlusion_map(extra_dir=""):
    """image -> (verdict, target class), from the measured occlusion partitions.

    Several roots are searched and the caller can pass one, because a missing
    partition would otherwise drop the occlusion slices without an error."""
    m = {}
    roots = [r for r in (extra_dir, ROOT, os.getcwd(), "/workspace") if r]
    for sc, ds in (("s2", "dataset/01_Dataset_SOCC_S2"), ("s3", "dataset/01_Dataset_SOCC_S3")):
        for root in roots:
            for cand in (os.path.join(root, ds, f"occlusion_partition_{sc}.csv"),
                         os.path.join(root, f"occlusion_partition_{sc}.csv")):
                if os.path.exists(cand):
                    for r in csv.DictReader(open(cand, encoding="utf-8")):
                        # the verdict describes ONE target in the frame, not the
                        # frame: S3 frames carry three vehicles and only the
                        # all-terrain vehicle is on the occlusion ladder
                        m[r["image"]] = (r["verdict"], r.get("target", ""))
                    break
            else:
                continue
            break
    return m


def fully_occluded_set(extra_dir=""):
    """The 159 frames at 100% occlusion: S1's 141 (FULLY_OCCLUDED_S1.csv) plus S2's
    18 (the `fully_occluded` column of the S2 release manifest).

    These frames carry no ground-truth box - a fully hidden target has nothing
    visible to label - so they add nothing to any recall band. They are used as a
    probe instead: the target is physically present and invisible, so any
    detection there is ungrounded.
    """
    out = set()
    for root in [r for r in (extra_dir, ROOT, os.getcwd(), "/workspace") if r]:
        for cand in (os.path.join(root, "dataset/01_Dataset_SOCC", "release",
                                  "FULLY_OCCLUDED_S1.csv"),
                     os.path.join(root, "FULLY_OCCLUDED_S1.csv")):
            if os.path.exists(cand):
                for line in open(cand, encoding="utf-8").readlines()[1:]:
                    n = line.split(",")[0].strip()
                    if n:
                        out.add(n if n.endswith(".jpg") else n + ".jpg")
                break
        else:
            continue
        break

    for root in [r for r in (extra_dir, ROOT, os.getcwd(), "/workspace") if r]:
        for cand in (os.path.join(root, "dataset/01_Dataset_SOCC_S2", "release",
                                  "socc_s2_release_manifest.csv"),
                     os.path.join(root, "socc_s2_release_manifest.csv")):
            if os.path.exists(cand):
                for r2 in csv.DictReader(open(cand, encoding="utf-8")):
                    if str(r2.get("fully_occluded", "")).strip().lower() in (
                            "1", "true", "yes"):
                        n = r2["image"].strip()
                        out.add(n if n.endswith(".jpg") else n + ".jpg")
                break
        else:
            continue
        break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--study", default="")
    ap.add_argument("--data", default="")
    ap.add_argument("--split", default="test")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--device", default="")
    ap.add_argument("--tag", default="")
    ap.add_argument("--csv", default="")
    ap.add_argument("--partition-dir", default="",
                    help="where occlusion_partition_s{2,3}.csv and "
                         "FULLY_OCCLUDED_S1.csv live, when run outside the repository")
    ap.add_argument("--limit", type=int, default=0,
                    help="score only the first N images (smoke test only)")
    a = ap.parse_args()

    if a.data:
        yml = os.path.abspath(a.data)
        base = os.path.dirname(yml)
    else:
        base = os.path.join(ROOT, "results", "protocol", f"study_{a.study}")
        yml = os.path.join(base, f"socc_study_{a.study}.yaml")
    imgdir = os.path.join(base, "images", a.split)
    labdir = os.path.join(base, "labels", a.split)
    for d in (imgdir, labdir):
        if not os.path.isdir(d):
            sys.exit(f"missing {d}")

    from ultralytics import YOLO
    m = YOLO(a.weights)
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))
    occ = occlusion_map(a.partition_dir)
    full = fully_occluded_set(a.partition_dir)
    print(f"  occlusion partition: {len(occ)} frames    "
          f"100%-occluded list: {len(full)} frames")
    if not occ:
        print("  *** WARNING: NO OCCLUSION PARTITION FOUND. The occlusion axis")
        print("  *** will be ABSENT from this result. Pass --partition-dir.")

    imgs = sorted(f for f in os.listdir(imgdir)
                  if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if a.limit:
        imgs = imgs[:a.limit]
        print(f"  *** LIMIT {a.limit} - SMOKE TEST, not a result")
    print(f"  scoring {len(imgs)} {a.split} images at conf {a.conf}, IoU {a.iou}")

    size_tot = collections.Counter(); size_hit = collections.Counter()
    size_hit_any = collections.Counter()
    occ_tot = collections.Counter(); occ_hit = collections.Counter()
    cls_tot = collections.Counter(); cls_hit = collections.Counter()
    npred = 0
    halluc_frames = 0      # 100%-occluded frames actually scored
    halluc_det = 0         # detections fired on them - all ungrounded

    B = 32
    for i in range(0, len(imgs), B):
        chunk = imgs[i:i + B]
        paths = [os.path.join(imgdir, f) for f in chunk]
        kw = dict(imgsz=a.imgsz, conf=a.conf, verbose=False)
        if a.device:
            kw["device"] = a.device
        res = m.predict(paths, **kw)
        for f, r in zip(chunk, res):
            stem = os.path.splitext(f)[0]
            lp = os.path.join(labdir, stem + ".txt")
            gts = []
            if os.path.exists(lp):
                for line in open(lp, encoding="utf-8"):
                    q = line.split()
                    if len(q) != 5:
                        continue
                    c = int(float(q[0]))
                    xc, yc, bw, bh = (float(x) for x in q[1:])
                    gts.append((c, (xc - bw / 2) * W, (yc - bh / 2) * H,
                                (xc + bw / 2) * W, (yc + bh / 2) * H,
                                ((bw * W) * (bh * H)) ** 0.5))
            preds = []
            if r.boxes is not None and len(r.boxes):
                xy = r.boxes.xyxy.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                for k in range(len(cl)):
                    preds.append((int(cl[k]), tuple(float(v) for v in xy[k])))
            npred += len(preds)

            # 100%-occluded frames hold no ground truth: every detection on them
            # is counted as ungrounded
            if f in full:
                halluc_frames += 1
                halluc_det += len(preds)

            # the occlusion verdict belongs to one target, so it is matched on
            # class as well as frame
            vv = occ.get(f)
            v_verdict, v_target = vv if vv else (None, None)
            for c, x0, y0, x1, y1, sa in gts:
                sb = band_of(sa)
                size_tot[sb] += 1
                cls_tot[c] += 1
                if f in full and names.get(c) == "rc_car":
                    v = "FULLY_OCCLUDED"      # S1's list covers the red car only
                elif v_verdict and v_target and names.get(c) == v_target:
                    v = v_verdict
                else:
                    v = None
                if v:
                    occ_tot[v] += 1
                same = any(pc == c and iou_xyxy((x0, y0, x1, y1), pb) >= a.iou
                           for pc, pb in preds)
                anyc = any(iou_xyxy((x0, y0, x1, y1), pb) >= a.iou
                           for pc, pb in preds)
                if same:
                    size_hit[sb] += 1; cls_hit[c] += 1
                    if v:
                        occ_hit[v] += 1
                if anyc:
                    size_hit_any[sb] += 1
        if (i // B) % 10 == 0:
            print(f"    {min(i+B, len(imgs))}/{len(imgs)}")

    rows = []
    tag = a.tag or os.path.basename(os.path.dirname(os.path.dirname(a.weights)))
    print()
    print("  RECALL BY TARGET SIZE  (sqrt-area px)")
    print(f"    {'band':<10}{'GT':>7}{'found':>8}{'recall':>9}{'any-class':>11}")
    for b in ("<12", "12-20", "20-32", "32+"):
        if not size_tot[b]:
            continue
        rc = size_hit[b] / size_tot[b]
        ra = size_hit_any[b] / size_tot[b]
        print(f"    {b:<10}{size_tot[b]:>7}{size_hit[b]:>8}{rc:>9.3f}{ra:>11.3f}")
        rows.append(dict(tag=tag, slice=f"size:{b}", n=size_tot[b],
                         found=size_hit[b], recall=f"{rc:.4f}",
                         recall_any_class=f"{ra:.4f}"))
    if occ_tot:
        print()
        print("  RECALL BY MEASURED OCCLUSION")
        print(f"    {'band':<12}{'GT':>7}{'found':>8}{'recall':>9}")
        for b in ("FULLY_OCCLUDED", "OCCLUDED", "LIGHT", "BYPASSED", "AMBIGUOUS"):
            if not occ_tot[b]:
                continue
            rc = occ_hit[b] / occ_tot[b]
            print(f"    {b:<12}{occ_tot[b]:>7}{occ_hit[b]:>8}{rc:>9.3f}")
            rows.append(dict(tag=tag, slice=f"occ:{b}", n=occ_tot[b],
                             found=occ_hit[b], recall=f"{rc:.4f}",
                             recall_any_class=""))
    print()
    print("  RECALL BY CLASS")
    for c in sorted(cls_tot):
        rc = cls_hit[c] / cls_tot[c]
        print(f"    {names.get(c, c):<18}{cls_tot[c]:>7}{cls_hit[c]:>8}{rc:>9.3f}")
        rows.append(dict(tag=tag, slice=f"class:{names.get(c, c)}", n=cls_tot[c],
                         found=cls_hit[c], recall=f"{rc:.4f}", recall_any_class=""))
    ppi = npred / max(1, len(imgs))
    print(f"\n  {npred} predictions over {len(imgs)} images "
          f"({ppi:.2f}/image) at conf {a.conf}")

    # The box budget travels with the recall numbers. Recall counts a ground-truth
    # box as found if ANY same-class prediction overlaps it, so it is blind to false
    # positives, and a detector emitting more boxes gets more chances at the same
    # target. RT-DETR applies no NMS, so across paradigms the budget can differ
    # several-fold. Written into the same six columns as every other slice, so the
    # ledger schema is identical for all three trainers.
    rows.append(dict(tag=tag, slice="pred_budget", n=len(imgs), found=npred,
                     recall=f"{ppi:.4f}", recall_any_class=f"{a.conf:.2f}"))

    if halluc_frames:
        hr = halluc_det / halluc_frames
        print(f"  100%-occluded frames scored: {halluc_frames}   "
              f"ungrounded detections: {halluc_det} ({hr:.3f}/frame)")
        rows.append(dict(tag=tag, slice="halluc_fullyocc", n=halluc_frames,
                         found=halluc_det, recall=f"{hr:.4f}",
                         recall_any_class=f"{a.conf:.2f}"))
    else:
        # an absent row would be indistinguishable from zero detections
        print("  no 100%-occluded frames in this split - ungrounded-detection probe absent")

    out = a.csv or os.path.join(ROOT, "results", "protocol", "BANDS.csv")
    new = not os.path.exists(out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["tag", "slice", "n", "found",
                                           "recall", "recall_any_class"])
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"  appended {len(rows)} rows to {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
