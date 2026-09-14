"""Tiled inference: score a trained model on an n x n grid of overlapping tiles.

    python code/tile_probe.py --weights <best.pt> --data <yaml> --tiles 2
    python code/tile_probe.py --weights <best.pt> --data <yaml> --tiles 2 \
        --csv <BANDS csv> --tag M4_yolov8n-p2_augmented_s42_tiled2

Tiling does not raise the network's input size. Each tile is resized to the same
1280 px the model was trained on, so there is no train/test resolution mismatch,
but a target inside a tile reaches the network at about 1.7 times the scale it has
in the whole frame: with 2 x 2 tiles of 552 x 414 px, a 12 px target arrives at
about 28 px instead of 16 px.
Nothing is retrained and the confidence threshold is unchanged, so the box budget
is reported beside recall - a recall gain that arrives with a large budget
increase is not a gain.

Tiles overlap (--overlap, default 0.15) so a target on a seam is whole in at least
one tile. Detections are mapped back to full-frame coordinates and merged by
greedy highest-confidence-first suppression per class.

With --csv the result is appended in the band-ledger schema of score_bands.py -
size bands, pred_budget and halluc_fullyocc - so tiled runs are seed-paired and
bootstrapped like every other arm. For tiled rows, recall_any_class repeats
recall: class-agnostic matching is not computed here.
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from score_bands import band_of, iou_xyxy  # noqa: E402


def tiles_for(w, h, n, overlap):
    """n x n grid with fractional overlap. Returns [(x0,y0,x1,y1)]."""
    out = []
    tw, th = w / n, h / n
    ox, oy = tw * overlap, th * overlap
    for i in range(n):
        for j in range(n):
            x0 = max(0, i * tw - ox); x1 = min(w, (i + 1) * tw + ox)
            y0 = max(0, j * th - oy); y1 = min(h, (j + 1) * th + oy)
            out.append((int(x0), int(y0), int(x1), int(y1)))
    return out


def merge(dets, iou_thr=0.5):
    """Greedy highest-confidence-first suppression across tile boundaries."""
    dets = sorted(dets, key=lambda d: -d[1])
    kept = []
    for cls, conf, box in dets:
        if any(c == cls and iou_xyxy(box, b) >= iou_thr for c, _, b in kept):
            continue
        kept.append((cls, conf, box))
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--tiles", type=int, default=2, help="n for an n x n grid; 1 = whole frame")
    ap.add_argument("--overlap", type=float, default=0.15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--csv", default="", help="append band-ledger rows here")
    ap.add_argument("--tag", default="", help="run tag for the csv rows")
    a = ap.parse_args()

    import cv2
    from ultralytics import YOLO

    base = os.path.dirname(os.path.abspath(a.data))
    imgdir = os.path.join(base, "images", "test")
    labdir = os.path.join(base, "labels", "test")
    m = YOLO(a.weights)
    names = m.names if isinstance(m.names, dict) else dict(enumerate(m.names))

    imgs = sorted(f for f in os.listdir(imgdir) if f.lower().endswith(".jpg"))
    if a.limit:
        imgs = imgs[:a.limit]
    print(f"  {len(imgs)} frames | {a.tiles}x{a.tiles} tiles, overlap {a.overlap} | "
          f"imgsz {a.imgsz} conf {a.conf}")

    tot = {}; hit = {}; npred = 0
    for f in imgs:
        im = cv2.imread(os.path.join(imgdir, f))
        if im is None:
            continue
        H, W = im.shape[:2]
        regions = [(0, 0, W, H)] if a.tiles <= 1 else tiles_for(W, H, a.tiles, a.overlap)
        dets = []
        for (x0, y0, x1, y1) in regions:
            crop = im[y0:y1, x0:x1]
            r = m.predict(crop, imgsz=a.imgsz, conf=a.conf, device=a.device,
                          verbose=False)[0]
            if r.boxes is None or not len(r.boxes):
                continue
            xy = r.boxes.xyxy.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            cf = r.boxes.conf.cpu().numpy()
            sx = (x1 - x0) / crop.shape[1]; sy = (y1 - y0) / crop.shape[0]
            for k in range(len(cl)):
                b = (x0 + xy[k][0] * sx, y0 + xy[k][1] * sy,
                     x0 + xy[k][2] * sx, y0 + xy[k][3] * sy)
                dets.append((int(cl[k]), float(cf[k]), b))
        dets = merge(dets, a.iou)
        npred += len(dets)

        lp = os.path.join(labdir, os.path.splitext(f)[0] + ".txt")
        if not os.path.exists(lp):
            continue
        for line in open(lp, encoding="utf-8"):
            q = line.split()
            if len(q) != 5:
                continue
            c = int(float(q[0])); xc, yc, bw, bh = (float(v) for v in q[1:])
            gb = ((xc - bw / 2) * W, (yc - bh / 2) * H,
                  (xc + bw / 2) * W, (yc + bh / 2) * H)
            band = band_of(((bw * W) * (bh * H)) ** 0.5)
            tot[band] = tot.get(band, 0) + 1
            if any(cc == c and iou_xyxy(gb, bb) >= a.iou for cc, _, bb in dets):
                hit[band] = hit.get(band, 0) + 1

    print(f"\n  {'band':<10}{'GT':>7}{'found':>8}{'recall':>9}")
    for b in ("<12", "12-20", "20-32", "32+"):
        if tot.get(b):
            print(f"  {b:<10}{tot[b]:>7}{hit.get(b,0):>8}{hit.get(b,0)/tot[b]:>9.4f}")
    print(f"\n  box budget: {npred/max(1,len(imgs)):.2f} boxes/image "
          f"({npred} over {len(imgs)} frames)")
    print("  NOTE: a recall gain that arrives with a large budget increase is not a")
    print("  gain - compare the budget against the whole-frame run.")

    if a.csv:
        import csv as _csv
        from score_bands import fully_occluded_set
        # the ungrounded-detection probe, computed as score_bands.py does:
        # detections fired on frames whose target is fully occluded and unlabelled
        full = fully_occluded_set()
        halluc_frames = halluc_det = 0
        for f in imgs:
            if f not in full:
                continue
            im = cv2.imread(os.path.join(imgdir, f))
            if im is None:
                continue
            H, W = im.shape[:2]
            regions = [(0, 0, W, H)] if a.tiles <= 1 else tiles_for(W, H, a.tiles, a.overlap)
            dets = []
            for (x0, y0, x1, y1) in regions:
                crop = im[y0:y1, x0:x1]
                r = m.predict(crop, imgsz=a.imgsz, conf=a.conf, device=a.device,
                              verbose=False)[0]
                if r.boxes is None or not len(r.boxes):
                    continue
                xy = r.boxes.xyxy.cpu().numpy()
                cl = r.boxes.cls.cpu().numpy().astype(int)
                cf = r.boxes.conf.cpu().numpy()
                for k in range(len(cl)):
                    dets.append((int(cl[k]), float(cf[k]),
                                 (x0 + xy[k][0], y0 + xy[k][1],
                                  x0 + xy[k][2], y0 + xy[k][3])))
            dets = merge(dets, a.iou)
            halluc_frames += 1
            halluc_det += len(dets)

        tag = a.tag or f"tiled{a.tiles}x{a.tiles}"
        rows = []
        for b in ("<12", "12-20", "20-32", "32+"):
            if tot.get(b):
                rows.append(dict(tag=tag, slice=f"size:{b}", n=tot[b],
                                 found=hit.get(b, 0),
                                 recall=f"{hit.get(b,0)/tot[b]:.4f}",
                                 recall_any_class=f"{hit.get(b,0)/tot[b]:.4f}"))
        rows.append(dict(tag=tag, slice="pred_budget", n=len(imgs), found=npred,
                         recall=f"{npred/max(1,len(imgs)):.4f}",
                         recall_any_class=f"{a.conf:.2f}"))
        rows.append(dict(tag=tag, slice="halluc_fullyocc", n=halluc_frames,
                         found=halluc_det,
                         recall=f"{halluc_det/max(1,halluc_frames):.4f}",
                         recall_any_class=f"{a.conf:.2f}"))
        cols = ["tag", "slice", "n", "found", "recall", "recall_any_class"]
        new = not os.path.exists(a.csv)
        os.makedirs(os.path.dirname(os.path.abspath(a.csv)), exist_ok=True)
        with open(a.csv, "a", newline="", encoding="utf-8") as fh:
            w = _csv.DictWriter(fh, fieldnames=cols)
            if new:
                w.writeheader()
            w.writerows(rows)
        print(f"\n  appended {len(rows)} rows as tag {tag} -> {a.csv}")
        print(f"  ungrounded detections on {halluc_frames} fully-occluded frames: "
              f"{halluc_det}")


if __name__ == "__main__":
    main()
