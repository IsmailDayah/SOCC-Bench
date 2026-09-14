"""Faster R-CNN (ResNet50-FPN) on a study split - the two-stage paradigm.

    python code/train_frcnn_protocol.py --study M --seed 42 --epochs 100

Faster R-CNN comes from torchvision, not Ultralytics, so it has its own trainer
and scorer. Everything else matches the Ultralytics path: the same study splits,
the same 1280 input, the same seeds, and the same size and occlusion band
definitions (imported from score_bands.py), so its rows land in the same ledgers
and mean the same thing.

`weights="DEFAULT"` loads the COCO-pretrained ResNet50-FPN; the box predictor is
replaced for 3 classes + background, so the backbone is pretrained and the head is
new - the same asymmetry as the P2 and CBAM arms.

torchvision reserves class 0 for background, so YOLO class c is c+1 internally and
is mapped back when scoring.

Checkpoint selection is on held-out validation loss, early stopping at patience 30.
Training applies a p=0.5 horizontal flip and no other augmentation; the Ultralytics
arms also use mosaic, scale, HSV jitter and random erasing.
"""
import argparse
import collections
import csv
import glob
import os
import random
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO = os.path.join(ROOT, "results", "protocol")
W, H = 960, 720
NC = 3                       # rc_car, atv_sherp, truck_peterbilt
NAMES = ["rc_car", "atv_sherp", "truck_peterbilt"]


def band_of(sqrt_area):
    return ("<12" if sqrt_area < 12 else "12-20" if sqrt_area < 20
            else "20-32" if sqrt_area < 32 else "32+")


# The occlusion axis is defined in one place, score_bands.py, and imported here so
# the two-stage arm uses exactly the same partition, target matching and
# 100%-occluded frame list as every other arm.
try:
    from score_bands import occlusion_map, fully_occluded_set
except ImportError as _e:                       # pragma: no cover
    sys.exit(f"cannot import score_bands (needed for the occlusion axis): {_e}\n"
             "  it must sit beside this file.")


def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class YoloDS:
    """Reads the same YOLO label files the Ultralytics arms train on, so the two
    paradigms cannot diverge on data."""

    def __init__(self, img_dir, imgsz, train=False):
        import cv2
        self.cv2 = cv2
        self.imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                           glob.glob(os.path.join(img_dir, "*.png")))
        self.lbl_dir = img_dir.replace(os.sep + "images" + os.sep,
                                       os.sep + "labels" + os.sep)
        self.imgsz = imgsz
        # Horizontal flip on the train split only. It is the one augmentation the
        # torchvision reference detection recipe and the Ultralytics defaults
        # share; the remaining difference (mosaic above all, which manufactures
        # small-object training signal) is declared, not matched.
        self.train = train

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        import numpy as np, torch
        p = self.imgs[i]
        im = self.cv2.imread(p)[:, :, ::-1]
        h0, w0 = im.shape[:2]
        im = self.cv2.resize(im, (self.imgsz, self.imgsz))
        sx, sy = self.imgsz / w0, self.imgsz / h0
        t = torch.from_numpy(np.ascontiguousarray(im)).permute(2, 0, 1).float() / 255
        lp = os.path.join(self.lbl_dir,
                          os.path.splitext(os.path.basename(p))[0] + ".txt")
        boxes, labels = [], []
        if os.path.exists(lp):
            for line in open(lp, encoding="utf-8"):
                q = line.split()
                if len(q) != 5:
                    continue
                c = int(float(q[0])); xc, yc, bw, bh = (float(v) for v in q[1:])
                x0 = (xc - bw/2) * w0 * sx; x1 = (xc + bw/2) * w0 * sx
                y0 = (yc - bh/2) * h0 * sy; y1 = (yc + bh/2) * h0 * sy
                if x1 > x0 and y1 > y0:
                    boxes.append([x0, y0, x1, y1])
                    labels.append(c + 1)          # torchvision: 0 is BACKGROUND
        if not boxes:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)

        # p=0.5 horizontal flip, train split only - validation selects the
        # checkpoint and test is the result, so neither is perturbed.
        # The draw uses torch's RNG: DataLoader seeds each worker from the main
        # generator, which torch.manual_seed(--seed) controls, so the flips are
        # reproducible for a seed and differ between epochs. Python's `random`
        # would not be where workers are spawned rather than forked.
        if self.train and float(torch.rand(())) < 0.5:
            t = torch.flip(t, dims=[2])
            if len(boxes):
                W = float(self.imgsz)
                x0 = boxes[:, 0].clone()
                boxes[:, 0] = W - boxes[:, 2]
                boxes[:, 2] = W - x0
        return t, {"boxes": boxes, "labels": labels}, os.path.basename(p), (w0, h0)


def collate(b):
    return tuple(zip(*b))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="M")
    ap.add_argument("--data", default="")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--out-root", default=PROTO)
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--partition-dir", default="",
                    help="where occlusion_partition_s{2,3}.csv and "
                         "FULLY_OCCLUDED_S1.csv live, when run outside the "
                         "repository. Without the partition no result is written.")
    ap.add_argument("--smoke", type=int, default=0,
                    help="use only N train and N test images (validation only)")
    a = ap.parse_args()

    if a.batch < 1:
        sys.exit("batch must be >= 1")

    import numpy as np, torch, torchvision
    from torch.utils.data import DataLoader
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    # Share worker tensors through shared memory, not file descriptors. With the
    # default `file_descriptor` strategy every tensor a worker returns holds an
    # open descriptor in the parent; with many workers at 1280 px that exhausts
    # RLIMIT_NOFILE and the loader fails with "received 0 items of ancdata".
    torch.multiprocessing.set_sharing_strategy("file_system")

    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)

    base = a.data or os.path.join(a.out_root, f"study_{a.study}")
    tr_dir = os.path.join(base, "images", "train")
    te_dir = os.path.join(base, "images", "test")
    for d in (tr_dir, te_dir):
        if not os.path.isdir(d):
            sys.exit(f"missing {d}")

    tag = f"{a.study}_fasterrcnn_baseline_s{a.seed}"
    run = os.path.join(a.out_root, "runs", tag)
    os.makedirs(os.path.join(run, "weights"), exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print("=" * 74)
    print(f"TRAIN  study {a.study}  model fasterrcnn_resnet50_fpn  seed {a.seed}")
    print("=" * 74)
    print(f"  device {dev}  imgsz {a.imgsz}  batch {a.batch}  epochs {a.epochs}")

    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    inf = m.roi_heads.box_predictor.cls_score.in_features
    m.roi_heads.box_predictor = FastRCNNPredictor(inf, NC + 1)
    n_all = sum(p.numel() for p in m.parameters())
    print(f"  init: torchvision COCO-pretrained backbone, new {NC+1}-class head "
          f"({n_all/1e6:.1f}M params)")
    m.to(dev)

    ds = YoloDS(tr_dir, a.imgsz, train=True)   # the only split that is flipped
    if a.smoke:
        ds.imgs = ds.imgs[:a.smoke]
    dl = DataLoader(ds, batch_size=a.batch, shuffle=True, num_workers=a.workers,
                    collate_fn=collate, pin_memory=(dev == "cuda"))
    print(f"  train images {len(ds)}  ({len(dl)} batches/epoch)")

    # Held-out validation, so checkpoint selection matches the Ultralytics arms.
    # torchvision returns a loss dict only in train() mode, so validation loss is
    # computed in train mode under no_grad. The backbone uses FrozenBatchNorm, so
    # no running statistics change; train mode only enables the RPN/ROI sampling
    # path that produces losses.
    va_dir = os.path.join(base, "images", "val")
    vdl = None
    if os.path.isdir(va_dir):
        vds = YoloDS(va_dir, a.imgsz)
        if a.smoke:
            vds.imgs = vds.imgs[:a.smoke]
        vdl = DataLoader(vds, batch_size=a.batch, shuffle=False,
                         num_workers=a.workers, collate_fn=collate)
        print(f"  val images   {len(vds)}  (selection is on VAL loss)")
    else:
        print("  *** NO VAL SPLIT FOUND - selection would fall back to train loss")
        sys.exit("refusing to train without held-out validation")

    opt = torch.optim.SGD([p for p in m.parameters() if p.requires_grad],
                          lr=a.lr, momentum=0.9, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    scaler = torch.amp.GradScaler(dev) if dev == "cuda" else None

    best, bad, t0 = float("inf"), 0, time.time()
    hist = os.path.join(run, "results.csv")
    with open(hist, "w", newline="") as fh:
        csv.writer(fh).writerow(["epoch", "time", "train_loss", "val_loss"])

    for ep in range(1, a.epochs + 1):
        m.train(); tot = 0.0; nb = 0
        for imgs, tgts, _, _ in dl:
            imgs = [i.to(dev) for i in imgs]
            tgts = [{k: v.to(dev) for k, v in t.items()} for t in tgts]
            opt.zero_grad(set_to_none=True)
            if scaler:
                with torch.amp.autocast(dev):
                    loss = sum(m(imgs, tgts).values())
                scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
            else:
                loss = sum(m(imgs, tgts).values())
                loss.backward(); opt.step()
            tot += loss.detach().item(); nb += 1
        sched.step()
        avg = tot / max(1, nb)

        vtot, vnb = 0.0, 0
        with torch.no_grad():
            for vi, vt, _, _ in vdl:
                vi = [i.to(dev) for i in vi]
                vt = [{k: v.to(dev) for k, v in t.items()} for t in vt]
                vtot += float(sum(m(vi, vt).values())); vnb += 1
        vloss = vtot / max(1, vnb)
        with open(hist, "a", newline="") as fh:
            csv.writer(fh).writerow([ep, f"{time.time()-t0:.2f}", f"{avg:.5f}", f"{vloss:.5f}"])
        print(f"  epoch {ep}/{a.epochs}  train {avg:.4f}  val {vloss:.4f}  "
              f"{(time.time()-t0)/ep:.0f}s/epoch", flush=True)
        # selection on held-out validation loss, as the Ultralytics arms select on
        # validation fitness
        if vloss < best - 1e-4:
            best, bad = vloss, 0
            torch.save(m.state_dict(), os.path.join(run, "weights", "best.pt"))
        else:
            bad += 1
            if bad >= a.patience:
                print(f"  early stop at epoch {ep} (patience {a.patience})")
                break
    torch.save(m.state_dict(), os.path.join(run, "weights", "last.pt"))
    train_min = (time.time() - t0) / 60
    print(f"\n  trained in {train_min:.1f} min")

    # ---- score: same bands, same definitions as score_bands.py ---------------
    print("\n  scoring on the TEST split ...")
    m.load_state_dict(torch.load(os.path.join(run, "weights", "best.pt"),
                                 map_location=dev))
    m.eval()
    te = YoloDS(te_dir, a.imgsz)
    if a.smoke:
        te.imgs = te.imgs[:a.smoke]
    occ = occlusion_map(a.partition_dir)
    full = fully_occluded_set(a.partition_dir)
    print(f"  occlusion partition: {len(occ)} frames    "
          f"100%-occluded list: {len(full)} frames")
    if not occ:
        # the two-stage arm is the evidence for H5, an occlusion hypothesis, so a
        # result without the occlusion axis is refused rather than written
        sys.exit("NO OCCLUSION PARTITION FOUND. H5 is an occlusion hypothesis and "
                 "this arm is its evidence.\n  pass --partition-dir, or place "
                 "occlusion_partition_s{2,3}.csv and FULLY_OCCLUDED_S1.csv beside "
                 "this script.")
    npred = 0
    halluc_frames = 0
    halluc_det = 0
    size_tot = collections.Counter(); size_hit = collections.Counter()
    size_any = collections.Counter()
    occ_tot = collections.Counter(); occ_hit = collections.Counter()
    cls_tot = collections.Counter(); cls_hit = collections.Counter()

    # AP needs EVERY prediction ranked by confidence, not only those above --conf.
    # Thresholding first would truncate the precision-recall curve and inflate AP.
    # det[class] = list of (score, image_index, box); gt[class] = {img_idx: [boxes]}
    det = collections.defaultdict(list)
    gtc = collections.defaultdict(lambda: collections.defaultdict(list))
    img_i = 0

    tl = DataLoader(te, batch_size=a.batch, shuffle=False, num_workers=a.workers,
                    collate_fn=collate)
    with torch.no_grad():
        for imgs, tgts, names, sizes in tl:
            outs = m([i.to(dev) for i in imgs])
            for out, tgt, nm, (w0, h0) in zip(outs, tgts, names, sizes):
                # unthresholded, for AP
                ab = out["boxes"].cpu().numpy()
                al = out["labels"].cpu().numpy() - 1
                asc = out["scores"].cpu().numpy()
                for b, c, sc in zip(ab, al, asc):
                    det[int(c)].append((float(sc), img_i, b))
                for gb2, gl2 in zip(tgt["boxes"].numpy(), tgt["labels"].numpy()):
                    gtc[int(gl2) - 1][img_i].append(gb2)
                img_i += 1

                keep = out["scores"] >= a.conf
                pb = out["boxes"][keep].cpu().numpy()
                pl = out["labels"][keep].cpu().numpy() - 1     # back to YOLO ids
                sx, sy = a.imgsz / w0, a.imgsz / h0
                npred += int(len(pb))
                # ungrounded-detection probe, as in score_bands.py: these frames
                # hold no ground truth, so recall cannot see them
                if nm in full:
                    halluc_frames += 1
                    halluc_det += int(len(pb))
                # the partition names the one target it measured: match on class
                # as well as frame, exactly as score_bands.py does
                vv = occ.get(nm)
                v_verdict, v_target = vv if vv else (None, None)
                for gb, gl in zip(tgt["boxes"].numpy(), tgt["labels"].numpy()):
                    gc = int(gl) - 1
                    # measure size in ORIGINAL pixels, never resized pixels
                    ow = (gb[2]-gb[0]) / sx; oh = (gb[3]-gb[1]) / sy
                    sb = band_of((ow * oh) ** 0.5)
                    size_tot[sb] += 1; cls_tot[gc] += 1
                    if nm in full and NAMES[gc] == "rc_car":
                        v = "FULLY_OCCLUDED"   # S1's list covers the red car only
                    elif v_verdict and v_target and NAMES[gc] == v_target:
                        v = v_verdict
                    else:
                        v = None
                    if v:
                        occ_tot[v] += 1
                    same = any(int(c) == gc and iou(gb, b) >= a.iou
                               for b, c in zip(pb, pl))
                    anyc = any(iou(gb, b) >= a.iou for b in pb)
                    if same:
                        size_hit[sb] += 1; cls_hit[gc] += 1
                        if v:
                            occ_hit[v] += 1
                    if anyc:
                        size_any[sb] += 1

    rows = []
    print("\n  RECALL BY TARGET SIZE (sqrt-area px, ORIGINAL resolution)")
    for b in ("<12", "12-20", "20-32", "32+"):
        if not size_tot[b]:
            continue
        rc = size_hit[b]/size_tot[b]; ra = size_any[b]/size_tot[b]
        print(f"    {b:<8}{size_tot[b]:>7}{size_hit[b]:>8}{rc:>9.3f}{ra:>11.3f}")
        rows.append(dict(tag=f"{a.study}_fasterrcnn_baseline_s{a.seed}",
                         slice=f"size:{b}", n=size_tot[b], found=size_hit[b],
                         recall=f"{rc:.4f}", recall_any_class=f"{ra:.4f}"))
    # the same occlusion bands, in the same order, as score_bands.py
    print("\n  RECALL BY MEASURED OCCLUSION")
    for b in ("FULLY_OCCLUDED", "OCCLUDED", "LIGHT", "BYPASSED", "AMBIGUOUS"):
        if occ_tot[b]:
            rc = occ_hit[b]/occ_tot[b]
            print(f"    {b:<16}{occ_tot[b]:>7}{occ_hit[b]:>8}{rc:>9.3f}")
            rows.append(dict(tag=f"{a.study}_fasterrcnn_baseline_s{a.seed}",
                             slice=f"occ:{b}", n=occ_tot[b], found=occ_hit[b],
                             recall=f"{rc:.4f}", recall_any_class=""))
    for c in sorted(cls_tot):
        rc = cls_hit[c]/cls_tot[c]
        rows.append(dict(tag=f"{a.study}_fasterrcnn_baseline_s{a.seed}",
                         slice=f"class:{NAMES[c]}", n=cls_tot[c], found=cls_hit[c],
                         recall=f"{rc:.4f}", recall_any_class=""))

    # The box budget travels with the recall, as in score_bands.py. torchvision
    # caps detections_per_img at 100 and applies NMS in the box head, unlike
    # RT-DETR's unsuppressed queries.
    n_img = len(te.imgs)
    ppi = npred / max(1, n_img)
    print(f"\n  {npred} predictions over {n_img} images ({ppi:.2f}/image) "
          f"at conf {a.conf}")
    rows.append(dict(tag=f"{a.study}_fasterrcnn_baseline_s{a.seed}",
                     slice="pred_budget", n=n_img, found=npred,
                     recall=f"{ppi:.4f}", recall_any_class=f"{a.conf:.2f}"))
    if halluc_frames:
        hr = halluc_det / halluc_frames
        print(f"  100%-occluded frames scored: {halluc_frames}   "
              f"ungrounded detections: {halluc_det} ({hr:.3f}/frame)")
        rows.append(dict(tag=f"{a.study}_fasterrcnn_baseline_s{a.seed}",
                         slice="halluc_fullyocc", n=halluc_frames,
                         found=halluc_det, recall=f"{hr:.4f}",
                         recall_any_class=f"{a.conf:.2f}"))
    else:
        print("  no 100%-occluded frames in this split - "
              "ungrounded-detection probe absent")

    bands_csv = os.path.join(a.out_root, "BANDS.csv")
    new = not os.path.exists(bands_csv)
    with open(bands_csv, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["tag", "slice", "n", "found",
                                           "recall", "recall_any_class"])
        if new:
            w.writeheader()
        w.writerows(rows)
    print(f"\n  appended {len(rows)} rows to {bands_csv}")

    # ---- AP@0.5 per class, so a RESULTS.csv row exists like every other arm ---
    # VOC-style: rank all detections by confidence, greedily match each to an
    # unmatched ground truth at IoU >= 0.5, then integrate precision over recall.
    # Duplicate detections on an already-matched ground truth are false positives.
    def ap50(c):
        gts = gtc.get(c, {})
        ngt = sum(len(v) for v in gts.values())
        if ngt == 0:
            return None
        used = {k: [False]*len(v) for k, v in gts.items()}
        tp_fp = []
        for sc, ii, b in sorted(det.get(c, []), key=lambda x: -x[0]):
            best, bi = 0.0, -1
            for j, g in enumerate(gts.get(ii, [])):
                if used[ii][j]:
                    continue
                v2 = iou(g, b)
                if v2 > best:
                    best, bi = v2, j
            if best >= 0.5 and bi >= 0:
                used[ii][bi] = True; tp_fp.append(1)
            else:
                tp_fp.append(0)
        if not tp_fp:
            return 0.0
        tp = 0; fp = 0; pr = []
        for t in tp_fp:
            tp += t; fp += (1 - t)
            pr.append((tp/ngt, tp/max(1, tp+fp)))
        # all-point interpolation
        apv = 0.0; prev_r = 0.0; peak = 0.0
        for r, p in pr:
            peak = max(peak, p)
        mrec = [0.0] + [r for r, _ in pr]
        mpre = [0.0] + [p for _, p in pr]
        for i in range(len(mpre)-2, -1, -1):
            mpre[i] = max(mpre[i], mpre[i+1])
        for i in range(1, len(mrec)):
            apv += (mrec[i] - mrec[i-1]) * mpre[i]
        return apv

    aps = {c: ap50(c) for c in range(NC)}
    valid = [v for v in aps.values() if v is not None]
    mAP50 = sum(valid)/len(valid) if valid else 0.0
    tot_gt = sum(cls_tot.values()); tot_hit = sum(cls_hit.values())
    rec_all = tot_hit/max(1, tot_gt)
    print(f"\n  mAP@0.5 {mAP50:.4f}   recall {rec_all:.4f}")
    for c in range(NC):
        if aps[c] is not None:
            print(f"    {NAMES[c]:<18} AP50 {aps[c]:.4f}")

    res = os.path.join(a.out_root, "RESULTS.csv")
    base_row = dict(study=a.study, model="fasterrcnn", preset="baseline",
                    seed=a.seed, imgsz=a.imgsz, batch=a.batch, epochs=a.epochs,
                    train_frames=len(ds), test_frames=len(te),
                    train_min=f"{train_min:.1f}",
                    gpu=(torch.cuda.get_device_name(0) if dev == "cuda" else "cpu"))
    out_rows = [dict(base_row, slice="ALL", n="", mAP50=f"{mAP50:.4f}",
                     mAP50_95="", precision="", recall=f"{rec_all:.4f}")]
    for c in range(NC):
        if aps[c] is not None:
            out_rows.append(dict(base_row, slice=f"class:{NAMES[c]}",
                                 n=cls_tot[c], mAP50=f"{aps[c]:.4f}",
                                 mAP50_95="", precision="",
                                 recall=f"{cls_hit[c]/max(1,cls_tot[c]):.4f}"))
    flds = list(out_rows[0])
    newf = not os.path.exists(res)
    with open(res, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=flds)
        if newf:
            w.writeheader()
        w.writerows(out_rows)
    print(f"  appended {len(out_rows)} rows to {res}")
    print("  NOTE: mAP50_95 is left blank - this scorer computes AP at IoU 0.5 only;")
    print("        the endpoint is band recall, not mAP.")
    print("ALL DONE")


if __name__ == "__main__":
    main()
