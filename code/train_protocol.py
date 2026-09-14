"""Train one Ultralytics model on one study split, then score it on the test split.

    python code/train_protocol.py --study M --model yolov8n --seed 42
    python code/train_protocol.py --study M4 --model yolov8n-p2 --preset augmented --seed 42
    python code/train_protocol.py --study M --model yolov8n --check      # validate only

Each invocation trains ONE (study, model, preset, seed) and appends one row per
evaluation slice to results/protocol/RESULTS.csv: overall mAP50 / mAP50-95,
per-class AP, and the number of test boxes in every size and occlusion band.
Per-band RECALL comes from the detection-level pass in score_bands.py.

PRESETS
  baseline   Ultralytics defaults, which already include mosaic, horizontal flip,
             translation, scale 0.5, HSV jitter and random erasing - with one
             corpus-specific change, kept in every preset:
                 hsv_h = 0.005   The default 0.015 rotates hue by +-5.4 degrees.
                                 At 12-20 px that walks a red car toward orange and
                                 a yellow all-terrain vehicle toward green; at these
                                 sizes colour carries the class, so the default
                                 would corrupt class identity.
  augmented  baseline + scale 0.75, rotation 8 degrees and close_mosaic 15. Scale
             0.75 widens the random shrink factor to 0.25x, which raises the share
             of sub-12 px boxes the network is shown from 4.8% to 17.6%.

INITIALISATION
  `YOLO("arch.yaml")` builds a graph with random weights, and the `pretrained`
  training argument does not change that. Every yaml-built arm therefore has the
  COCO tensors whose name and shape still match transferred from yolov8n.pt, so the
  shared backbone is identical across arms and only the new components (the P2
  branch, the CBAM modules) start from random. Measured transfer: baseline
  355/355, CBAM 270/364, P2 219/437, P2+CBAM 219/449. yolov8s and RT-DETR-L load
  their own COCO-pretrained checkpoints.

Batch and workers are explicit; AutoBatch (batch -1) is refused.
"""
import argparse
import collections
import csv
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROTO = os.path.join(ROOT, "results", "protocol")


def _find_arch(name):
    """Locate an architecture yaml in the repository, next to this script, in a
    container's /workspace/models, or under the working directory."""
    here = os.path.dirname(os.path.abspath(__file__))
    for c in (os.path.join(ROOT, "code", "models", name),   # repository checkout
              os.path.join(here, "models", name),                  # next to the script
              os.path.join("/workspace", "models", name),          # container layout
              os.path.join(os.getcwd(), "models", name)):          # working directory
        if os.path.exists(c):
            return c
    return f"__MISSING__{name}"


ARMS = {
    "yolov8n":         None,                                     # stock .pt
    "yolov8n-p2":      "yolov8n-p2.yaml",                         # ships with ultralytics
    "yolov8n-cbam":    _find_arch("yolov8n-cbam.yaml"),
    "yolov8n-p2-cbam": _find_arch("yolov8n-p2-cbam.yaml"),
    # model scale: yolov8s loads its own COCO-pretrained checkpoint
    "yolov8s":         None,                                     # stock .pt, 11M
    # paradigm comparison: RT-DETR ships a COCO-pretrained checkpoint
    "rtdetr-l":        None,
}
PRETRAIN_SRC = "yolov8n.pt"

# arms that are a different model family and load through their own class
FAMILY = {"rtdetr-l": "RTDETR"}


def build_model(name):
    """Return (YOLO, provenance string). Never silently trains from scratch."""
    from ultralytics import YOLO
    if name not in ARMS:
        sys.exit(f"unknown --model {name!r}. known arms: {sorted(ARMS)}")
    cfg = ARMS[name]

    if name in FAMILY:
        import ultralytics
        klass = getattr(ultralytics, FAMILY[name])
        ckpt = f"{name}.pt"
        m = klass(ckpt)
        n = sum(p.numel() for p in m.model.parameters())
        return m, f"{ckpt} ({FAMILY[name]}, COCO-pretrained, {n/1e6:.1f}M params)"

    if cfg is None:
        return YOLO(f"{name}.pt"), f"{name}.pt (COCO-pretrained, 355/355)"

    # CBAM lives in ultralytics.nn.modules, but parse_model resolves layer names
    # through the globals of ultralytics.nn.tasks, where it is not imported.
    # Without this a CBAM yaml fails with KeyError: 'CBAM'.
    import ultralytics.nn.tasks as _T
    from ultralytics.nn.modules import CBAM as _CBAM
    _T.CBAM = _CBAM

    if cfg.startswith("__MISSING__"):
        want = cfg.replace("__MISSING__", "")
        sys.exit(f"CANNOT FIND architecture yaml {want!r}. Looked in: "
                 f"{os.path.join(ROOT, 'code', 'models')}, "
                 f"{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')}, "
                 f"/workspace/models, {os.path.join(os.getcwd(), 'models')}. "
                 f"Provide the yaml - do NOT fall back to a stock model.")
    if not os.path.exists(cfg) and not cfg.startswith("yolov8"):
        sys.exit(f"missing architecture yaml: {cfg}")
    m = YOLO(cfg, task="detect")
    before = m.model.state_dict()["model.0.conv.weight"].clone()
    m.load(PRETRAIN_SRC)
    after = m.model.state_dict()["model.0.conv.weight"]
    if (before - after).abs().max().item() == 0:
        # the backbone did not change -> the transfer silently did nothing
        sys.exit(f"REFUSING TO TRAIN {name}: .load({PRETRAIN_SRC}) transferred "
                 f"nothing. An untransferred arm would compare random "
                 f"initialisation against a pretrained baseline.")
    ref = YOLO(PRETRAIN_SRC).model.state_dict()
    own = m.model.state_dict()
    n = sum(1 for k, v in own.items()
            if k in ref and ref[k].shape == v.shape and (ref[k] == v).all())
    return m, f"{os.path.basename(cfg)} + COCO backbone transfer ({n}/{len(own)})"
RESULTS = os.path.join(PROTO, "RESULTS.csv")
W, H = 960, 720

PRESETS = {
    # Ultralytics defaults, except the hue fix
    "baseline": dict(hsv_h=0.005),
    # the small-object augmentation, declared in full
    "augmented": dict(hsv_h=0.005, scale=0.75, degrees=8.0, translate=0.1,
                      mosaic=1.0, close_mosaic=15, fliplr=0.5, flipud=0.0),
}


def workers_for(n_train, batch):
    """Dataloader workers from batches per epoch, not from the CPU count: on short
    epochs extra workers only add start-up cost. On Windows validation-time worker
    processes are unreliable, so the answer there is 0."""
    if os.name == "nt":
        return 0
    b = max(1, n_train // max(1, batch))
    if b < 50:
        return 2
    if b < 300:
        return 4
    return 8


def size_band(w, h):
    a = ((w * W) * (h * H)) ** 0.5
    return "<12" if a < 12 else "12-20" if a < 20 else "20-32" if a < 32 else "32+"


def occlusion_map():
    """image -> verdict, from the measured occlusion partition, never from flight names."""
    m = {}
    for sc, ds in (("s2", "dataset/01_Dataset_SOCC_S2"), ("s3", "dataset/01_Dataset_SOCC_S3")):
        p = os.path.join(ROOT, ds, f"occlusion_partition_{sc}.csv")
        if os.path.exists(p):
            for r in csv.DictReader(open(p, encoding="utf-8")):
                m[r["image"]] = r["verdict"]
    return m


def append_rows(rows):
    new = not os.path.exists(RESULTS)
    os.makedirs(os.path.dirname(RESULTS), exist_ok=True)
    with open(RESULTS, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", required=True)
    ap.add_argument("--model", default="yolov8n")
    ap.add_argument("--preset", default="baseline", choices=sorted(PRESETS))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--patience", type=int, default=30)
    ap.add_argument("--batch", type=int, default=0,
                    help="0 = pick from GPU memory. -1 (AutoBatch) is refused.")
    ap.add_argument("--workers", type=int, default=0, help="0 = from batches/epoch")
    ap.add_argument("--device", default="0")
    ap.add_argument("--data", default="",
                    help="explicit dataset yaml; needed whenever this script runs "
                         "outside the repository, since ROOT is derived from its "
                         "location")
    ap.add_argument("--out-root", default="",
                    help="where runs and RESULTS.csv go (default: the repo)")
    ap.add_argument("--check", action="store_true",
                    help="validate the dataset, yaml and settings WITHOUT training")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run to prove the pipeline; not written to RESULTS.csv")
    a = ap.parse_args()
    if a.batch == -1:
        sys.exit("REFUSING batch=-1. Pass an explicit batch size.")

    global PROTO, RESULTS
    if a.out_root:
        PROTO = os.path.abspath(a.out_root)
        RESULTS = os.path.join(PROTO, "RESULTS.csv")
    if a.data:
        yml = os.path.abspath(a.data)
        out = os.path.dirname(yml)
    else:
        out = os.path.join(PROTO, f"study_{a.study}")
        yml = os.path.join(out, f"socc_study_{a.study}.yaml")
    if not os.path.exists(yml):
        sys.exit(f"missing {yml}\n  populate the study from its assignment.csv first")

    n_train = len(os.listdir(os.path.join(out, "images", "train")))
    n_test = len(os.listdir(os.path.join(out, "images", "test")))

    import torch
    from ultralytics import YOLO
    vram = (torch.cuda.get_device_properties(0).total_memory / 1e9
            if torch.cuda.is_available() else 0)
    if not a.batch:
        # at imgsz 1280: 8 GB fits yolov8n at batch 4-6
        a.batch = 4 if vram < 12 else 8 if vram < 20 else 16
    if not a.workers:
        a.workers = workers_for(n_train, a.batch)
    epochs = 2 if a.smoke else a.epochs

    tag = f"{a.study}_{a.model}_{a.preset}_s{a.seed}" + ("_smoke" if a.smoke else "")
    print("=" * 74)
    print(f"TRAIN  study {a.study}  model {a.model}  preset {a.preset}  seed {a.seed}")
    print("=" * 74)
    print(f"  gpu          {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
          f"  {vram:.0f} GB")
    print(f"  train/test   {n_train} / {n_test} frames")
    print(f"  imgsz {a.imgsz}  batch {a.batch}  workers {a.workers}"
          f"  ({n_train // max(1, a.batch)} batches/epoch)")
    print(f"  epochs {epochs}  patience {a.patience}  seed {a.seed}")
    print(f"  preset       {a.preset}  {PRESETS[a.preset]}")
    print("=" * 74)

    if a.check:
        # load the dataset the way Ultralytics will, and prove every split resolves
        from ultralytics.data.utils import check_det_dataset
        d = check_det_dataset(yml)
        print("")
        print("  dataset resolved by ultralytics:")
        for k in ("train", "val", "test"):
            v = d.get(k)
            ok = v and os.path.exists(v if isinstance(v, str) else v[0])
            print(f"    {k:<6}{'OK  ' if ok else 'FAIL'}  {v}")
            if not ok:
                sys.exit(f"REFUSING: split '{k}' does not resolve. Populate "
                         f"study {a.study} on this machine first.")
        print(f"    nc={d.get('nc')}  names={d.get('names')}")
        n_lab = sum(len(os.listdir(os.path.join(out, 'labels', sp)))
                    for sp in ('train', 'val', 'test'))
        print("")
        print(f"  labels on disk: {n_lab}")
        print(f"  would train: imgsz {a.imgsz} batch {a.batch} workers {a.workers} "
              f"epochs {epochs} seed {a.seed}")
        print(f"  would write: {os.path.join(PROTO, 'runs', tag)}")
        print("")
        print("  CHECK ONLY - nothing trained, nothing written.")
        return

    t0 = time.time()
    m, init_note = build_model(a.model)
    print(f"  init: {init_note}")
    m.train(data=yml, imgsz=a.imgsz, epochs=epochs, patience=a.patience,
            batch=a.batch, workers=a.workers, device=a.device,
            project=os.path.join(PROTO, "runs"), name=tag, exist_ok=True,
            seed=a.seed, deterministic=True, val=True, plots=not a.smoke,
            **PRESETS[a.preset])
    train_s = time.time() - t0
    best = os.path.join(PROTO, "runs", tag, "weights", "best.pt")
    print(f"\n  trained in {train_s/60:.1f} min  ->  {best}")

    # ---- score on the test split --------------------------------------------
    print("\n  scoring on the TEST split ...")
    mm = YOLO(best)
    r = mm.val(data=yml, split="test", imgsz=a.imgsz, batch=a.batch,
               device=a.device, workers=a.workers, plots=False, verbose=False)
    names = r.names if hasattr(r, "names") else {}
    rows = []
    base = dict(study=a.study, model=a.model, preset=a.preset, seed=a.seed,
                imgsz=a.imgsz, batch=a.batch, epochs=epochs,
                train_frames=n_train, test_frames=n_test,
                train_min=f"{train_s/60:.1f}",
                gpu=(torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"))
    rows.append(dict(base, slice="ALL", n="", mAP50=f"{r.box.map50:.4f}",
                     mAP50_95=f"{r.box.map:.4f}", precision=f"{r.box.mp:.4f}",
                     recall=f"{r.box.mr:.4f}"))
    # r.box.ap50 is a numpy array: test its length, never its truth value
    ap50 = getattr(r.box, "ap50", None)
    ap = getattr(r.box, "ap", None)
    if ap50 is not None and len(ap50):
        for i in range(len(ap50)):
            rows.append(dict(base, slice=f"class:{names.get(i, i)}", n="",
                             mAP50=f"{float(ap50[i]):.4f}",
                             mAP50_95=(f"{float(ap[i]):.4f}"
                                       if ap is not None and i < len(ap) else ""),
                             precision="", recall=""))

    # composition of the test set, so a reader can see what each slice is worth
    occ = occlusion_map()
    band = collections.Counter()
    occb = collections.Counter()
    ld = os.path.join(out, "labels", "test")
    for f in os.listdir(ld):
        img = os.path.splitext(f)[0] + ".jpg"
        v = occ.get(img)
        for line in open(os.path.join(ld, f), encoding="utf-8"):
            q = line.split()
            if len(q) == 5:
                band[size_band(float(q[3]), float(q[4]))] += 1
                if v:
                    occb[v] += 1
    for k, n in sorted(band.items()):
        rows.append(dict(base, slice=f"size_band:{k}", n=n, mAP50="", mAP50_95="",
                         precision="", recall=""))
    for k, n in sorted(occb.items()):
        rows.append(dict(base, slice=f"occ_band:{k}", n=n, mAP50="", mAP50_95="",
                         precision="", recall=""))

    print(f"\n  mAP50 {r.box.map50:.4f}   mAP50-95 {r.box.map:.4f}   "
          f"P {r.box.mp:.4f}   R {r.box.mr:.4f}")
    print(f"  test composition  size {dict(band)}")
    if occb:
        print(f"                    occ  {dict(occb)}")
    if a.smoke:
        print("\n  SMOKE RUN - not written to RESULTS.csv")
        return
    append_rows(rows)
    print(f"\n  appended {len(rows)} rows to {os.path.relpath(RESULTS, ROOT)}")
    print("\n  Per-band recall comes from the detection-level pass:")
    print("  python code/score_bands.py --weights <best.pt> --study <study>")


if __name__ == "__main__":
    main()
