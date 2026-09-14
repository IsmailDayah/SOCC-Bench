"""Tier 4 (data regime and inference): the 2x2 and its added arms, judged against
the pre-registered plan.

    python code/analyse_tier4.py
    python code/analyse_tier4.py --bands <BANDS csv> [--out report.md]

Tags are parsed on all five fields - study, arm, preset, seed and inference mode -
because Tier 4 compares cells that differ ONLY in study, ONLY in preset or ONLY in
inference mode; a parser that dropped any field would merge two cells and average
them into a difference of nearly zero.

THE CONTRASTS

    H7   = M4/augmented       - M4/baseline     augmentation, split held fixed (PRIMARY)
    H8   = M4/baseline        - M/baseline      split and checkpoint selection, preset held fixed
    H9   = M4/augmented       - M/baseline      the full training recipe
    H10  = M4/baseline tiled  - M4/baseline     inference mode, same weights
    H10b = M4/augmented tiled - M4/augmented    does tiling add to augmentation?
    H11  = M4/p2+augmented    - M4/augmented    P2 once it has small examples to learn from
    H12  = M4/yolov8s+aug     - M4/augmented    model capacity

H8 is the control: without it an H9 gain would be confounded across augmentation,
checkpoint selection and a 4.2% smaller training set.

A gain from an arm emitting more than 2.0x its comparator's boxes per image is
disqualified. A cell with fewer than three COMPLETED training runs (per
results/protocol/runs_t4/RUN_STATUS.csv) is underpowered and gets no interval.
An absent cell is reported as not run, never as a refuted hypothesis.
"""
import argparse
import collections
import csv
import os
import re
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from analyse_tier2 import bc_ci, load_bands  # noqa: E402

RES = os.path.join(ROOT, "results", "protocol", "results")
PRIMARY = "size:<12"
FLOOR = 0.02            # pre-specified, absolute recall
BUDGET_MAX = 2.0        # pre-specified box-budget guard
SEEDS = ["42", "123", "456"]

# the Tier 1 baseline is tagged `v8n`
ALIASES = {"v8n": "yolov8n", "yolov8n": "yolov8n"}

# The preset is matched as an explicit alternation rather than [a-z]+, because a
# greedy character class can eat the tail of an arm name and shift every field.
# Each run is scored twice - whole-frame by score_bands.py and tiled by
# tile_probe.py - and the tiled rows carry a `_tiled<N>` suffix, so inference mode
# is captured as a factor.
TAG_RE = re.compile(
    r"^(?P<study>M4|M)_(?P<arm>.+?)_(?P<preset>baseline|augmented)"
    r"_s(?P<seed>\d+)(?:_tiled(?P<tiles>\d+))?$")

# cell -> (study, preset, tiles); tiles 0 = whole frame, 2 = 2x2 tiled inference
CELLS = collections.OrderedDict([
    ("M/baseline",             ("M",  "baseline",  0)),
    ("M4/baseline",            ("M4", "baseline",  0)),
    ("M4/augmented",           ("M4", "augmented", 0)),
    ("M4/p2+augmented",        ("M4", "augmented", 0)),
    ("M4/yolov8s+aug",         ("M4", "augmented", 0)),
    # the same weights, evaluated tiled - inference mode, not a different model
    ("M4/baseline TILED",      ("M4", "baseline",  2)),
    ("M4/augmented TILED",     ("M4", "augmented", 2)),
    ("M4/p2+augmented TILED",  ("M4", "augmented", 2)),
])

# the architecture behind each cell: several cells share a preset and differ
# only by arm, so the arm cannot be inferred from the cell key
CELL_ARM = {
    "M/baseline": "yolov8n", "M4/baseline": "yolov8n",
    "M4/augmented": "yolov8n", "M4/p2+augmented": "yolov8n-p2",
    "M4/yolov8s+aug": "yolov8s",
    "M4/baseline TILED": "yolov8n", "M4/augmented TILED": "yolov8n",
    "M4/p2+augmented TILED": "yolov8n-p2",
}

# H7 is the single pre-registered primary; the rest are secondary. The plan names
# a Holm correction for the secondary family. It is not applied: every interval
# here is a bootstrap over three seed pairs, so a wider interval cannot change a
# verdict whose three paired differences share a sign, and every other verdict
# already includes zero.
CONTRASTS = [
    ("H7", "M4/augmented", "M4/baseline",
     "Small-object augmentation raises sub-12px recall", "supported",
     "augmentation, split held fixed", "PRIMARY"),
    ("H8", "M4/baseline", "M/baseline",
     "Small-object-aware validation alone is not sufficient", "null",
     "split and checkpoint selection, preset held fixed", "secondary"),
    ("H9", "M4/augmented", "M/baseline",
     "The full training recipe beats the standard recipe", "supported",
     "the full training recipe", "secondary"),
    ("H10", "M4/baseline TILED", "M4/baseline",
     "Tiled inference raises sub-12px recall within the budget guard",
     "supported", "inference mode, same weights", "secondary"),
    ("H10b", "M4/augmented TILED", "M4/augmented",
     "Tiling still helps once training already emphasises small objects",
     "supported", "tiling on top of augmentation", "secondary"),
    ("H11", "M4/p2+augmented", "M4/augmented",
     "P2 helps once its stride-4 head has in-band examples to learn from",
     "supported", "P2 x augmentation", "secondary"),
    ("H12", "M4/yolov8s+aug", "M4/augmented",
     "Model scale moves the band", "null",
     "model capacity", "secondary"),
]

STATUS_CSV = os.path.join(ROOT, "results", "protocol", "runs_t4", "RUN_STATUS.csv")
BASELINE_BANDS = os.path.join(RES, "tier1_backfill", "BANDS_tier1_backfill.csv")


def parse_tag(tag):
    """tag -> (study, arm, preset, seed, tiles) or five Nones."""
    m = TAG_RE.match(tag)
    if not m:
        return (None, None, None, None, None)
    return (m.group("study"), ALIASES.get(m.group("arm"), m.group("arm")),
            m.group("preset"), m.group("seed"),
            int(m.group("tiles") or 0))


def tag_for(cell, seed, tags, arm=None):
    """Exact study+arm+preset+seed+tiles match. Ambiguity is fatal, never resolved."""
    study, preset, tiles = CELLS[cell]
    want_arm = arm or CELL_ARM.get(cell, "yolov8n")
    hits = [t for t in tags
            if parse_tag(t) == (study, want_arm, preset, str(seed), tiles)]
    if len(hits) > 1:
        sys.exit(f"AMBIGUOUS: {len(hits)} tags claim {cell} seed={seed}: {hits}")
    return hits[0] if hits else None


def val(bands, tag, slc, idx=2):
    if tag is None or tag not in bands or slc not in bands[tag]:
        return None
    return bands[tag][slc][idx]


def load_run_status(path=STATUS_CSV):
    """run name (no `_tiledN` suffix, no `baseline` in M4 names) -> row; {} if absent."""
    if not os.path.exists(path):
        return {}
    return {r["run"]: r for r in csv.DictReader(open(path, encoding="utf-8"))}


def run_complete(tag, status):
    """True when the training run behind a scored tag ran to completion.
    Tier 1 (study M) runs are not in the Tier 4 status file and all completed."""
    if tag is None:
        return False
    base = re.sub(r"_tiled\d+$", "", tag).replace("_baseline_s", "_s")
    row = status.get(base)
    if row is None:
        return True
    return row["status"].startswith("completed")


def run_note(tag, status):
    base = re.sub(r"_tiled\d+$", "", tag or "").replace("_baseline_s", "_s")
    row = status.get(base)
    if row is None or row["status"].startswith("completed"):
        return ""
    return f"s{row['seed']}: stopped at epoch {row['epochs_run']}"


def cell_mean(bands, tags, cell, slc, idx=2, fallback=None):
    """Mean of one slice over the seeds present; `fallback` supplies rows the main
    ledger lacks (the Tier 1 baseline was scored before pred_budget existed)."""
    vs = []
    for s in SEEDS:
        v = val(bands, tag_for(cell, s, tags), slc, idx)
        if v is None and fallback is not None:
            v = val(fallback, tag_for(cell, s, list(fallback)), slc, idx)
        if v is not None:
            vs.append(v)
    return (st.mean(vs), len(vs)) if vs else (None, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bands", default=os.path.join(RES, "tier4", "BANDS_tier4.csv"))
    ap.add_argument("--baseline-bands", default=BASELINE_BANDS,
                    help="ledger supplying the Tier 1 baseline's box budget")
    ap.add_argument("--out", default=None,
                    help="also write the report to this markdown file")
    a = ap.parse_args()

    status = load_run_status()
    bands = load_bands(a.bands)
    base_bands = load_bands(a.baseline_bands) if os.path.exists(a.baseline_bands) else None
    tags = list(bands)

    # tags that cannot be parsed are reported, never dropped in silence
    unparsed = [t for t in tags if parse_tag(t)[0] is None]

    out = []

    def say(s=""):
        out.append(s)
        print(s)

    say("# Tier 4 - data regime and inference, judged against the pre-registered plan")
    say()
    say("Primary endpoint: **recall in the < 12 px band**, seed-paired, "
        f"bias-corrected bootstrap 95% CI over seed pairs, floor **+{FLOOR}**.")
    say()
    if unparsed:
        say(f"> **{len(unparsed)} tag(s) not parsed and excluded:** "
            + ", ".join(f"`{t}`" for t in sorted(unparsed)))
        say()

    # ---- coverage -----------------------------------------------------------
    say("## Coverage")
    say()
    say("| cell | seeds found | missing | training completed | truncated runs |")
    say("|---|---|---|---|---|")
    complete = {}
    for cell in CELLS:
        found, missing, done, notes = [], [], [], []
        for s in SEEDS:
            tg = tag_for(cell, s, tags)
            (found if tg else missing).append(s)
            if tg and run_complete(tg, status):
                done.append(s)
            elif tg:
                notes.append(run_note(tg, status))
        complete[cell] = done
        say(f"| `{cell}` | {', '.join(found) or '-'} | "
            f"{', '.join(missing) or '-'} | {', '.join(done) or '-'} | "
            f"{'; '.join(n for n in notes if n) or '-'} |")
    say()
    if status:
        say("Per-run epochs, exit codes and completion are in "
            "`results/protocol/runs_t4/RUN_STATUS.csv`. A run is complete when "
            "the trainer exited normally or early stopping ran its course "
            "(epochs run = best epoch + patience 30). Runs that stopped before that "
            "were scored from the best checkpoint they had reached, and any cell "
            "with fewer than three completed seeds is reported below as "
            "**underpowered, with no confidence interval**.")
        say()

    # ---- validity: box budget ----------------------------------------------
    say("## Validity check - box budget and ungrounded detections")
    say()
    say("Recall is blind to false positives. An arm emitting more boxes can raise")
    say(f"recall without seeing anything new, so any arm above **{BUDGET_MAX}x** its")
    say("comparator's budget is disqualified from claiming a gain. Ungrounded")
    say("detections are those fired on the 159 test frames whose target is fully hidden.")
    say()
    say("| cell | boxes/image | vs `M/baseline` | ungrounded dets per frame |")
    say("|---|---|---|---|")
    budget = {}
    for cell in CELLS:
        b, _ = cell_mean(bands, tags, cell, "pred_budget", 2, base_bands)
        h, _ = cell_mean(bands, tags, cell, "halluc_fullyocc", 2, base_bands)
        if b is None:
            say(f"| `{cell}` | - | - | - |")
            continue
        budget[cell] = b
        base = budget.get("M/baseline")
        rel = f"{b / base:.2f}x" if base else "-"
        say(f"| `{cell}` | {b:.2f} | {rel} | {h:.4f} |" if h is not None else
            f"| `{cell}` | {b:.2f} | {rel} | - |")
    say()

    # ---- primary endpoint ---------------------------------------------------
    say(f"## Primary endpoint - recall at {PRIMARY}")
    say()
    say("| contrast | what it isolates | n | from | to | mean paired diff "
        "| 95% CI | verdict |")
    say("|---|---|---|---|---|---|---|---|")
    verdicts = {}
    for h, arm_cell, ref_cell, _text, _expect, isolates, rank in CONTRASTS:
        pairs, a_vals, b_vals = [], [], []
        for s in SEEDS:
            ta, tb = tag_for(arm_cell, s, tags), tag_for(ref_cell, s, tags)
            va, vb = val(bands, ta, PRIMARY), val(bands, tb, PRIMARY)
            if va is None or vb is None:
                continue
            pairs.append(va - vb); a_vals.append(va); b_vals.append(vb)
        if not pairs:
            say(f"| **{h}** | {isolates} | 0 | - | - | - | - | **NOT RUN** |")
            verdicts[h] = "not run"
            continue
        d = st.mean(pairs)
        lo, hi = bc_ci(pairs)
        n_done = min(len(complete.get(arm_cell, [])), len(complete.get(ref_cell, [])))
        if n_done < 3:
            short = [c for c in (arm_cell, ref_cell) if len(complete.get(c, [])) < 3]
            ci = "no interval"
            v = (f"**UNDERPOWERED** - {n_done} of 3 seeds completed in "
                 + " / ".join(f"`{c}`" for c in short))
        elif lo is None:
            ci, v = "no interval", f"**UNDERPOWERED (n={len(pairs)})**"
        elif lo > 0 and d > FLOOR:
            ci, v = f"[{lo:+.4f}, {hi:+.4f}]", "**SUPPORTED**"
        elif hi < 0:
            ci, v = f"[{lo:+.4f}, {hi:+.4f}]", "**REFUTED (worse)**"
        elif lo > 0:
            ci = f"[{lo:+.4f}, {hi:+.4f}]"
            v = f"detectable but **below the +{FLOOR} floor**"
        else:
            ci, v = f"[{lo:+.4f}, {hi:+.4f}]", "**NULL** (CI includes 0)"
        # the budget guard overrides a positive verdict, and says so
        if "SUPPORTED" in v and budget.get(arm_cell) and budget.get(ref_cell):
            ratio = budget[arm_cell] / budget[ref_cell]
            if ratio > BUDGET_MAX:
                v = (f"**DISQUALIFIED** - box budget {ratio:.2f}x "
                     f"(> {BUDGET_MAX}x)")
        verdicts[h] = v
        say(f"| **{h}** | {isolates} | {len(pairs)} | {st.mean(b_vals):.4f} "
            f"| {st.mean(a_vals):.4f} | {d:+.4f} | {ci} | {v} |")
    say()

    # ---- hypotheses ---------------------------------------------------------
    say("## Hypotheses, judged against the pre-registration")
    say()
    for h, _a, _b, text, expect, _iso, rank in CONTRASTS:
        got = verdicts.get(h, "not run")
        if "NOT RUN" in got.upper() or "UNDERPOWERED" in got.upper():
            say(f"- **{h}** ({rank}) - {text}. Predicted: *{expect}*. "
                f"**NOT MEASURED** ({got}) - no conclusion is drawn.")
            continue
        met = ("SUPPORTED" in got) if expect == "supported" \
            else ("NULL" in got or "floor" in got)
        say(f"- **{h}** ({rank}) - {text}. Predicted: *{expect}*. Observed: {got}. "
            f"**{'As predicted' if met else 'CONTRADICTED'}.**")
    say()

    # ---- the pre-committed proposal decisions -------------------------------
    say("## Proposal conditions, fixed before any Tier 4 run")
    say()
    say("A training recipe is proposed only if all four hold:")
    say()
    conds = [
        ("H9 supported (CI excludes 0, diff > +0.02)",
         "SUPPORTED" in verdicts.get("H9", "")),
        (f"box budget within {BUDGET_MAX}x of the Tier 1 baseline",
         bool(budget.get("M4/augmented") and budget.get("M/baseline"))
         and budget["M4/augmented"] / budget["M/baseline"] <= BUDGET_MAX),
        ("no material inference-speed penalty (architecture unchanged)", True),
        ("H7 supported, so the gain is attributable to augmentation",
         "SUPPORTED" in verdicts.get("H7", "")),
    ]
    for n, ok in conds:
        say(f"- {'**MET**' if ok else 'NOT met'} - {n}")
    say()
    say("**All four met - the training recipe is proposed.**" if all(ok for _, ok in conds)
        else "**Not all met - the training recipe is not proposed.**")
    say()
    say("An inference-time intervention (tiling) is proposed only if all four hold:")
    say()
    hb, _ = cell_mean(bands, tags, "M4/baseline", "halluc_fullyocc")
    ht, _ = cell_mean(bands, tags, "M4/baseline TILED", "halluc_fullyocc")
    conds_b = [
        ("H10 supported (CI excludes 0, diff > +0.02)",
         "SUPPORTED" in verdicts.get("H10", "")),
        (f"box budget within {BUDGET_MAX}x of the whole-frame comparator",
         bool(budget.get("M4/baseline TILED") and budget.get("M4/baseline"))
         and budget["M4/baseline TILED"] / budget["M4/baseline"] <= BUDGET_MAX),
        ("ungrounded detections not increased by tiling",
         hb is not None and ht is not None and ht <= hb),
        ("inference cost measured on identical hardware", False),
    ]
    for n, ok in conds_b:
        say(f"- {'**MET**' if ok else 'NOT met'} - {n}")
    say()
    say("**All four met - tiled inference is proposed.**" if all(ok for _, ok in conds_b)
        else "**Not all met - tiled inference is not proposed.** Tiling was not timed; "
             "its cost is estimated as four passes of the network per frame.")
    say()
    say("Every cell is also compared with the Tier 1 baseline, seed-paired, by "
        "`code/make_tier4_table.py`.")
    say()

    # ---- secondary ----------------------------------------------------------
    say("## Secondary (exploratory - no inferential weight)")
    say()
    slices = ["size:<12", "size:12-20", "size:20-32", "size:32+",
              "occ:OCCLUDED", "occ:LIGHT", "occ:BYPASSED", "occ:AMBIGUOUS"]
    say("| band | " + " | ".join(f"`{c}`" for c in CELLS) + " |")
    say("|---" * (len(CELLS) + 1) + "|")
    for slc in slices:
        cells = []
        for cell in CELLS:
            m, _ = cell_mean(bands, tags, cell, slc)
            cells.append(f"{m:.4f}" if m is not None else "-")
        say(f"| {slc} | " + " | ".join(cells) + " |")
    say()
    say("The occlusion and size axes are **confounded by venue**. No claim ranks "
        "the two axes against each other.")

    if a.out:
        d = os.path.dirname(a.out)
        if d:
            os.makedirs(d, exist_ok=True)
        with open(a.out, "w", encoding="utf-8", newline="") as fh:
            fh.write("\n".join(out) + "\n")
        print(f"\n  wrote {a.out}")


if __name__ == "__main__":
    main()
