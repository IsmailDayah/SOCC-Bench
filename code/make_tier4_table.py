"""Tier 4 against the Tier 1 baseline: every configuration, recall below 12 px.

    python code/make_tier4_table.py
      -> results/protocol/results/tier4/TIER4_VS_BASELINE.csv  (what make_tier4_figure.py plots)
    python code/make_tier4_table.py --out table.md             (also write the markdown table)

Same ledger, bootstrap and completion rule as code/analyse_tier4.py: seed-paired
differences in sub-12 px recall against the Tier 1 baseline (`M_v8n_baseline_s*`),
a bias-corrected bootstrap interval over the seed pairs and the pre-registered
+0.02 floor. A configuration with fewer than three COMPLETED training runs is
reported as underpowered with no interval, whatever its seed-paired mean looks
like; which runs completed is read from results/protocol/runs_t4/RUN_STATUS.csv.
Recall values are the mean over every seed scored, truncated runs included.
"""
import argparse
import csv
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "code"))
from analyse_tier2 import bc_ci, load_bands  # noqa: E402
from analyse_tier4 import load_run_status, run_complete  # noqa: E402

RES = os.path.join(ROOT, "results", "protocol", "results", "tier4")
BANDS = os.path.join(RES, "BANDS_tier4.csv")
CSV_OUT = os.path.join(RES, "TIER4_VS_BASELINE.csv")
SEEDS = ["42", "123", "456"]
FLOOR = 0.02
PRIMARY = "size:<12"

# label -> run-tag pattern ({s} = seed)
ARMS = [
    ("Small-object augmentation",           "M4_yolov8n_augmented_s{s}"),
    ("Small-object-aware selection",        "M4_yolov8n_baseline_s{s}"),
    ("Larger backbone + augmentation",      "M4_yolov8s_augmented_s{s}"),
    ("Tiled inference alone",               "M4_yolov8n_baseline_s{s}_tiled2"),
    ("P2 + augmentation",                   "M4_yolov8n-p2_augmented_s{s}"),
    ("Augmentation + tiled inference",      "M4_yolov8n_augmented_s{s}_tiled2"),
    ("P2 + augmentation + tiled inference", "M4_yolov8n-p2_augmented_s{s}_tiled2"),
]
BASE = "M_v8n_baseline_s{s}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="also write the markdown table here")
    a = ap.parse_args()

    bands = load_bands(BANDS)
    status = load_run_status()
    rows, lines = [], []
    say = lines.append
    say("# Tier 4 against the Tier 1 baseline - recall below 12 px")
    say("")
    say("Seed-paired against `M_v8n_baseline_s{42,123,456}`; bias-corrected bootstrap 95% "
        f"interval over the seed pairs; floor **+{FLOOR}**. A configuration with fewer than "
        "three completed training runs (see `results/protocol/runs_t4/RUN_STATUS.csv`) is "
        "**underpowered** and carries no interval and no verdict.")
    say("")
    base_vals = {s: bands[BASE.format(s=s)][PRIMARY][2] for s in SEEDS}
    say(f"Baseline (Study M): {st.mean(base_vals.values()):.4f} "
        + " / ".join(f"s{s} {base_vals[s]:.4f}" for s in SEEDS))
    say("")
    say("| Configuration | Seeds scored | Seeds completed | Recall < 12 px | Paired difference (95% CI) | Verdict | Per-seed pairs |")
    say("|---|---:|---:|---:|---|---|---|")
    for label, pat in ARMS:
        pairs, vals, done = [], [], 0
        for s in SEEDS:
            tag = pat.format(s=s)
            if tag not in bands:
                continue
            v = bands[tag][PRIMARY][2]
            vals.append(v); pairs.append((s, v - base_vals[s]))
            done += run_complete(tag, status)
        d = st.mean(p for _, p in pairs)
        pair_txt = ", ".join(f"s{s} {p:+.4f}" for s, p in pairs)
        if done < 3:
            ci_txt = f"{d:+.4f} (no interval)"
            verdict = f"underpowered - {done} of 3 runs completed"
            lo = hi = None
        else:
            lo, hi = bc_ci([p for _, p in pairs])
            ci_txt = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            if lo > 0 and d > FLOOR:
                verdict = "clears the floor"
            elif hi < 0:
                verdict = "worse than baseline"
            elif lo > 0:
                verdict = f"detectable, below the +{FLOOR} floor"
            else:
                verdict = "null, CI includes zero"
        say(f"| {label} | {len(pairs)} | {done} | {st.mean(vals):.4f} | {ci_txt} | {verdict} | {pair_txt} |")
        rows.append(dict(arm=label, seeds=len(pairs), completed=done, recall=f"{st.mean(vals):.4f}",
                         diff=f"{d:.4f}", ci_lo="" if lo is None else f"{lo:.4f}",
                         ci_hi="" if hi is None else f"{hi:.4f}", verdict=verdict))
    os.makedirs(RES, exist_ok=True)
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()), lineterminator="\n")
        w.writeheader(); w.writerows(rows)
    if a.out:
        with open(a.out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\n  wrote {os.path.relpath(CSV_OUT, ROOT)}")


if __name__ == "__main__":
    main()
