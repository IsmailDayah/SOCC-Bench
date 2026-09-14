"""Merge per-run band ledgers into one tier ledger, with one scoring per run.

    python code/merge_runs.py                                   # Tier 2, report only
    python code/merge_runs.py --apply                           # Tier 2, write
    python code/merge_runs.py --runs-dir results/protocol/runs_t3 --require 5 \
        --out results/protocol/results/tier3/BANDS_tier3.csv \
        --out-results results/protocol/results/tier3/RESULTS_tier3.csv --apply

Each run folder holds the BANDS.csv and RESULTS.csv its scoring pass wrote.

One Tier 2 run, `yolov8n-cbam_s123`, was first scored without the occlusion
partition, so its own BANDS.csv has size and class rows but no occ: rows. Its
ledger is therefore taken from a complete re-score of the same checkpoint
(results/protocol/results/tier2/rescore_occ/), made on CPU where every other run
was scored on GPU. The two scorings of that checkpoint differ by one detection in
two slices (size:12-20 703 vs 702, class:truck_peterbilt 535 vs 534), 0.0008 of
recall; across the three Tier 1 baseline seeds, CPU and GPU scoring differ by at
most 0.0036 in any slice, against a floor of 0.02.

Exactly one scoring file is used per run: concatenating both would count that run's
size rows twice. A declared override whose file is missing, a duplicate
(tag, slice) or a run without occlusion rows is a hard error.
"""
import argparse
import csv
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "results", "protocol", "runs_t2")
RESULTS = os.path.join(ROOT, "results", "protocol", "results")

# run -> replacement scoring file. Anything not listed uses its own BANDS.csv.
OVERRIDE = {
    "yolov8n-cbam_s123": os.path.join(
        RESULTS, "tier2", "rescore_occ", "BANDS_cbam_s123_reoccl.csv"),
}
BANDS_COLS = ["tag", "slice", "n", "found", "recall", "recall_any_class"]


def read_rows(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--runs-dir", default=RUNS,
                    help="folder of run folders to merge (default: runs_t2)")
    ap.add_argument("--out", default=os.path.join(RESULTS, "tier2", "BANDS_tier2.csv"))
    ap.add_argument("--out-results", default=os.path.join(RESULTS, "tier2",
                                                          "RESULTS_tier2.csv"))
    ap.add_argument("--require", type=int, default=9,
                    help="expected number of runs; merging fewer is reported")
    a = ap.parse_args()

    # The output directory must name the same tier as the runs directory, so a
    # Tier 3 merge can never overwrite the Tier 2 ledger through a default path.
    runs_name = os.path.basename(os.path.abspath(a.runs_dir).rstrip("\\/"))
    want = "tier" + runs_name.rsplit("_t", 1)[-1] if "_t" in runs_name else None
    for path, kind in ((a.out, "--out"), (a.out_results, "--out-results")):
        tier_dir = os.path.basename(os.path.dirname(os.path.abspath(path)))
        if want and tier_dir != want:
            sys.exit(f"REFUSING TO MERGE: {runs_name} would be written to "
                     f"{tier_dir}/ via {kind}.\nPass {kind} under {want}/.")

    runs_root = os.path.abspath(a.runs_dir)
    print(f"  runs: {os.path.relpath(runs_root, ROOT)}")
    runs = sorted(d for d in os.listdir(runs_root)
                  if os.path.isdir(os.path.join(runs_root, d)))
    bands, results, seen, problems = [], [], {}, []
    merged = 0

    for r in runs:
        m = os.path.join(runs_root, r)
        ov = OVERRIDE.get(r) if runs_name.endswith("_t2") else None
        use_ov = bool(ov) and os.path.exists(ov)
        if ov and not use_ov:
            problems.append(f"{r}: override file declared but missing: {ov}")
        src = ov if use_ov else os.path.join(m, "BANDS.csv")
        rows = read_rows(src)
        if rows is None:
            print(f"  {r:<24} no BANDS.csv - skipped")
            continue
        merged += 1
        tag_src = "re-score (CPU)" if use_ov else "run ledger (GPU)"

        occ = [x for x in rows if x["slice"].startswith("occ:")]
        if not occ:
            problems.append(f"{r}: bands file has NO occ: rows ({src})")

        for x in rows:
            k = (x["tag"], x["slice"])
            if k in seen:
                problems.append(f"DUPLICATE {k} from {r} and {seen[k]}")
                continue
            seen[k] = r
            bands.append({c: x.get(c, "") for c in BANDS_COLS})

        rr = read_rows(os.path.join(m, "RESULTS.csv"))
        if rr is None:
            problems.append(f"{r}: no RESULTS.csv")
        else:
            results.extend(rr)
        print(f"  {r:<24} {len(rows):>3} band rows, {len(occ)} occ  [{tag_src}]")

    print(f"\n  {merged}/{a.require} runs merged, "
          f"{len(bands)} band rows, {len(results)} result rows")
    if merged < a.require:
        print(f"  *** ONLY {merged} OF {a.require} RUNS - this is a PARTIAL merge")
    for p in problems:
        print(f"  *** {p}")
    if problems:
        print("  refusing to write a merge with unresolved problems")
        sys.exit(1)

    if not a.apply:
        print("\n  (report only - re-run with --apply to write)")
        return

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=BANDS_COLS)
        w.writeheader()
        w.writerows(bands)
    print(f"  wrote {os.path.relpath(a.out, ROOT)}")

    if results:
        with open(a.out_results, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print(f"  wrote {os.path.relpath(a.out_results, ROOT)}")


if __name__ == "__main__":
    main()
