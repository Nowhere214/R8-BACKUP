"""Reanalyze saved Stage 3 timing; no model inference is performed.

Run with: pdm run python benches/analyze_stage3_break_even.py
"""

import argparse
import csv
import math
from pathlib import Path


DEFAULT_INPUT = Path(__file__).with_name("stage3_speculative_length_scan_summary.csv")


def analyze(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as source:
        rows = list(csv.DictReader(source))
    baselines = {}
    for row in rows:
        if row["mode"] == "baseline":
            key = (row["target_model"], int(row["output_len"]))
            if key in baselines:
                raise ValueError(f"Duplicate baseline: {key}")
            baselines[key] = row

    results = []
    for row in rows:
        if row["mode"] != "speculative":
            continue
        key = (row["target_model"], int(row["output_len"]))
        baseline = baselines[key]
        if row["matches_baseline_all"] != "1":
            raise ValueError(f"Output mismatch: {key}, k={row['draft_length']}")
        if int(row["decode_tokens"]) != int(baseline["decode_tokens"]):
            raise ValueError("Timing comparisons require equal decode token counts")
        base_time = float(baseline["decode_time_s_mean"])
        spec_time = float(row["decode_time_s_mean"])
        draft_time = float(row["draft_time_s_mean"])
        if not all(math.isfinite(t) and t > 0 for t in (base_time, spec_time, draft_time)):
            raise ValueError("Expected positive finite timing measurements")
        if draft_time > spec_time:
            raise ValueError("Draft time exceeds total decode time")
        # Hold acceptance, verification and the execution schedule fixed.
        saving_needed = max(0.0, spec_time - base_time)
        reduction = saving_needed / draft_time
        results.append({
            "target": key[0],
            "output": key[1],
            "k": int(row["draft_length"]),
            "ratio": base_time / spec_time,
            "draft_reduction": reduction,
            "draft_only_feasible": reduction <= 1.0,
            "base_time": base_time,
            "spec_time": spec_time,
        })
    if not results:
        raise ValueError("No speculative measurements found")
    return sorted(results, key=lambda item: (item["target"], item["output"], item["k"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    args = parser.parse_args()
    results = analyze(args.input)
    print("Saved-data analysis, not a new inference benchmark.")
    print("Throughput ratio = baseline mean decode time / speculative mean decode time.")
    print("| Output | k | Decode ratio | Draft time reduction to break even |")
    print("| ---: | ---: | ---: | ---: |")
    for item in results:
        reduction = f"{item['draft_reduction']:.1%}" if item["draft_only_feasible"] else "not feasible"
        print(f"| {item['output']} | {item['k']} | {item['ratio']:.3f}x | {reduction} |")
    print("\nBest fixed action for each measured workload, with baseline available:")
    for target, output in sorted({(item["target"], item["output"]) for item in results}):
        candidates = [item for item in results if (item["target"], item["output"]) == (target, output)]
        best = max(candidates, key=lambda item: item["ratio"])
        action = f"k={best['k']}" if best["ratio"] > 1 else "baseline"
        print(f"{target}, output={output}: {action}, ratio={max(1.0, best['ratio']):.3f}x")
    print("This is a workload-level comparison, not an upper bound on per-step switching.")
    print("Draft savings assume unchanged acceptance and verification; they do not predict a new drafter's speedup.")
    print("Calibration, exploration, prefill and cache realignment costs are not modeled.")


if __name__ == "__main__":
    main()
