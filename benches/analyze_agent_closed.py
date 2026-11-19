"""Summarize completed, paired real-agent episodes separately from replay."""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from analyze_agent_study import bootstrap_ratio


def summarize_closed(outcomes, metadata):
    expected = {
        (task, mode, rep)
        for task in metadata["task_ids"]
        for mode in metadata["modes"]
        for rep in range(metadata["reps"])
    }
    observed = [(row["task_id"], row["mode"], row["rep"]) for row in outcomes]
    if len(observed) != len(expected) or set(observed) != expected:
        raise ValueError("Incomplete or duplicate closed-loop episodes")
    baseline = {
        row["task_id"]: row["passed"] for row in outcomes if row["mode"] == "baseline"
    }
    result = []
    for dataset in ("all", "mbpp", "humaneval"):
        subset = [
            row for row in outcomes if dataset == "all" or row["dataset"] == dataset
        ]
        if not subset:
            continue
        for mode in metadata["modes"]:
            items = [row for row in subset if row["mode"] == mode]
            entry = {
                "dataset": dataset,
                "mode": mode,
                "episodes": len(items),
                "tasks": len({row["task_id"] for row in items}),
                "exact_episodes": sum(row["exact"] for row in items),
                "same_outcome_episodes": sum(
                    row["passed"] == baseline[row["task_id"]] for row in items
                ),
                "passed_episodes": sum(row["passed"] for row in items),
                "calls": sum(row["calls"] for row in items),
                "sequence_preservation": all(row["exact"] for row in items)
                and all(row["exact"] for row in subset if row["mode"] == "baseline"),
            }
            for field in ("agent_s", "model_s"):
                grouped = defaultdict(list)
                for row in subset:
                    if row["mode"] in {"baseline", mode}:
                        grouped[(row["task_id"], row["mode"])].append(row[field])
                ids = sorted({row["task_id"] for row in items})
                base = [statistics.mean(grouped[(task, "baseline")]) for task in ids]
                treatment = [statistics.mean(grouped[(task, mode)]) for task in ids]
                low, high = bootstrap_ratio(base, treatment)
                entry[field] = {
                    "speedup": sum(base) / sum(treatment),
                    "ci95_low": low,
                    "ci95_high": high,
                    "baseline_total_s": sum(base),
                    "method_total_s": sum(treatment),
                    "equivalent_output_speedup": sum(base) / sum(treatment)
                    if entry["sequence_preservation"]
                    else None,
                }
            entry["tool_and_framework_s"] = sum(
                row["agent_s"] - row["model_s"] for row in items
            )
            result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("benches/agent_study_20261021_main")
    )
    parser.add_argument("--label", default="closed_guarded")
    args = parser.parse_args()
    directory = args.out / args.label
    outcomes = json.loads((directory / "outcomes.json").read_text())
    metadata = json.loads((directory / "metadata.json").read_text())
    result = summarize_closed(outcomes, metadata)
    (directory / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    for row in result:
        if row["dataset"] == "all":
            print(row)


if __name__ == "__main__":
    main()
