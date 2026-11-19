"""Aggregate paired task timings and export figures with scientific-visualization."""

from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path


LABELS = {
    "baseline": "Ordinary KV",
    "small": "Small-model draft",
    "global": "Prompt history",
    "structure": "Code history",
    "adaptive": "Cost controller",
    "acceptance": "Acceptance rule",
    "native": "mlx-lm reference",
    "warm_fixed": "Repair history",
    "warm_adaptive": "Repair cost control",
    "warm_feedback": "Cost control + feedback",
    "warm_acceptance": "Repair acceptance rule",
}

METHOD_COLORS = {
    "baseline": "#888888",
    "small": "#E69F00",
    "global": "#CC79A7",
    "structure": "#56B4E9",
    "adaptive": "#0072B2",
    "acceptance": "#E69F00",
    "native": "#444444",
    "warm_fixed": "#009E73",
    "warm_adaptive": "#0072B2",
    "warm_acceptance": "#E69F00",
    "warm_feedback": "#D55E00",
}


def bootstrap_ratio(base, treatment, seed=42):
    rng = random.Random(seed)
    count = len(base)
    draws = []
    for _ in range(2000):
        indices = [rng.randrange(count) for _ in range(count)]
        draws.append(sum(base[i] for i in indices) / sum(treatment[i] for i in indices))
    draws.sort()
    return draws[49], draws[1949]


def summarize(rows, outcomes=(), reference_mode="baseline"):
    groups = defaultdict(list)
    outcome_by_task = {r["task_id"]: r["passed"] for r in outcomes}
    for row in rows:
        subsets = ["all", row["phase"]]
        if row["task_id"] in outcome_by_task:
            subsets.append("passed" if outcome_by_task[row["task_id"]] else "failed")
        for subset in subsets:
            groups[(row["dataset"], row["mode"], subset)].append(row)
            groups[("all", row["mode"], subset)].append(row)
    baseline = defaultdict(list)
    reference_exact = defaultdict(list)
    for row in rows:
        if row["mode"] == reference_mode:
            baseline[(row["task_id"], row["attempt"])].append(row["wall_s"])
            reference_exact[(row["task_id"], row["attempt"])].append(row["exact"])
    summaries = []
    for (dataset, mode, subset), items in sorted(groups.items()):
        by_request = defaultdict(list)
        for row in items:
            by_request[(row["task_id"], row["attempt"])].append(row["wall_s"])
        task_base, task_treatment = defaultdict(float), defaultdict(float)
        for key, values in by_request.items():
            if key not in baseline:
                raise ValueError(
                    f"Every request needs a timed {reference_mode} reference in the same replay"
                )
            task_base[key[0]] += statistics.mean(baseline[key])
            task_treatment[key[0]] += statistics.mean(values)
        ids = sorted(task_base)
        b, t = [task_base[key] for key in ids], [task_treatment[key] for key in ids]
        low, high = bootstrap_ratio(b, t)
        proposed = sum(r["proposed"] for r in items)
        preserved = all(row["exact"] for row in items) and all(
            all(reference_exact[key]) for key in by_request
        )
        summary = {
            "dataset": dataset,
            "mode": mode,
            "reference_mode": reference_mode,
            "subset": subset,
            "tasks": len(ids),
            "requests": len(by_request),
            "timed_runs": len(items),
            "exact_runs": sum(r["exact"] for r in items),
            "speedup": sum(b) / sum(t),
            "sequence_preservation": preserved,
            "equivalent_output_speedup": sum(b) / sum(t) if preserved else None,
            "ci95_low": low,
            "ci95_high": high,
            "baseline_model_s": sum(b),
            "method_model_s": sum(t),
            "median_task_speedup": statistics.median(x / y for x, y in zip(b, t)),
            "task_wins": sum(y < x for x, y in zip(b, t)),
            "acceptance": sum(r["accepted"] for r in items) / proposed
            if proposed
            else None,
            "proposed": proposed,
            "accepted": sum(r["accepted"] for r in items),
            "speculative_steps": sum(r["speculative_steps"] for r in items),
            "ordinary_steps": sum(r["ordinary_steps"] for r in items),
            "target_calls": sum(r.get("target_calls", 0) for r in items),
            "recovery_steps": sum(r.get("recovery_steps", 0) for r in items),
            "fallback_feedback_updates": sum(
                r.get("fallback_feedback_updates", 0) for r in items
            ),
            "fallback_feedback_matches": sum(
                r.get("fallback_feedback_matches", 0) for r in items
            ),
            "output_tokens": sum(len(r.get("tokens", [])) for r in items),
        }
        for field in (
            "wall_s",
            "decode_s",
            "prefill_s",
            "lookup_s",
            "decision_s",
            "draft_s",
            "verification_s",
            "cache_s",
        ):
            summary[field] = sum(r[field] for r in items)
        summary["task_speedups"] = [x / y for x, y in zip(b, t)]
        summaries.append(summary)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("benches/agent_study_20261021_main")
    )
    parser.add_argument("--label", default="replay")
    args = parser.parse_args()
    metadata = json.loads((args.out / args.label / "metadata.json").read_text())
    rows = [
        json.loads(line)
        for line in (args.out / args.label / "replay.jsonl").read_text().splitlines()
    ]
    counts = defaultdict(int)
    requests = {(row["task_id"], row["attempt"]) for row in rows}
    for row in rows:
        counts[(row["task_id"], row["attempt"], row["mode"])] += 1
    if any(
        counts[(task, attempt, mode)] != metadata["reps"]
        for task, attempt in requests
        for mode in metadata["modes"]
    ):
        raise ValueError(
            "Incomplete paired repetitions; do not summarize as a finished benchmark"
        )
    outcomes = json.loads((args.out / "baseline/outcomes.json").read_text())
    summaries = summarize(rows, outcomes)
    write_path = args.out / args.label
    originals = {
        (row["task_id"], row["attempt"]): row["result"]["tokens"]
        for row in (
            json.loads(line)
            for line in (args.out / "baseline/turns.jsonl").read_text().splitlines()
        )
    }
    failures = []
    for row in rows:
        if row["exact"]:
            continue
        expected = originals[(row["task_id"], row["attempt"])]
        actual = row["tokens"]
        first = next(
            (i for i, pair in enumerate(zip(expected, actual)) if pair[0] != pair[1]),
            min(len(expected), len(actual)),
        )
        failures.append(
            {key: row[key] for key in ("task_id", "attempt", "phase", "mode", "rep")}
            | {
                "first_difference": first,
                "expected_length": len(expected),
                "actual_length": len(actual),
            }
        )
    (write_path / "correctness_failures.json").write_text(
        json.dumps(failures, indent=2) + "\n"
    )
    (write_path / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    fixed_reference = (
        "warm_fixed"
        if any(row["mode"] == "warm_fixed" for row in rows)
        else "structure"
    )
    if any(row["mode"] == fixed_reference for row in rows):
        increments = [
            row
            for row in summarize(rows, outcomes, reference_mode=fixed_reference)
            if row["mode"]
            in {
                "adaptive",
                "acceptance",
                "warm_adaptive",
                "warm_acceptance",
                "warm_feedback",
            }
        ]
        (write_path / "controller_comparisons.json").write_text(
            json.dumps(increments, indent=2) + "\n"
        )
    fields = [key for key in summaries[0] if key != "task_speedups"]
    with (write_path / "summary.csv").open("w", newline="") as output:
        writer = csv.DictWriter(output, fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in summaries)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    skill = Path.home() / ".codex/skills/scientific-visualization"
    sys.path.insert(0, str(skill / "scripts"))
    sys.path.insert(0, str(skill / "assets"))
    from style_presets import apply_publication_style
    from figure_export import save_publication_figure
    from color_palettes import OKABE_ITO_LIST

    apply_publication_style("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.titlesize": 11,
        }
    )
    figures = write_path / "figures"
    figures.mkdir(exist_ok=True)
    order = [mode for mode in LABELS if any(r["mode"] == mode for r in rows)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained", sharey=True)
    for ax, dataset in zip(axes, ("mbpp", "humaneval")):
        points = {
            r["mode"]: r
            for r in summaries
            if r["dataset"] == dataset and r["subset"] == "all"
        }
        if not points:
            ax.set_visible(False)
            continue
        present = [mode for mode in order if mode in points]
        for i, mode in enumerate(present):
            row = points[mode]
            valid = row["exact_runs"] == row["timed_runs"]
            ax.bar(
                i,
                row["speedup"],
                color=METHOD_COLORS[mode],
                hatch="" if valid else "///",
                alpha=0.8,
            )
            ax.errorbar(
                i,
                row["speedup"],
                yerr=[
                    [max(0, row["speedup"] - row["ci95_low"])],
                    [max(0, row["ci95_high"] - row["speedup"])],
                ],
                fmt="none",
                color="black",
                capsize=3,
            )
        ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(
            range(len(present)), [LABELS[m] for m in present], rotation=35, ha="right"
        )
        ax.set_ylabel("Model wall-time speedup (x)")
        ax.set_title(
            dataset.upper() + f" (n={next(iter(points.values()))['tasks']} tasks)"
        )
    heading = (
        f"Paired replay, {metadata['reps']} repetition(s); 95% task-bootstrap intervals"
    )
    if any(not row["sequence_preservation"] for row in summaries):
        heading += "\nHatched bars change outputs and do not establish equivalent-output speedup"
    fig.suptitle(heading, fontsize=11)
    save_publication_figure(
        fig, figures / "paired_speedup", formats=["png", "pdf", "svg"], dpi=600
    )
    plt.close(fig)

    all_rows = {
        r["mode"]: r
        for r in summaries
        if r["dataset"] == "all" and r["subset"] == "all"
    }
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
    fields = [
        ("prefill_s", "Prefill"),
        ("verification_s", "Target forwards"),
        ("draft_s", "Draft model"),
        ("lookup_s", "History"),
        ("decision_s", "Controller"),
        ("cache_s", "Cache alignment"),
    ]
    bottom = [0.0] * len(order)
    for i, (field, label) in enumerate(fields):
        heights = [100 * all_rows[m][field] / all_rows[m]["wall_s"] for m in order]
        if not any(heights):
            continue
        axes[0].bar(
            range(len(order)),
            heights,
            bottom=bottom,
            label=label,
            color=OKABE_ITO_LIST[i],
        )
        bottom = [a + b for a, b in zip(bottom, heights)]
    axes[0].bar(
        range(len(order)),
        [max(0, 100 - value) for value in bottom],
        bottom=bottom,
        label="Other Python",
        color=OKABE_ITO_LIST[6],
    )
    axes[0].set_xticks(
        range(len(order)), [LABELS[m] for m in order], rotation=35, ha="right"
    )
    axes[0].set_ylabel("Share of measured model wall time (%)")
    axes[0].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=3)
    axes[0].set_title("Measured execution breakdown")
    for i, mode in enumerate(order):
        values = sorted(all_rows[mode]["task_speedups"])
        axes[1].plot(
            values,
            [(j + 1) / len(values) for j in range(len(values))],
            label=LABELS[mode],
            color=METHOD_COLORS[mode],
        )
    axes[1].axvline(1, color="black", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Per-task model speedup (x)")
    axes[1].set_ylabel("Fraction of tasks")
    axes[1].set_title("Task-to-task variation")
    axes[1].legend(fontsize=8)
    if any(not row["sequence_preservation"] for row in all_rows.values()):
        fig.suptitle(
            "Some methods change outputs. Their timing ratios are descriptive only",
            fontsize=10,
        )
    save_publication_figure(
        fig, figures / "cost_and_distribution", formats=["png", "pdf", "svg"], dpi=600
    )
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.1), layout="constrained")
    for ax, subsets, title in zip(
        axes,
        (("generation", "repair"), ("passed", "failed")),
        ("Generation and repair", "Final task outcome"),
    ):
        width = 0.36
        for j, subset in enumerate(subsets):
            points = {
                r["mode"]: r
                for r in summaries
                if r["dataset"] == "all" and r["subset"] == subset
            }
            x = [i + (j - 0.5) * width for i in range(len(order))]
            y = [points[m]["speedup"] for m in order]
            bars = ax.bar(
                x, y, width, label=subset.capitalize(), color=OKABE_ITO_LIST[j * 2]
            )
            ax.errorbar(
                x,
                y,
                yerr=[
                    [
                        max(0, points[m]["speedup"] - points[m]["ci95_low"])
                        for m in order
                    ],
                    [
                        max(0, points[m]["ci95_high"] - points[m]["speedup"])
                        for m in order
                    ],
                ],
                fmt="none",
                color="black",
                capsize=2,
                linewidth=0.8,
            )
            for bar, mode in zip(bars, order):
                if not points[mode]["sequence_preservation"]:
                    bar.set_hatch("///")
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(
            range(len(order)), [LABELS[m] for m in order], rotation=35, ha="right"
        )
        ax.set_ylabel("Model wall-time speedup (x)")
        ax.set_title(title)
        ax.legend(fontsize=9)
    if any(not row["sequence_preservation"] for row in all_rows.values()):
        fig.suptitle(
            "Hatched bars change outputs and do not establish equivalent-output speedup",
            fontsize=10,
        )
    save_publication_figure(
        fig, figures / "phase_and_outcome", formats=["png", "pdf", "svg"], dpi=600
    )
    plt.close(fig)
    for row in summaries:
        if row["subset"] != "all":
            continue
        print(
            {
                key: row[key]
                for key in (
                    "dataset",
                    "mode",
                    "tasks",
                    "speedup",
                    "ci95_low",
                    "ci95_high",
                    "exact_runs",
                    "timed_runs",
                )
            }
        )


if __name__ == "__main__":
    main()
