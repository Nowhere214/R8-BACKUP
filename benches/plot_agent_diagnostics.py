"""Explain controller decisions, history reuse and the cache regression case."""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tiny_llm.agent_pilot import repeat_fraction


def main():
    benches = Path(__file__).resolve().parents[1] / "benches"
    root = benches / "agent_study_20261021_holdout"
    directory = root / "replay"
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = [
        json.loads(line)
        for line in (directory / "replay.jsonl").read_text().splitlines()
    ]
    traces = [
        json.loads(line)
        for line in (root / "baseline/turns.jsonl").read_text().splitlines()
    ]
    if len(rows) != len(traces) * len(metadata["modes"]) * metadata["reps"]:
        raise ValueError("Diagnostics require a completed primary replay")
    audit = json.loads(
        (benches / "agent_study_20261021_main/full_acceptance_audit.json").read_text()
    )
    controller = "warm_adaptive"
    skill = Path.home() / ".codex/skills/scientific-visualization"
    sys.path[:0] = [str(skill / "scripts"), str(skill / "assets")]
    from style_presets import apply_publication_style
    from figure_export import save_publication_figure
    from color_palettes import OKABE_ITO_LIST
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_publication_style("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8), layout="constrained")
    ax = axes[0, 0]
    decision_counts = {}
    for phase, color in zip(
        ("generation", "repair"), (OKABE_ITO_LIST[0], OKABE_ITO_LIST[2])
    ):
        items = [
            row for row in rows if row["mode"] == controller and row["phase"] == phase
        ]
        counts = Counter({0: sum(row["ordinary_steps"] for row in items)})
        counts.update(event["k"] for row in items for event in row["decisions"])
        decision_counts[phase] = dict(counts)
        x = [i + (-0.18 if phase == "generation" else 0.18) for i in range(5)]
        total = sum(counts.values())
        ax.bar(
            x,
            [100 * counts[k] / total for k in (0, 2, 4, 8, 16)],
            0.36,
            label=phase.capitalize(),
            color=color,
        )
    ax.set_xticks(range(5), ["0 (ordinary)", "2", "4", "8", "16"])
    ax.set_xlabel("Chosen draft length")
    ax.set_ylabel("Decoder iterations (%)")
    ax.set_title("A  Cost-controller actions")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    modes = ("warm_fixed", "warm_adaptive", "warm_acceptance")
    labels = ("Repair history", "Repair cost control", "Repair acceptance rule")
    for phase, color in zip(
        ("generation", "repair"), (OKABE_ITO_LIST[0], OKABE_ITO_LIST[2])
    ):
        rates = []
        for mode in modes:
            items = [
                row for row in rows if row["mode"] == mode and row["phase"] == phase
            ]
            proposed = sum(row["proposed"] for row in items)
            rates.append(
                100 * sum(row["accepted"] for row in items) / proposed
                if proposed
                else 0
            )
        x = [i + (-0.18 if phase == "generation" else 0.18) for i in range(len(modes))]
        ax.bar(x, rates, 0.36, label=phase.capitalize(), color=color)
    ax.set_xticks(range(len(modes)), labels, rotation=20, ha="right")
    ax.set_ylabel("Accepted / proposed tokens (%)")
    ax.set_ylim(0, 105)
    ax.set_title("B  Proposal acceptance (no drafts in generation)")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    times = defaultdict(list)
    for row in rows:
        times[(row["task_id"], row["attempt"], row["mode"])].append(row["wall_s"])
    reuse = []
    mismatched = {
        (row["task_id"], row["attempt"])
        for row in rows
        if row["mode"] == controller and not row["exact"]
    }
    for phase, marker, color in (
        ("generation", "o", OKABE_ITO_LIST[0]),
        ("repair", "^", OKABE_ITO_LIST[2]),
    ):
        points = []
        for trace in traces:
            if trace["phase"] != phase:
                continue
            prefix = (trace["task_id"], trace["attempt"])
            coverage = repeat_fraction(trace["result"]["tokens"], trace["history"])
            ratio = statistics.mean(times[(*prefix, "baseline")]) / statistics.mean(
                times[(*prefix, controller)]
            )
            if prefix not in mismatched:
                points.append((100 * coverage, ratio))
            reuse.append(
                {
                    "task_id": trace["task_id"],
                    "attempt": trace["attempt"],
                    "phase": phase,
                    "prior_code_ngram_coverage": coverage,
                    "controller_time_ratio": ratio,
                    "exact": prefix not in mismatched,
                }
            )
        ax.scatter(
            [p[0] for p in points],
            [p[1] for p in points],
            marker=marker,
            color=color,
            alpha=0.65,
            s=28,
            label=phase.capitalize(),
        )
    ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Output covered by prior-code 4-grams (%)")
    ax.set_ylabel("Cost-controller request speedup (x)")
    ax.set_title("C  Reuse and measured request speedup")
    ax.legend(fontsize=9)
    if mismatched:
        ax.text(
            0.03,
            0.97,
            f"Excludes {len(mismatched)} non-equivalent requests",
            transform=ax.transAxes,
            va="top",
            fontsize=8,
        )

    ax = axes[1, 1]
    errors = audit["cache_differences"]
    layers = range(1, len(errors) + 1)
    ax.plot(
        layers,
        [row["key_max_abs"] for row in errors],
        label="Key after full acceptance",
        color=OKABE_ITO_LIST[0],
    )
    ax.plot(
        layers,
        [row["value_max_abs"] for row in errors],
        label="Value after full acceptance",
        color=OKABE_ITO_LIST[2],
    )
    ax.axhline(
        audit["restored_cache_max_abs"],
        color="black",
        linestyle="--",
        linewidth=0.8,
        label="After restore and ordinary replay",
    )
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("Retained KV max. absolute difference")
    ax.set_title("D  Matched tokens, different KV (one request)")
    ax.legend(fontsize=8)
    save_publication_figure(
        fig,
        directory / "figures/controller_and_correctness",
        formats=["png", "pdf", "svg"],
        dpi=600,
    )
    plt.close(fig)
    (directory / "diagnostics.json").write_text(
        json.dumps(
            {
                "controller_iterations": decision_counts,
                "reuse_requests": reuse,
                "recovery_steps": {
                    mode: sum(
                        row["recovery_steps"] for row in rows if row["mode"] == mode
                    )
                    for mode in ("baseline", *modes)
                },
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
