"""Plot the feedback revision, tuned fixed lengths and controlled edits."""

import json
import sys
from pathlib import Path

from analyze_agent_study import METHOD_COLORS


def overall(path):
    return {
        row["mode"]: row
        for row in json.loads(path.read_text())
        if row["dataset"] == "all" and row["subset"] == "all"
    }


def main():
    benches = Path(__file__).resolve().parent
    root = benches / "agent_feedback_study_20261021"
    selected = json.loads(
        (root / "fixed_length_ablation/selected_fixed.json").read_text()
    )["mode"]
    controlled = benches / "controlled_edit_study_20261021"
    outcomes = json.loads((controlled / "baseline/outcomes.json").read_text())
    panels = (
        (
            overall(root / "replay/summary.json"),
            ["baseline", "warm_fixed", "warm_adaptive", "warm_feedback"],
            "A  Fresh public tasks\n20 tasks, fixed-input replay",
        ),
        (
            overall(root / "fixed_length_ablation/evaluation/summary.json"),
            ["fixed_k2", "fixed_k4", "fixed_k8", "fixed_k16", "warm_feedback"],
            f"B  Fixed-length ablation\nValidation selected k={selected.removeprefix('fixed_k')}",
        ),
        (
            overall(controlled / "replay/summary.json"),
            [
                "baseline",
                "warm_fixed",
                "warm_adaptive",
                "warm_feedback",
                "warm_acceptance",
            ],
            f"C  Ten authored edits\n{sum(row['passed'] for row in outcomes)}/10 passed both contracts",
        ),
    )
    skill = Path.home() / ".codex/skills/scientific-visualization"
    sys.path[:0] = [str(skill / "scripts"), str(skill / "assets")]
    from style_presets import apply_publication_style
    from figure_export import save_publication_figure
    from color_palettes import OKABE_ITO
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
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), sharey=True, layout="constrained")
    names = {
        "baseline": "Ordinary",
        "warm_fixed": "History\nfixed",
        "warm_adaptive": "Original\ncontrol",
        "warm_feedback": "Control\n+ feedback",
        "warm_acceptance": "Acceptance\nrule",
        **{f"fixed_k{k}": f"k={k}" for k in (2, 4, 8, 16)},
    }
    fixed_colors = [
        OKABE_ITO[name]
        for name in ("sky_blue", "reddish_purple", "blue", "bluish_green")
    ]
    colors = dict(
        METHOD_COLORS, **dict(zip([f"fixed_k{k}" for k in (2, 4, 8, 16)], fixed_colors))
    )
    maximum = max(
        point["ci95_high"]
        for points, modes, _ in panels
        for mode, point in points.items()
        if mode in modes
    )
    invalid = False
    for ax, (points, modes, title) in zip(axes, panels):
        for index, mode in enumerate(modes):
            row = points[mode]
            preserved = row["sequence_preservation"]
            invalid |= not preserved
            ax.bar(
                index,
                row["speedup"],
                color=colors[mode],
                width=0.65,
                hatch="" if preserved else "///",
            )
            ax.errorbar(
                index,
                row["speedup"],
                yerr=[
                    [max(0, row["speedup"] - row["ci95_low"])],
                    [max(0, row["ci95_high"] - row["speedup"])],
                ],
                fmt="none",
                color="black",
                capsize=3,
                linewidth=1,
            )
            ax.text(
                index,
                row["ci95_high"] + 0.03 * maximum,
                f"{row['speedup']:.2f}x",
                ha="center",
                fontsize=8,
            )
        ax.axhline(1, color="black", linewidth=0.8, linestyle="--")
        ax.set_xticks(range(len(modes)), [names[mode] for mode in modes])
        ax.set_ylim(0, maximum * 1.2)
        ax.set_title(title)
    axes[0].set_ylabel("Model wall-time ratio: ordinary / method (x)")
    heading = "Three repetitions; 95% task-bootstrap intervals; each panel uses its own paired baseline"
    if invalid:
        heading += "\nHatched bars change output; their ratios are not equivalent-output speedups"
    fig.suptitle(heading, fontsize=10)
    save_publication_figure(
        fig,
        root / "figures/feedback_and_fixed_lengths",
        formats=["png", "pdf", "svg"],
        dpi=600,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
