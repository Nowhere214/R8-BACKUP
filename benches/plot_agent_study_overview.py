"""Show replay, real-agent and native-model timing results on separate panels."""

import json
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1] / "benches/agent_study_20261021_holdout"
    skill = Path.home() / ".codex/skills/scientific-visualization"
    sys.path[:0] = [str(skill / "scripts"), str(skill / "assets")]
    from style_presets import apply_publication_style
    from figure_export import save_publication_figure
    from color_palettes import OKABE_ITO
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def replay_points(path):
        rows = json.loads(path.read_text())
        return {
            row["mode"]: row
            for row in rows
            if row["dataset"] == "all" and row["subset"] == "all"
        }

    replay = replay_points(root / "replay/summary.json")
    closed = {
        row["mode"]: dict(
            row["agent_s"], sequence_preservation=row["sequence_preservation"]
        )
        for row in json.loads((root / "closed/summary.json").read_text())
        if row["dataset"] == "all"
    }
    native_path = root / "native_extension/replay/summary.json"
    native = replay_points(native_path) if native_path.exists() else {}
    apply_publication_style("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
        }
    )
    fig, axes = plt.subplots(
        1, 3, figsize=(12.5, 4.3), layout="constrained", sharey=True
    )
    maximum = max(
        [1.2]
        + [
            point["ci95_high"]
            for group in (replay, closed, native)
            for point in group.values()
        ]
    )
    methods = ("baseline", "warm_fixed", "warm_adaptive", "warm_acceptance")
    names = {
        "baseline": "Ordinary",
        "warm_fixed": "History\nfixed",
        "warm_adaptive": "Cost\ncontrol",
        "warm_acceptance": "Acceptance\nrule",
    }
    colors = {
        "baseline": "#707070",
        "warm_fixed": OKABE_ITO["bluish_green"],
        "warm_adaptive": OKABE_ITO["blue"],
        "warm_acceptance": OKABE_ITO["orange"],
    }
    for ax, points, title in zip(
        axes,
        (replay, closed, native),
        (
            "A  SiliconLLM replay\n30 tasks, model wall time",
            "B  Actual agent\n20 tasks, agent wall time",
            "C  Native model and cache\n20 tasks, model wall time",
        ),
    ):
        if not points:
            ax.text(
                0.5,
                0.5,
                "No validated result",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            ax.set_title(title)
            continue
        present = [mode for mode in methods if mode in points]
        for i, mode in enumerate(present):
            point = points[mode]
            if not point["sequence_preservation"]:
                ax.text(i, 0.1, "Output\nmismatch", ha="center", fontsize=8)
                continue
            value, low, high = point["speedup"], point["ci95_low"], point["ci95_high"]
            ax.bar(i, value, color=colors[mode], width=0.65)
            ax.errorbar(
                i,
                value,
                yerr=[[max(0, value - low)], [max(0, high - value)]],
                fmt="none",
                color="black",
                capsize=3,
                linewidth=1,
            )
            ax.text(i, high + 0.06, f"{value:.2f}x", ha="center", fontsize=9)
        ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
        ax.set_xticks(range(len(present)), [names[mode] for mode in present])
        ax.set_ylim(0, maximum * 1.15)
        ax.set_ylabel("Speedup over the matching ordinary baseline (x)")
        ax.set_title(title)
    fig.suptitle(
        "Repair-only speculation; three repetitions; 95% task-bootstrap intervals",
        fontsize=11,
    )
    save_publication_figure(
        fig, root / "figures/study_overview", formats=["png", "pdf", "svg"], dpi=600
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
