"""Plot actual agent workload and calibration measurements using the figure skill."""

import argparse
import json
import statistics
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
RUN = ROOT / "benches/agent_study_20261021_main"
SKILL = Path.home() / ".codex/skills/scientific-visualization"
sys.path.insert(0, str(SKILL / "scripts"))
sys.path.insert(0, str(SKILL / "assets"))
from style_presets import apply_publication_style  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from color_palettes import OKABE_ITO  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=RUN)
    parser.add_argument("--cost-file", default="calibration.json")
    args = parser.parse_args()
    run = args.out
    apply_publication_style("default")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
        }
    )
    outcomes = json.loads((run / "baseline/outcomes.json").read_text())
    turns = [
        json.loads(line)
        for line in (run / "baseline/turns.jsonl").read_text().splitlines()
    ]
    costs = json.loads((run / args.cost_file).read_text())
    output = run / "figures"
    output.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), layout="constrained")
    for i, dataset in enumerate(("mbpp", "humaneval")):
        selected = [r for r in outcomes if r["dataset"] == dataset]
        passed = sum(r["passed"] for r in selected)
        axes[0].bar(
            i,
            passed,
            color=OKABE_ITO["bluish_green"],
            label="Passed" if i == 0 else None,
        )
        axes[0].bar(
            i,
            len(selected) - passed,
            bottom=passed,
            color=OKABE_ITO["vermillion"],
            label="Failed" if i == 0 else None,
        )
        axes[0].text(
            i,
            len(selected) + 1,
            f"{passed}/{len(selected)} passed",
            ha="center",
            fontsize=10,
        )
    axes[0].set_ylim(
        0,
        max(
            sum(row["dataset"] == dataset for row in outcomes)
            for dataset in ("mbpp", "humaneval")
        )
        * 1.2,
    )
    axes[0].set_xticks([0, 1], ["MBPP", "HumanEval"])
    axes[0].set_ylabel("Tasks")
    axes[0].set_title("A  Final task outcomes")
    axes[0].legend(fontsize=9)
    for i, phase in enumerate(("generation", "repair")):
        selected = [r for r in turns if r["phase"] == phase]
        prefill = sum(r["result"]["prefill_s"] for r in selected)
        decode = sum(r["result"]["decode_s"] for r in selected)
        axes[1].bar(
            i, prefill, color=OKABE_ITO["sky_blue"], label="Prefill" if i == 0 else None
        )
        axes[1].bar(
            i,
            decode,
            bottom=prefill,
            color=OKABE_ITO["blue"],
            label="Decode" if i == 0 else None,
        )
        axes[1].text(
            i, prefill + decode + 2, f"{len(selected)} calls", ha="center", fontsize=10
        )
    axes[1].set_xticks([0, 1], ["Generation", "Repair"])
    axes[1].set_ylabel("Aggregate model time (s)")
    axes[1].set_title("B  Baseline execution cost")
    axes[1].legend(fontsize=9)
    k_values = [0, 2, 4, 8, 16]
    median_ms = [
        1000 * (costs["baseline_s"] if k == 0 else costs["verification_s"][str(k)])
        for k in k_values
    ]
    quartiles = [
        statistics.quantiles(costs["cost_samples_s"][str(k)], n=4) for k in k_values
    ]
    error_low = [m - q[0] * 1000 for m, q in zip(median_ms, quartiles)]
    error_high = [q[2] * 1000 - m for m, q in zip(median_ms, quartiles)]
    axes[2].errorbar(
        range(5),
        median_ms,
        yerr=[error_low, error_high],
        marker="o",
        capsize=3,
        color=OKABE_ITO["bluish_green"],
    )
    axes[2].set_xticks(range(5), ["1", "3", "5", "9", "17"])
    axes[2].set_ylim(bottom=0)
    axes[2].set_xlabel("Tokens in target forward")
    axes[2].set_ylabel("Median forward time (ms)")
    axes[2].set_title("C  Calibration median and IQR\n10 tasks, 50 probes per shape")
    save_publication_figure(
        fig, output / "workload_and_calibration", formats=["png", "pdf", "svg"], dpi=600
    )
    plt.close(fig)
    summary = {
        "tasks": len(outcomes),
        "passed": sum(r["passed"] for r in outcomes),
        "initial_visible_passed": sum(r["initial_visible_pass"] for r in outcomes),
        "calls": len(turns),
        "repairs": sum(r["calls"] - 1 for r in outcomes),
        "duplicate_repairs": sum(r["duplicate_repairs"] for r in outcomes),
        "length_stops": sum(r["length_stops"] for r in outcomes),
    }
    (run / "baseline/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(summary)


if __name__ == "__main__":
    main()
