"""
Plot workload length scan results for multiple model sizes.

Reads:
    benches/workload_length_scan_summary.csv
    benches/workload_length_scan_1p5b_summary.csv

Writes:
    benches/workload_length_scan_combined.png
    benches/workload_length_scan_combined.pdf
    benches/workload_length_scan_combined.svg
    benches/workload_length_scan_combined.tiff
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parent.parent
INPUTS = [
    ROOT / "benches" / "workload_length_scan_summary.csv",
    ROOT / "benches" / "workload_length_scan_1p5b_summary.csv",
]
OUT = ROOT / "benches" / "workload_length_scan_combined"
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from color_palettes import OKABE_ITO  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from style_presets import get_base_style  # noqa: E402

MODEL_LABEL = {
    "qwen2-0.5b": "Qwen2 0.5B",
    "qwen2-1.5b": "Qwen2 1.5B",
}

MODEL_STYLE = {
    "qwen2-0.5b": {"marker": "o", "linestyle": "-"},
    "qwen2-1.5b": {"marker": "s", "linestyle": "--"},
}

PANEL_COLOR = {
    "gap": OKABE_ITO["bluish_green"],
    "ref": OKABE_ITO["vermillion"],
}


def load_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in INPUTS:
        with path.open() as f:
            rows.extend(csv.DictReader(f))
    return rows


def grouped_by_model(rows: list[dict[str, str]]) -> dict[str, list[tuple[int, float, float, float]]]:
    by_prompt: dict[tuple[str, int], dict[str, float]] = {}
    for row in rows:
        key = (row["model"], int(row["requested_prompt_len"]))
        by_prompt.setdefault(key, {})[row["engine"]] = float(row["decode_tps_mean"])

    out: dict[str, list[tuple[int, float, float, float]]] = {}
    for (model, prompt_len), values in by_prompt.items():
        if "mlx_lm" not in values or "silicon_w1ref" not in values:
            continue
        gap = values["mlx_lm"] / values["silicon_w1ref"]
        out.setdefault(model, []).append((prompt_len, values["silicon_w1ref"], values["mlx_lm"], gap))

    for model in out:
        out[model].sort(key=lambda x: x[0])
    return out


def apply_style() -> None:
    style = get_base_style()
    style.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.3,
        }
    )
    plt.rcParams.update(style)


def main() -> None:
    apply_style()
    rows = load_rows()
    grouped = grouped_by_model(rows)

    fig, (ax_gap, ax_ref) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.patch.set_facecolor("white")

    for ax in (ax_gap, ax_ref):
        ax.set_facecolor("white")
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 512])
        ax.set_xticklabels(["64", "128", "512"])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Prompt length (tokens)")

    ax_gap.text(-0.12, 1.06, "A", transform=ax_gap.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_ref.text(-0.12, 1.06, "B", transform=ax_ref.transAxes, fontsize=10, fontweight="bold", va="top")

    for model, points in grouped.items():
        xs = [p[0] for p in points]
        ref_decode = [p[1] for p in points]
        gaps = [p[3] for p in points]
        style = MODEL_STYLE[model]
        label = MODEL_LABEL[model]
        ax_gap.plot(
            xs,
            gaps,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=PANEL_COLOR["gap"],
            label=label,
        )
        ax_ref.plot(
            xs,
            ref_decode,
            marker=style["marker"],
            linestyle=style["linestyle"],
            color=PANEL_COLOR["ref"],
            label=label,
        )
    ax_gap.set_title("Decode gap")
    ax_gap.set_ylabel("mlx-lm / Stage 1 ref")
    ax_gap.set_ylim(0, 66)
    ax_gap.legend(frameon=False, loc="upper left")

    ax_ref.set_title("Stage 1 ref decode")
    ax_ref.set_ylabel("Decode throughput (tok/s)")
    ax_ref.set_yscale("log")
    ax_ref.set_ylim(1.2, 28)
    ax_ref.set_yticks([2, 5, 10, 20])
    ax_ref.set_yticklabels(["2", "5", "10", "20"])
    ax_ref.legend(frameon=False, loc="upper right")

    fig.tight_layout(pad=0.8)
    save_publication_figure(
        fig,
        OUT,
        formats=["png", "pdf", "svg", "tiff"],
        dpi=600,
        transparent=False,
        facecolor="white",
    )
    print(f"Wrote {OUT}.png/.pdf/.svg/.tiff")


if __name__ == "__main__":
    main()
