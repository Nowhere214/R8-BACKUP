"""
benches/plot_phase0.py
======================

Reads `benches/biweekly_n3_summary.csv` and produces a 2-panel figure showing
prefill and decode throughput vs model size for both engines, with error bars
from the N=3 stdev.

First-time use:
    pdm add matplotlib
    pdm run python benches/plot_phase0.py

The figure is suitable for embedding in the bi-weekly report.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

HERE = Path(__file__).resolve().parent
SUMMARY_CSV = HERE / "biweekly_n3_summary.csv"
OUT = HERE / "phase0_n3"
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from color_palettes import OKABE_ITO  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from style_presets import get_base_style  # noqa: E402

# Approximate parameter counts for the x-axis (in billions)
PARAM_COUNTS_B = {
    "qwen2-0.5b": 0.494,
    "qwen2-1.5b": 1.54,
    "qwen2-7b":   7.62,
}

ENGINE_LABEL = {
    "silicon_w1ref": "SiliconLLM Stage 1 ref",
    "mlx_lm":        "mlx-lm (production)",
}
ENGINE_COLOR = {
    "silicon_w1ref": OKABE_ITO["vermillion"],
    "mlx_lm":        OKABE_ITO["blue"],
}
ENGINE_MARKER = {
    "silicon_w1ref": "o",
    "mlx_lm":        "s",
}


def load_summary() -> list[dict]:
    with SUMMARY_CSV.open() as f:
        return list(csv.DictReader(f))


def by_engine(rows: list[dict]) -> dict[str, list[tuple[float, float, float]]]:
    """Returns {engine: [(params_B, prefill_mean, prefill_stdev), ...]} sorted by params."""
    out: dict[str, list[tuple[float, float, float, float, float]]] = {}
    for r in rows:
        eng = r["engine"]
        params = PARAM_COUNTS_B.get(r["model"])
        if params is None:
            continue
        out.setdefault(eng, []).append(
            (
                params,
                float(r["prefill_tps_mean"]),
                float(r["prefill_tps_stdev"]),
                float(r["decode_tps_mean"]),
                float(r["decode_tps_stdev"]),
            )
        )
    for eng in out:
        out[eng].sort(key=lambda t: t[0])
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
    rows = load_summary()
    grouped = by_engine(rows)

    fig, (ax_pf, ax_dc) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.patch.set_facecolor("white")

    for eng, points in grouped.items():
        xs = [p[0] for p in points]
        pf_means = [p[1] for p in points]
        pf_errs = [p[2] for p in points]
        dc_means = [p[3] for p in points]
        dc_errs = [p[4] for p in points]
        label = ENGINE_LABEL.get(eng, eng)
        color = ENGINE_COLOR.get(eng, None)
        marker = ENGINE_MARKER.get(eng, "o")

        ax_pf.errorbar(
            xs,
            pf_means,
            yerr=pf_errs,
            label=label,
            color=color,
            marker=marker,
            markersize=4.2,
            capsize=2.2,
            capthick=0.7,
        )
        ax_dc.errorbar(
            xs,
            dc_means,
            yerr=dc_errs,
            label=label,
            color=color,
            marker=marker,
            markersize=4.2,
            capsize=2.2,
            capthick=0.7,
        )

    for ax, title, ylabel in [
        (ax_pf, "Prefill throughput", "Throughput (tok/s)"),
        (ax_dc, "Decode throughput",  "Throughput (tok/s)"),
    ]:
        ax.set_facecolor("white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.494, 1.54, 7.62])
        ax.set_xticklabels(["0.5B", "1.5B", "7B"])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.set_xlabel("Model size")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(axis="y", which="major", color="#d9dde3", linewidth=0.6, alpha=0.75)
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if ax is ax_pf:
            ax.legend(loc="lower left", frameon=False, handlelength=2.0)

    ax_pf.text(-0.12, 1.06, "A", transform=ax_pf.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_dc.text(-0.12, 1.06, "B", transform=ax_dc.transAxes, fontsize=10, fontweight="bold", va="top")

    fig.tight_layout(pad=0.8)
    save_publication_figure(
        fig,
        OUT,
        formats=["png", "pdf", "svg", "tiff"],
        dpi=600,
        transparent=False,
        facecolor="white",
    )
    print(f"Wrote: {OUT}.png/.pdf/.svg/.tiff")


if __name__ == "__main__":
    main()
