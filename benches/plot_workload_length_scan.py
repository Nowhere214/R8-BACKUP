"""
Plot the Phase 0 workload length scan.

Reads:
    benches/workload_length_scan_summary.csv

Writes:
    benches/workload_length_scan.png
    benches/workload_length_scan.pdf
    benches/workload_length_scan.svg
    benches/workload_length_scan.tiff
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "benches" / "workload_length_scan_summary.csv"
OUT = ROOT / "benches" / "workload_length_scan"
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from color_palettes import OKABE_ITO  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from style_presets import get_base_style  # noqa: E402

ENGINE_LABEL = {
    "silicon_w1ref": "SiliconLLM Stage 1 ref",
    "mlx_lm": "mlx-lm",
}

ENGINE_COLOR = {
    "silicon_w1ref": OKABE_ITO["vermillion"],
    "mlx_lm": OKABE_ITO["blue"],
}

ENGINE_MARKER = {
    "silicon_w1ref": "o",
    "mlx_lm": "s",
}


def load_rows() -> list[dict[str, str]]:
    with SUMMARY.open() as f:
        return list(csv.DictReader(f))


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["engine"], []).append(row)
    for engine_rows in out.values():
        engine_rows.sort(key=lambda r: int(r["requested_prompt_len"]))
    return out


def gap_rows(rows: list[dict[str, str]]) -> list[tuple[int, float]]:
    by_prompt: dict[int, dict[str, float]] = {}
    for row in rows:
        prompt_len = int(row["requested_prompt_len"])
        by_prompt.setdefault(prompt_len, {})[row["engine"]] = float(row["decode_tps_mean"])
    out = []
    for prompt_len, values in by_prompt.items():
        if "mlx_lm" in values and "silicon_w1ref" in values:
            out.append((prompt_len, values["mlx_lm"] / values["silicon_w1ref"]))
    return sorted(out)


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
    by_engine = grouped(rows)
    gaps = gap_rows(rows)

    fig, (ax_decode, ax_gap) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.patch.set_facecolor("white")

    for ax in (ax_decode, ax_gap):
        ax.set_facecolor("white")
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 512])
        ax.set_xticklabels(["64", "128", "512"])
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Prompt length (tokens)")

    ax_decode.set_yscale("log")
    ax_decode.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_decode.text(-0.12, 1.06, "A", transform=ax_decode.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_gap.text(-0.12, 1.06, "B", transform=ax_gap.transAxes, fontsize=10, fontweight="bold", va="top")

    for engine in ("silicon_w1ref", "mlx_lm"):
        engine_rows = by_engine.get(engine, [])
        xs = [int(row["requested_prompt_len"]) for row in engine_rows]
        ys = [float(row["decode_tps_mean"]) for row in engine_rows]
        yerr = [float(row["decode_tps_stdev"]) for row in engine_rows]
        ax_decode.errorbar(
            xs,
            ys,
            yerr=yerr,
            label=ENGINE_LABEL[engine],
            color=ENGINE_COLOR[engine],
            marker=ENGINE_MARKER[engine],
            markersize=4.2,
            capsize=2.2,
            capthick=0.7,
        )

    gap_xs = [prompt_len for prompt_len, _ in gaps]
    gap_ys = [gap for _, gap in gaps]
    ax_gap.plot(
        gap_xs,
        gap_ys,
        color=OKABE_ITO["bluish_green"],
        marker="D",
    )
    for x, y in zip(gap_xs, gap_ys):
        ax_gap.text(x, y * 1.05, f"{y:.1f}x", ha="center", va="bottom", fontsize=7)

    ax_decode.set_title("Decode throughput")
    ax_decode.set_ylabel("Throughput (tok/s)")
    ax_decode.legend(loc="lower left", frameon=False)
    ax_decode.set_ylim(4, 420)

    ax_gap.set_title("Decode gap")
    ax_gap.set_ylabel("mlx-lm / Stage 1 ref")
    ax_gap.set_ylim(0, max(gap_ys) * 1.28 if gap_ys else 1)

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
