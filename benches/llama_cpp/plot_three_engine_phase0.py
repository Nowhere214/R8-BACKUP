from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parents[2]
PHASE0_SUMMARY = ROOT / "benches" / "biweekly_n3_summary.csv"
LLAMA_SUMMARY = ROOT / "benches" / "llama_cpp" / "llama_cpp_b9354_summary.csv"
OUT = ROOT / "benches" / "llama_cpp" / "phase0_three_engine_llama_cpp"
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from color_palettes import OKABE_ITO  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from style_presets import get_base_style  # noqa: E402

PARAM_COUNTS_B = {
    "qwen2-0.5b": 0.494,
    "qwen2-1.5b": 1.54,
    "qwen2-7b": 7.62,
}

ENGINE_LABEL = {
    "silicon_w1ref": "SiliconLLM Stage 1 ref",
    "mlx_lm": "mlx-lm",
    "llama_cpp_metal": "llama.cpp Metal",
}

ENGINE_COLOR = {
    "silicon_w1ref": OKABE_ITO["vermillion"],
    "mlx_lm": OKABE_ITO["blue"],
    "llama_cpp_metal": OKABE_ITO["bluish_green"],
}

ENGINE_MARKER = {
    "silicon_w1ref": "o",
    "mlx_lm": "s",
    "llama_cpp_metal": "^",
}


def load_rows() -> list[dict[str, str]]:
    with PHASE0_SUMMARY.open() as f:
        rows = list(csv.DictReader(f))
    with LLAMA_SUMMARY.open() as f:
        rows.extend(csv.DictReader(f))
    return rows


def grouped_points(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["engine"], []).append(row)
    for engine in out:
        out[engine].sort(key=lambda r: PARAM_COUNTS_B[r["model"]])
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
    grouped = grouped_points(rows)

    fig, (ax_pf, ax_dc) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.patch.set_facecolor("white")

    for ax in (ax_pf, ax_dc):
        ax.set_facecolor("white")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xticks([0.494, 1.54, 7.62])
        ax.set_xticklabels(["0.5B", "1.5B", "7B"])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
        ax.grid(axis="y", which="major", color="#d9dde3", linewidth=0.6, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Model size")
        ax.set_ylabel("Throughput (tok/s)")

    ax_pf.text(-0.12, 1.06, "A", transform=ax_pf.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_dc.text(-0.12, 1.06, "B", transform=ax_dc.transAxes, fontsize=10, fontweight="bold", va="top")

    for engine in ["silicon_w1ref", "llama_cpp_metal", "mlx_lm"]:
        points = grouped.get(engine, [])
        xs = [PARAM_COUNTS_B[p["model"]] for p in points]
        pf = [float(p["prefill_tps_mean"]) for p in points]
        dc = [float(p["decode_tps_mean"]) for p in points]
        pf_err = [float(p["prefill_tps_stdev"]) for p in points]
        dc_err = [float(p["decode_tps_stdev"]) for p in points]
        ax_pf.errorbar(
            xs,
            pf,
            yerr=pf_err,
            label=ENGINE_LABEL[engine],
            color=ENGINE_COLOR[engine],
            marker=ENGINE_MARKER[engine],
            markersize=4.2,
            capsize=2.2,
            capthick=0.7,
        )
        ax_dc.errorbar(
            xs,
            dc,
            yerr=dc_err,
            label=ENGINE_LABEL[engine],
            color=ENGINE_COLOR[engine],
            marker=ENGINE_MARKER[engine],
            markersize=4.2,
            capsize=2.2,
            capthick=0.7,
        )

    ax_pf.set_title("Prefill throughput")
    ax_dc.set_title("Decode throughput")
    ax_pf.set_ylim(150, 6500)
    ax_dc.set_ylim(0.7, 650)

    for ax in (ax_pf, ax_dc):
        ax.legend(loc="lower left", frameon=False, handlelength=2.0)

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
