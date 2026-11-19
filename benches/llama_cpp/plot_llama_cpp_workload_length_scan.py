"""
Plot llama.cpp prompt+generation workload length scan.

Reads:
    benches/llama_cpp/llama_cpp_workload_length_scan_summary.csv

Writes:
    benches/llama_cpp/llama_cpp_workload_length_scan.png
    benches/llama_cpp/llama_cpp_workload_length_scan.pdf
    benches/llama_cpp/llama_cpp_workload_length_scan.svg
    benches/llama_cpp/llama_cpp_workload_length_scan.tiff
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "benches" / "llama_cpp" / "llama_cpp_workload_length_scan_summary.csv"
OUT = ROOT / "benches" / "llama_cpp" / "llama_cpp_workload_length_scan"
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

MODEL_COLOR = {
    "qwen2-0.5b": OKABE_ITO["blue"],
    "qwen2-1.5b": OKABE_ITO["vermillion"],
}

MODEL_MARKER = {
    "qwen2-0.5b": "o",
    "qwen2-1.5b": "s",
}


def load_rows() -> list[dict[str, str]]:
    with SUMMARY.open() as f:
        return list(csv.DictReader(f))


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["model"], []).append(row)
    for model_rows in out.values():
        model_rows.sort(key=lambda r: int(r["requested_prompt_len"]))
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
    by_model = grouped(rows)

    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    for model in ("qwen2-0.5b", "qwen2-1.5b"):
        model_rows = by_model[model]
        xs = [int(row["requested_prompt_len"]) for row in model_rows]
        ys = [float(row["mixed_tps_mean"]) for row in model_rows]
        yerr = [float(row["mixed_tps_stdev"]) for row in model_rows]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=MODEL_MARKER[model],
            markersize=4.2,
            capsize=2.4,
            capthick=0.7,
            color=MODEL_COLOR[model],
            label=MODEL_LABEL[model],
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 512])
    ax.set_xticklabels(["64", "128", "512"])
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Mixed throughput (tok/s)")
    ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.75)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left")

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
