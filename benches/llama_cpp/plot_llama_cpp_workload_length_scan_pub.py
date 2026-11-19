"""
Publication-style redraw of the llama.cpp workload length scan.

Reads:
    benches/llama_cpp/llama_cpp_workload_length_scan_summary.csv

Writes:
    benches/llama_cpp/llama_cpp_workload_length_scan_pub.png
    benches/llama_cpp/llama_cpp_workload_length_scan_pub.pdf
    benches/llama_cpp/llama_cpp_workload_length_scan_pub.svg
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
SUMMARY = ROOT / "benches" / "llama_cpp" / "llama_cpp_workload_length_scan_summary.csv"
OUT_STEM = ROOT / "benches" / "llama_cpp" / "llama_cpp_workload_length_scan_pub"

MODEL_LABEL = {
    "qwen2-0.5b": "Qwen2 0.5B",
    "qwen2-1.5b": "Qwen2 1.5B",
}

# Okabe-Ito colorblind-safe colors.
MODEL_COLOR = {
    "qwen2-0.5b": "#0072B2",
    "qwen2-1.5b": "#D55E00",
}

MODEL_MARKER = {
    "qwen2-0.5b": "o",
    "qwen2-1.5b": "s",
}


def apply_publication_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


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


def main() -> None:
    apply_publication_style()
    rows = load_rows()
    by_model = grouped(rows)

    fig, ax = plt.subplots(figsize=(5.2, 3.2))

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
            markersize=4.8,
            linewidth=1.4,
            capsize=2.5,
            capthick=0.8,
            color=MODEL_COLOR[model],
            label=MODEL_LABEL[model],
        )

    ax.set_xscale("log", base=2)
    ax.set_xticks([64, 128, 512])
    ax.set_xticklabels(["64", "128", "512"])
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Mixed throughput (tok/s)")
    ax.grid(axis="y", color="#d0d4d9", linewidth=0.55, alpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, loc="upper left", handlelength=2.0)

    fig.tight_layout(pad=0.8)
    for ext in ("png", "pdf", "svg"):
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if ext == "png":
            kwargs["dpi"] = 600
        fig.savefig(f"{OUT_STEM}.{ext}", **kwargs)
    print(f"Wrote {OUT_STEM}.png/.pdf/.svg")


if __name__ == "__main__":
    main()
