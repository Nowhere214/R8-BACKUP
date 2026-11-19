"""
Plot KV-cache microbenchmark results.

Reads:
    benches/kv_cache_microbench_summary.csv

Writes:
    benches/kv_cache_microbench.png
    benches/kv_cache_microbench.pdf
    benches/kv_cache_microbench.svg
    benches/kv_cache_microbench.tiff
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "benches" / "kv_cache_microbench_summary.csv"
OUT_BASE = ROOT / "benches" / "kv_cache_microbench"
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from color_palettes import OKABE_ITO  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from style_presets import get_base_style  # noqa: E402

PATH_LABEL = {
    "full_context_recompute": "Full-context recompute",
    "cached_one_token": "Cached one-token decode",
}

PATH_COLOR = {
    "full_context_recompute": OKABE_ITO["vermillion"],
    "cached_one_token": OKABE_ITO["blue"],
}

PATH_MARKER = {
    "full_context_recompute": "o",
    "cached_one_token": "s",
}


def load_rows() -> list[dict[str, str]]:
    with SUMMARY.open() as f:
        return list(csv.DictReader(f))


def grouped(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["path"], []).append(row)
    for path_rows in out.values():
        path_rows.sort(key=lambda r: int(r["context_len"]))
    return out


def speedup_rows(rows: list[dict[str, str]]) -> list[tuple[int, float]]:
    seen = {}
    for row in rows:
        context_len = int(row["context_len"])
        speedup = row.get("cached_speedup_x", "")
        if speedup:
            seen[context_len] = float(speedup)
    return sorted(seen.items())


def apply_style() -> None:
    style = get_base_style()
    style.update(
        {
            "axes.facecolor": "white",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.5,
        }
    )
    plt.rcParams.update(style)


def main() -> None:
    apply_style()
    rows = load_rows()
    by_path = grouped(rows)
    speedups = speedup_rows(rows)

    fig, (ax_latency, ax_speedup) = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.patch.set_facecolor("white")

    for ax in (ax_latency, ax_speedup):
        ax.set_facecolor("white")
        ax.set_xscale("log", base=2)
        ax.set_xticks([128, 256, 512, 1024, 2048])
        ax.set_xticklabels(["128", "256", "512", "1024", "2048"])
        ax.grid(axis="y", color="#d9dde3", linewidth=0.7, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b8bec8")
        ax.spines["bottom"].set_color("#b8bec8")
        ax.tick_params(color="#606875")
        ax.set_xlabel("Context length (tokens)")

    ax_latency.text(-0.13, 1.05, "A", transform=ax_latency.transAxes, fontsize=11, va="top")
    ax_speedup.text(-0.13, 1.05, "B", transform=ax_speedup.transAxes, fontsize=11, va="top")

    for path in ("full_context_recompute", "cached_one_token"):
        path_rows = by_path[path]
        xs = [int(row["context_len"]) for row in path_rows]
        ys = [float(row["latency_ms_median"]) for row in path_rows]
        ax_latency.plot(
            xs,
            ys,
            marker=PATH_MARKER[path],
            linewidth=1.7,
            markersize=5.0,
            color=PATH_COLOR[path],
            label=PATH_LABEL[path],
        )

    speedup_x = [x for x, _ in speedups]
    speedup_y = [y for _, y in speedups]
    ax_speedup.plot(
        speedup_x,
        speedup_y,
        marker="D",
        linewidth=1.7,
        markersize=4.8,
        color=OKABE_ITO["bluish_green"],
    )

    ax_latency.set_title("Decode-step latency", pad=8)
    ax_latency.set_ylabel("Median latency (ms, log scale)")
    ax_latency.set_yscale("log")
    ax_latency.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_latency.legend(frameon=False, loc="upper left")

    ax_speedup.set_title("Effective speedup", pad=8)
    ax_speedup.set_ylabel("Full recompute / cached")
    ax_speedup.set_ylim(0, max(speedup_y) * 1.14)

    fig.text(
        0.5,
        -0.015,
        "Synthetic Qwen2 0.5B-style grouped attention; hidden size 896; 14 query heads; 2 KV heads; N=20.",
        ha="center",
        fontsize=8,
        color="#526070",
    )
    fig.tight_layout()
    save_publication_figure(
        fig,
        OUT_BASE,
        formats=["png", "pdf", "svg", "tiff"],
        dpi=600,
        transparent=False,
        facecolor="white",
    )


if __name__ == "__main__":
    main()
