"""
Generate presentation-friendly figures from the Phase 0 N=3 benchmark data.

This script does not import MLX or run model inference. It only reads the CSV
files already produced by `benches/biweekly_n3.py`.

Usage:
    pdm run python benches/plot_presentation_assets.py
"""

from __future__ import annotations

import csv
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np


HERE = Path(__file__).resolve().parent
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SKILL_DIR / "scripts"))

from figure_export import save_publication_figure  # noqa: E402

SUMMARY_CSV = HERE / "biweekly_n3_summary.csv"
RESULTS_CSV = HERE / "biweekly_n3_results.csv"
OUT_DIR = HERE / "presentation_assets"

MODEL_LABEL = {
    "qwen2-0.5b": "Qwen2 0.5B",
    "qwen2-1.5b": "Qwen2 1.5B",
    "qwen2-7b": "Qwen2 7B",
}

ENGINE_LABEL = {
    "silicon_w1ref": "SiliconLLM Stage 1 ref",
    "mlx_lm": "mlx-lm",
}

ENGINE_COLOR = {
    "silicon_w1ref": "#c83e4d",
    "mlx_lm": "#2f6f9f",
}
NEUTRAL = "#596273"
NEUTRAL_LIGHT = "#a4acb8"

MODEL_ORDER = ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b"]


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def summary_by_key(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(r["engine"], r["model"]): r for r in rows}


def save(fig: plt.Figure, filename: str) -> None:
    OUT_DIR.mkdir(exist_ok=True)
    fig.patch.set_facecolor("#ffffff")
    save_publication_figure(
        fig,
        OUT_DIR / Path(filename).stem,
        formats=["png", "pdf", "svg", "tiff"],
        dpi=300,
        transparent=False,
        facecolor="#ffffff",
    )
    plt.close(fig)


def annotate_bars(ax: plt.Axes, bars, suffix: str = "", fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        if height <= 0:
            continue
        ax.annotate(
            fmt.format(height) + suffix,
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_gap_ratios(summary: list[dict[str, str]]) -> None:
    data = summary_by_key(summary)
    models = ["qwen2-0.5b", "qwen2-1.5b"]
    prefill = []
    decode = []
    for model in models:
        ref = data[("silicon_w1ref", model)]
        mlx = data[("mlx_lm", model)]
        prefill.append(float(mlx["prefill_tps_mean"]) / float(ref["prefill_tps_mean"]))
        decode.append(float(mlx["decode_tps_mean"]) / float(ref["decode_tps_mean"]))

    x = np.arange(len(models))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars_pf = ax.bar(x - width / 2, prefill, width, label="Prefill gap", color=NEUTRAL_LIGHT)
    bars_dc = ax.bar(x + width / 2, decode, width, label="Decode gap", color=NEUTRAL)
    annotate_bars(ax, bars_pf, suffix="x")
    annotate_bars(ax, bars_dc, suffix="x")
    ax.set_title("mlx-lm speedup over SiliconLLM Stage 1 ref")
    ax.set_ylabel("Throughput ratio, higher means larger gap")
    ax.set_xticks(x, [MODEL_LABEL[m] for m in models])
    ax.set_ylim(0, max(decode) * 1.22)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "phase0_gap_ratios.png")


def plot_phase0_overview_slide(summary: list[dict[str, str]]) -> None:
    params = {"qwen2-0.5b": 0.494, "qwen2-1.5b": 1.54, "qwen2-7b": 7.62}
    grouped: dict[str, list[dict[str, str]]] = {"silicon_w1ref": [], "mlx_lm": []}
    for row in summary:
        grouped.setdefault(row["engine"], []).append(row)
    for engine in grouped:
        grouped[engine].sort(key=lambda r: params[r["model"]])

    fig, (ax_pf, ax_dc) = plt.subplots(1, 2, figsize=(13.5, 7.2))
    fig.patch.set_facecolor("#ffffff")
    for ax in (ax_pf, ax_dc):
        ax.set_facecolor("#ffffff")
        for spine in ax.spines.values():
            spine.set_color("#d5d9df")
        ax.grid(which="major", color="#9aa3ad", alpha=0.22, linewidth=0.9)
        ax.grid(which="minor", color="#9aa3ad", alpha=0.12, linewidth=0.6)
        ax.tick_params(colors="#25313f", labelsize=10)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Model parameters, billions", fontsize=11, color="#25313f")
        ax.set_xticks([0.494, 1.54, 7.62])
        ax.set_xticklabels(["0.5B", "1.5B", "7B"])
        ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    fig.suptitle(
        "Phase 0 Benchmark: Decode Is the Main Bottleneck",
        fontsize=23,
        color="#1f2937",
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        "Apple Silicon, Qwen2 models, N=3 repeated runs; markers show mean throughput, error bars show standard deviation",
        ha="center",
        fontsize=11.5,
        color="#526070",
    )

    for engine, marker in [("silicon_w1ref", "o"), ("mlx_lm", "s")]:
        rows = grouped[engine]
        xs = np.array([params[r["model"]] for r in rows])
        color = ENGINE_COLOR[engine]
        label = ENGINE_LABEL[engine]

        pf = np.array([float(r["prefill_tps_mean"]) for r in rows])
        pf_err = np.array([float(r["prefill_tps_stdev"]) for r in rows])
        dc = np.array([float(r["decode_tps_mean"]) for r in rows])
        dc_err = np.array([float(r["decode_tps_stdev"]) for r in rows])

        ax_pf.errorbar(
            xs,
            pf,
            yerr=pf_err,
            color=color,
            marker=marker,
            markersize=8,
            capsize=4,
            linewidth=2.4,
            label=label,
        )
        ax_dc.errorbar(
            xs,
            dc,
            yerr=dc_err,
            color=color,
            marker=marker,
            markersize=8,
            capsize=4,
            linewidth=2.4,
            label=label,
        )

    ax_pf.set_title("Prefill throughput", fontsize=16, color="#25313f", pad=12)
    ax_dc.set_title("Decode throughput", fontsize=16, color="#25313f", pad=12)
    ax_pf.set_ylabel("tokens/sec, log scale", fontsize=11, color="#25313f")
    ax_dc.set_ylabel("tokens/sec, log scale", fontsize=11, color="#25313f")

    ax_pf.set_ylim(170, 7000)
    ax_dc.set_ylim(0.7, 650)

    ax_pf.text(
        7.62,
        470,
        "No measured\nStage 1 ref 7B",
        ha="center",
        va="center",
        fontsize=10,
        color="#526070",
        bbox={"boxstyle": "round,pad=0.35", "fc": "#ffffff", "ec": "#d5d9df", "alpha": 0.96},
    )

    for ax in (ax_pf, ax_dc):
        ax.legend(
            loc="lower left",
            frameon=False,
            fontsize=11,
            handlelength=2.5,
        )

    footer = (
        "Measured configurations: Stage 1 ref on Qwen2 0.5B/1.5B; "
        "mlx-lm on Qwen2 0.5B/1.5B/7B. Stage 1 ref 7B omitted due to memory feasibility risk."
    )
    fig.text(0.5, 0.035, footer, ha="center", fontsize=9.5, color="#667085")

    fig.subplots_adjust(left=0.07, right=0.985, top=0.84, bottom=0.14, wspace=0.16)
    save(fig, "phase0_n3_slide.png")


def plot_wall_time(summary: list[dict[str, str]]) -> None:
    rows = [r for r in summary if (r["engine"], r["model"]) != ("silicon_w1ref", "qwen2-7b")]
    labels = [f"{MODEL_LABEL[r['model']]}\n{ENGINE_LABEL[r['engine']]}" for r in rows]
    values = [float(r["wall_s_mean"]) for r in rows]
    colors = [ENGINE_COLOR[r["engine"]] for r in rows]

    fig, ax = plt.subplots(figsize=(9, 4.8))
    bars = ax.bar(np.arange(len(rows)), values, color=colors)
    annotate_bars(ax, bars, suffix="s", fmt="{:.1f}")
    ax.set_title("End-to-end wall time for the benchmark prompt")
    ax.set_ylabel("Seconds, log scale")
    ax.set_xticks(np.arange(len(rows)), labels)
    ax.set_yscale("log")
    ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    save(fig, "phase0_wall_time.png")


def plot_memory(summary: list[dict[str, str]]) -> None:
    x = np.arange(len(MODEL_ORDER))
    width = 0.34
    data = summary_by_key(summary)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for idx, engine in enumerate(["silicon_w1ref", "mlx_lm"]):
        vals = []
        for model in MODEL_ORDER:
            row = data.get((engine, model))
            vals.append(float(row["peak_memory_gb_max"]) if row else np.nan)
        offset = (idx - 0.5) * width
        bars = ax.bar(
            x + offset,
            vals,
            width,
            label=ENGINE_LABEL[engine],
            color=ENGINE_COLOR[engine],
        )
        annotate_bars(ax, bars, suffix=" GB")

    ax.annotate(
        "Stage 1 ref 7B not feasible on 18 GB",
        xy=(x[2] - width / 2, 8.0),
        xytext=(x[1] + 0.1, 9.5),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
        fontsize=9,
    )
    ax.set_title("Peak memory during Phase 0 runs")
    ax.set_ylabel("Peak memory, GB")
    ax.set_xticks(x, [MODEL_LABEL[m] for m in MODEL_ORDER])
    ax.set_ylim(0, 11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "phase0_memory.png")


def plot_decode_scaling(summary: list[dict[str, str]]) -> None:
    params = {"qwen2-0.5b": 0.494, "qwen2-1.5b": 1.54, "qwen2-7b": 7.62}
    mlx_rows = [r for r in summary if r["engine"] == "mlx_lm"]
    mlx_rows.sort(key=lambda r: params[r["model"]])
    xs = np.array([params[r["model"]] for r in mlx_rows])
    ys = np.array([float(r["decode_tps_mean"]) for r in mlx_rows])

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.plot(xs, ys, marker="o", color=ENGINE_COLOR["mlx_lm"], linewidth=2.2, markersize=8)
    for x, y, r in zip(xs, ys, mlx_rows):
        ax.annotate(MODEL_LABEL[r["model"]], xy=(x, y), xytext=(6, 6), textcoords="offset points")
    ax.annotate(
        "Params: 0.5B -> 7B is 15.4x\nDecode slowdown: 9.0x",
        xy=(xs[-1], ys[-1]),
        xytext=(0.8, 60),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
        fontsize=10,
    )
    ax.set_title("mlx-lm decode scaling is sublinear in model size")
    ax.set_xlabel("Model parameters, billions")
    ax.set_ylabel("Decode throughput, tokens/sec")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.grid(which="both", alpha=0.25)
    fig.tight_layout()
    save(fig, "phase0_mlx_decode_scaling.png")


def fit_power_law(x1: float, y1: float, x2: float, y2: float, x3: float) -> float:
    """Fit y = a * x^b through two measured points and estimate y at x3."""
    exponent = np.log(y2 / y1) / np.log(x2 / x1)
    return y2 * (x3 / x2) ** exponent


def plot_week1_ref_7b_boundary(summary: list[dict[str, str]]) -> None:
    params = {"qwen2-0.5b": 0.494, "qwen2-1.5b": 1.54, "qwen2-7b": 7.62}
    data = summary_by_key(summary)
    ref_05 = data[("silicon_w1ref", "qwen2-0.5b")]
    ref_15 = data[("silicon_w1ref", "qwen2-1.5b")]

    x_measured = np.array([params["qwen2-0.5b"], params["qwen2-1.5b"]])
    decode_measured = np.array([
        float(ref_05["decode_tps_mean"]),
        float(ref_15["decode_tps_mean"]),
    ])
    prefill_measured = np.array([
        float(ref_05["prefill_tps_mean"]),
        float(ref_15["prefill_tps_mean"]),
    ])
    memory_measured = np.array([
        float(ref_05["peak_memory_gb_max"]),
        float(ref_15["peak_memory_gb_max"]),
    ])

    x_7b = params["qwen2-7b"]
    decode_est = fit_power_law(x_measured[0], decode_measured[0], x_measured[1], decode_measured[1], x_7b)
    prefill_est = fit_power_law(x_measured[0], prefill_measured[0], x_measured[1], prefill_measured[1], x_7b)
    memory_est = fit_power_law(x_measured[0], memory_measured[0], x_measured[1], memory_measured[1], x_7b)

    fig, (ax_tps, ax_mem) = plt.subplots(1, 2, figsize=(11, 4.8))
    red = ENGINE_COLOR["silicon_w1ref"]

    ax_tps.plot(x_measured, decode_measured, marker="o", color=red, linewidth=2.2, label="Decode measured")
    ax_tps.plot(
        [x_measured[-1], x_7b],
        [decode_measured[-1], decode_est],
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markeredgewidth=2,
        color=red,
        linewidth=1.8,
        label="Decode extrapolated",
    )
    ax_tps.plot(
        x_measured,
        prefill_measured,
        marker="s",
        color=red,
        alpha=0.62,
        linewidth=2.0,
        label="Prefill measured",
    )
    ax_tps.plot(
        [x_measured[-1], x_7b],
        [prefill_measured[-1], prefill_est],
        linestyle="--",
        marker="s",
        markerfacecolor="white",
        markeredgewidth=2,
        color=red,
        alpha=0.62,
        linewidth=1.8,
        label="Prefill extrapolated",
    )
    ax_tps.annotate(
        f"7B not measured\nestimated decode ≈ {decode_est:.1f} tok/s",
        xy=(x_7b, decode_est),
        xytext=(1.2, 1.8),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
        fontsize=9,
    )
    ax_tps.set_title("Stage 1 ref scaling: measured vs extrapolated")
    ax_tps.set_xlabel("Model parameters, billions")
    ax_tps.set_ylabel("Throughput, tokens/sec")
    ax_tps.set_xscale("log")
    ax_tps.set_yscale("log")
    ax_tps.grid(which="both", alpha=0.25)
    ax_tps.legend(frameon=False, fontsize=8)

    ax_mem.plot(x_measured, memory_measured, marker="o", color=red, linewidth=2.0, label="Measured")
    ax_mem.plot(
        [x_measured[-1], x_7b],
        [memory_measured[-1], memory_est],
        linestyle="--",
        marker="o",
        markerfacecolor="white",
        markeredgewidth=2,
        color=red,
        linewidth=1.8,
        label="Extrapolated",
    )
    ax_mem.axhline(18, color="#333333", linestyle=":", linewidth=1.6, label="18 GB unified memory")
    ax_mem.annotate(
        f"Estimated peak ≈ {memory_est:.1f} GB\nnear the machine limit",
        xy=(x_7b, memory_est),
        xytext=(0.75, 12.0),
        arrowprops={"arrowstyle": "->", "lw": 1.2, "color": "#333333"},
        fontsize=9,
    )
    ax_mem.set_title("Why Stage 1 ref 7B was not a safe run")
    ax_mem.set_xlabel("Model parameters, billions")
    ax_mem.set_ylabel("Peak memory, GB")
    ax_mem.set_xscale("log")
    ax_mem.set_ylim(0, 20)
    ax_mem.grid(axis="y", alpha=0.25)
    ax_mem.legend(frameon=False, fontsize=8)

    fig.suptitle("SiliconLLM Stage 1 ref 7B boundary, not a measured datapoint", y=1.03)
    fig.tight_layout()
    save(fig, "phase0_stage1_ref_7b_boundary.png")


def plot_variability(summary: list[dict[str, str]]) -> None:
    labels = [f"{MODEL_LABEL[r['model']]}\n{ENGINE_LABEL[r['engine']]}" for r in summary]
    prefill_cv = [float(r["prefill_tps_cv_pct"]) for r in summary]
    decode_cv = [float(r["decode_tps_cv_pct"]) for r in summary]
    x = np.arange(len(summary))
    width = 0.34

    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    bars_pf = ax.bar(x - width / 2, prefill_cv, width, label="Prefill CV", color=NEUTRAL_LIGHT)
    bars_dc = ax.bar(x + width / 2, decode_cv, width, label="Decode CV", color=NEUTRAL)
    annotate_bars(ax, bars_pf, suffix="%")
    annotate_bars(ax, bars_dc, suffix="%")
    ax.set_title("N=3 repeatability check")
    ax.set_ylabel("Coefficient of variation, percent")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, max(prefill_cv + decode_cv) * 1.35)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "phase0_repeatability_cv.png")


def plot_rep_scatter(results: list[dict[str, str]]) -> None:
    fig, (ax_pf, ax_dc) = plt.subplots(1, 2, figsize=(11, 4.8), sharex=False)
    for ax, metric, title in [
        (ax_pf, "prefill_tps", "Prefill per-run measurements"),
        (ax_dc, "decode_tps", "Decode per-run measurements"),
    ]:
        positions = []
        labels = []
        for i, (engine, model) in enumerate(
            [
                ("silicon_w1ref", "qwen2-0.5b"),
                ("mlx_lm", "qwen2-0.5b"),
                ("silicon_w1ref", "qwen2-1.5b"),
                ("mlx_lm", "qwen2-1.5b"),
                ("mlx_lm", "qwen2-7b"),
            ]
        ):
            vals = [
                float(r[metric])
                for r in results
                if r["engine"] == engine and r["model"] == model
            ]
            xs = np.full(len(vals), i, dtype=float) + np.linspace(-0.08, 0.08, len(vals))
            ax.scatter(xs, vals, color=ENGINE_COLOR[engine], s=42)
            positions.append(i)
            labels.append(f"{MODEL_LABEL[model]}\n{ENGINE_LABEL[engine]}")
        ax.set_title(title)
        ax.set_xticks(positions, labels, rotation=0)
        ax.set_yscale("log")
        ax.set_ylabel("tokens/sec, log scale")
        ax.grid(axis="y", which="both", alpha=0.25)
    fig.tight_layout()
    save(fig, "phase0_rep_scatter.png")


def main() -> None:
    summary = load_csv(SUMMARY_CSV)
    results = load_csv(RESULTS_CSV)

    plot_phase0_overview_slide(summary)
    plot_gap_ratios(summary)
    plot_wall_time(summary)
    plot_memory(summary)
    plot_decode_scaling(summary)
    plot_week1_ref_7b_boundary(summary)
    plot_variability(summary)
    plot_rep_scatter(results)

    print(f"Wrote presentation figures to: {OUT_DIR}")


if __name__ == "__main__":
    main()
