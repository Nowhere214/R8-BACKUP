"""
Analyze prompt-length scaling from the Phase 0 workload scan.

This script reuses existing CSV results. It does not run model inference.
The goal is to turn the prompt-length scan into a small cost model:

    decode_latency_per_token ~= intercept + slope * effective_context_len

where effective_context_len approximates the average context seen during
fixed-length generation.

Outputs:
    benches/decode_gap_scaling_analysis.csv
    benches/decode_gap_scaling_fit.csv
    benches/decode_gap_scaling_analysis.png
    benches/decode_gap_scaling_analysis.pdf
    benches/decode_gap_scaling_analysis.svg
    benches/decode_gap_scaling_analysis.tiff
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


ROOT = Path(__file__).resolve().parent.parent
INPUTS = [
    ROOT / "benches" / "workload_length_scan_summary.csv",
    ROOT / "benches" / "workload_length_scan_1p5b_summary.csv",
]
OUT_ANALYSIS = ROOT / "benches" / "decode_gap_scaling_analysis.csv"
OUT_FIT = ROOT / "benches" / "decode_gap_scaling_fit.csv"
OUT_FIG = ROOT / "benches" / "decode_gap_scaling_analysis"
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

MODEL_LINESTYLE = {
    "qwen2-0.5b": "-",
    "qwen2-1.5b": "--",
}


@dataclass(frozen=True)
class Row:
    engine: str
    model: str
    prompt_len: int
    gen_len: int
    decode_tps: float
    decode_tps_stdev: float

    @property
    def effective_context_len(self) -> float:
        # Generation length is fixed in this scan. During token-by-token
        # decode, the average context is prompt + half of the generated tail.
        return self.prompt_len + (self.gen_len - 1) / 2

    @property
    def decode_ms_per_token(self) -> float:
        return 1000.0 / self.decode_tps


def read_rows() -> list[Row]:
    rows: list[Row] = []
    for path in INPUTS:
        with path.open() as f:
            for item in csv.DictReader(f):
                rows.append(
                    Row(
                        engine=item["engine"],
                        model=item["model"],
                        prompt_len=int(item["requested_prompt_len"]),
                        gen_len=int(item["requested_gen_len"]),
                        decode_tps=float(item["decode_tps_mean"]),
                        decode_tps_stdev=float(item["decode_tps_stdev"]),
                    )
                )
    return rows


def linear_fit(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    ss_xx = sum((x - x_mean) ** 2 for x in xs)
    ss_xy = sum((x - x_mean) * (y - y_mean) for x, y in points)
    slope = ss_xy / ss_xx if ss_xx else 0.0
    intercept = y_mean - slope * x_mean
    ss_tot = sum((y - y_mean) ** 2 for y in ys)
    ss_res = sum((y - (intercept + slope * x)) ** 2 for x, y in points)
    r2 = 1.0 - ss_res / ss_tot if ss_tot else 1.0
    return intercept, slope, r2


def write_analysis(rows: list[Row]) -> None:
    fields = [
        "engine",
        "model",
        "prompt_len",
        "gen_len",
        "effective_context_len",
        "decode_tps",
        "decode_tps_stdev",
        "decode_ms_per_token",
    ]
    with OUT_ANALYSIS.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda r: (r.model, r.engine, r.prompt_len)):
            writer.writerow(
                {
                    "engine": row.engine,
                    "model": row.model,
                    "prompt_len": row.prompt_len,
                    "gen_len": row.gen_len,
                    "effective_context_len": f"{row.effective_context_len:.1f}",
                    "decode_tps": f"{row.decode_tps:.6f}",
                    "decode_tps_stdev": f"{row.decode_tps_stdev:.6f}",
                    "decode_ms_per_token": f"{row.decode_ms_per_token:.6f}",
                }
            )


def write_fit(rows: list[Row]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for model in sorted({r.model for r in rows}):
        for engine in sorted({r.engine for r in rows if r.model == model}):
            group = [r for r in rows if r.model == model and r.engine == engine]
            points = [(r.effective_context_len, r.decode_ms_per_token) for r in group]
            intercept, slope, r2 = linear_fit(points)
            predicted_1024 = intercept + slope * (1024 + (group[0].gen_len - 1) / 2)
            out.append(
                {
                    "engine": engine,
                    "model": model,
                    "intercept_ms": f"{intercept:.6f}",
                    "slope_ms_per_context_token": f"{slope:.6f}",
                    "r2": f"{r2:.6f}",
                    "predicted_decode_ms_per_token_at_prompt_1024": f"{predicted_1024:.6f}",
                    "predicted_decode_tps_at_prompt_1024": f"{1000.0 / predicted_1024:.6f}"
                    if predicted_1024 > 0
                    else "",
                }
            )

    with OUT_FIT.open("w", newline="") as f:
        fieldnames = list(out[0])
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(out)
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


def plot(rows: list[Row], fits: list[dict[str, str]]) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0))
    fig.patch.set_facecolor("white")

    for ax in axes:
        ax.set_facecolor("white")
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.75)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel("Effective context length (tokens)")

    ax_latency, ax_slope = axes
    ax_latency.text(-0.12, 1.06, "A", transform=ax_latency.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_slope.text(-0.12, 1.06, "B", transform=ax_slope.transAxes, fontsize=10, fontweight="bold", va="top")
    ax_latency.set_title("Decode latency vs. context length")
    ax_latency.set_ylabel("Decode latency per token (ms)")
    ax_latency.set_yscale("log")
    ax_latency.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}"))

    fit_by_key = {(fit["model"], fit["engine"]): fit for fit in fits}
    for model in ("qwen2-0.5b", "qwen2-1.5b"):
        for engine in ["silicon_w1ref", "mlx_lm"]:
            group = sorted(
                [r for r in rows if r.model == model and r.engine == engine],
                key=lambda r: r.effective_context_len,
            )
            if not group:
                continue
            color = ENGINE_COLOR[engine]
            label = f"{MODEL_LABEL[model]} {ENGINE_LABEL[engine]}"
            xs = [r.effective_context_len for r in group]
            ys = [r.decode_ms_per_token for r in group]
            ax_latency.plot(
                xs,
                ys,
                marker=ENGINE_MARKER[engine],
                linestyle=MODEL_LINESTYLE[model],
                color=color,
                label=label,
            )
            fit = fit_by_key[(model, engine)]
            intercept = float(fit["intercept_ms"])
            slope = float(fit["slope_ms_per_context_token"])
            fit_xs = [min(xs), max(xs)]
            fit_ys = [intercept + slope * x for x in fit_xs]
            ax_latency.plot(fit_xs, fit_ys, linestyle=":", linewidth=1.0, color=color, alpha=0.65)

    ax_latency.legend(frameon=False, loc="upper left", handlelength=2.4)

    ax_slope.set_title("Fitted context-length cost")
    ax_slope.set_ylabel("Slope (ms per context token)")
    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []
    for model in ["qwen2-0.5b", "qwen2-1.5b"]:
        for engine in ["silicon_w1ref", "mlx_lm"]:
            fit = fit_by_key[(model, engine)]
            labels.append(f"{MODEL_LABEL[model]}\n{ENGINE_LABEL[engine]}")
            values.append(float(fit["slope_ms_per_context_token"]))
            colors.append(ENGINE_COLOR[engine])
    ax_slope.bar(range(len(values)), values, color=colors, width=0.64)
    ax_slope.set_xticks(range(len(labels)))
    ax_slope.set_xticklabels(labels, fontsize=6.2)

    fig.tight_layout(pad=0.8)
    save_publication_figure(
        fig,
        OUT_FIG,
        formats=["png", "pdf", "svg", "tiff"],
        dpi=600,
        transparent=False,
        facecolor="white",
    )


def print_summary(fits: list[dict[str, str]]) -> None:
    print("Decode gap scaling fit")
    print("-" * 96)
    print(
        f"{'Model':<12} {'Engine':<16} {'Slope ms/token':>16} "
        f"{'R^2':>8} {'Pred 1024 tok/s':>18}"
    )
    print("-" * 96)
    for fit in fits:
        print(
            f"{fit['model']:<12} {fit['engine']:<16} "
            f"{float(fit['slope_ms_per_context_token']):>16.4f} "
            f"{float(fit['r2']):>8.3f} "
            f"{float(fit['predicted_decode_tps_at_prompt_1024']):>18.2f}"
        )
    print("-" * 96)
    print(f"Analysis -> {OUT_ANALYSIS}")
    print(f"Fit      -> {OUT_FIT}")
    print(f"Figure   -> {OUT_FIG}.png/.pdf/.svg/.tiff")


def main() -> None:
    rows = read_rows()
    write_analysis(rows)
    fits = write_fit(rows)
    plot(rows, fits)
    print_summary(fits)


if __name__ == "__main__":
    main()
