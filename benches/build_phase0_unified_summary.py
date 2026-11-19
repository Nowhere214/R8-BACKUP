"""
Build a unified Phase 0 result table and reproducibility coverage figure.

This script reads existing CSV files only. It does not import MLX or run model
inference. The goal is to make the Phase 0 evidence package easier to audit:

    - one long-format CSV for all key measurements
    - one compact coverage matrix figure for reports

Outputs:
    benches/phase0_unified_summary.csv
    benches/phase0_reproducibility_matrix.png
    benches/phase0_reproducibility_matrix.pdf
    benches/phase0_reproducibility_matrix.svg
    benches/phase0_reproducibility_matrix.tiff
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent.parent
OUT_CSV = ROOT / "benches" / "phase0_unified_summary.csv"
OUT_FIG = ROOT / "benches" / "phase0_reproducibility_matrix"
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from color_palettes import OKABE_ITO  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from style_presets import get_base_style  # noqa: E402


FILES = {
    "model_size_mlx": ROOT / "benches" / "biweekly_n3_summary.csv",
    "model_size_llama_cpp": ROOT / "benches" / "llama_cpp" / "llama_cpp_b9354_summary.csv",
    "length_scan_0p5b": ROOT / "benches" / "workload_length_scan_summary.csv",
    "length_scan_1p5b": ROOT / "benches" / "workload_length_scan_1p5b_summary.csv",
    "llama_cpp_mixed_workload": ROOT
    / "benches"
    / "llama_cpp"
    / "llama_cpp_workload_length_scan_summary.csv",
    "kv_cache_microbench": ROOT / "benches" / "kv_cache_microbench_summary.csv",
    "decode_gap_scaling": ROOT / "benches" / "decode_gap_scaling_fit.csv",
}

MODEL_LABEL = {
    "qwen2-0.5b": "Qwen2 0.5B",
    "qwen2-1.5b": "Qwen2 1.5B",
    "qwen2-7b": "Qwen2 7B",
    "synthetic-qwen2-0.5b-attn": "Synthetic Qwen2 0.5B attention",
}

ENGINE_LABEL = {
    "silicon_w1ref": "SiliconLLM Stage 1 ref",
    "mlx_lm": "mlx-lm",
    "llama_cpp_metal": "llama.cpp Metal",
    "synthetic_full_context": "Full-context recompute",
    "synthetic_cached_decode": "Cached one-token decode",
}

EXPERIMENT_LABEL = {
    "model_size_engine_comparison": "Model-size baseline",
    "llama_cpp_metal_baseline": "llama.cpp baseline",
    "prompt_length_decode_scan": "Prompt-length scan",
    "llama_cpp_mixed_workload_scan": "llama.cpp mixed scan",
    "kv_cache_microbenchmark": "KV-cache microbench",
    "decode_gap_scaling_fit": "Decode cost model",
}

UNIFIED_FIELDS = [
    "experiment_id",
    "source_file",
    "engine",
    "engine_label",
    "model",
    "model_label",
    "workload",
    "metric",
    "statistic",
    "value",
    "stdev",
    "n",
    "unit",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def row(
    *,
    experiment_id: str,
    source_file: Path,
    engine: str,
    model: str,
    workload: str,
    metric: str,
    value: Any,
    unit: str,
    statistic: str = "mean",
    stdev: Any = "",
    n: Any = "",
    notes: str = "",
) -> dict[str, str]:
    return {
        "experiment_id": experiment_id,
        "source_file": str(source_file.relative_to(ROOT)),
        "engine": engine,
        "engine_label": ENGINE_LABEL.get(engine, engine),
        "model": model,
        "model_label": MODEL_LABEL.get(model, model),
        "workload": workload,
        "metric": metric,
        "statistic": statistic,
        "value": f"{float(value):.6f}" if isinstance(value, (float, int)) or str(value) else "",
        "stdev": f"{float(stdev):.6f}" if stdev not in ("", None) else "",
        "n": str(n) if n not in ("", None) else "",
        "unit": unit,
        "notes": notes,
    }


def add_model_size_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    source = FILES["model_size_mlx"]
    workload = "4 requests; input [64,256]; output [64,256]; seed 0; warmup 1"
    for item in rows:
        for metric, unit in [
            ("prefill_tps", "tok/s"),
            ("decode_tps", "tok/s"),
            ("wall_s", "s"),
            ("peak_memory_gb", "GB"),
        ]:
            mean_key = {
                "prefill_tps": "prefill_tps_mean",
                "decode_tps": "decode_tps_mean",
                "wall_s": "wall_s_mean",
                "peak_memory_gb": "peak_memory_gb_max",
            }[metric]
            stdev_key = {
                "prefill_tps": "prefill_tps_stdev",
                "decode_tps": "decode_tps_stdev",
                "wall_s": "wall_s_stdev",
                "peak_memory_gb": "",
            }[metric]
            out.append(
                row(
                    experiment_id="model_size_engine_comparison",
                    source_file=source,
                    engine=item["engine"],
                    model=item["model"],
                    workload=workload,
                    metric=metric,
                    value=item[mean_key],
                    stdev=item.get(stdev_key, ""),
                    n=item["n"],
                    unit=unit,
                )
            )
    return out


def add_llama_cpp_baseline_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    source = FILES["model_size_llama_cpp"]
    workload = "llama-bench -p 128 -n 128 -t 4 -ngl 99 -r 3"
    for item in rows:
        for metric, mean_key, stdev_key, unit in [
            ("prefill_tps", "prefill_tps_mean", "prefill_tps_stdev", "tok/s"),
            ("decode_tps", "decode_tps_mean", "decode_tps_stdev", "tok/s"),
            ("model_size_gb", "model_size_gb", "", "GB"),
            ("model_n_params", "model_n_params", "", "params"),
        ]:
            out.append(
                row(
                    experiment_id="llama_cpp_metal_baseline",
                    source_file=source,
                    engine=item["engine"],
                    model=item["model"],
                    workload=workload,
                    metric=metric,
                    value=item[mean_key],
                    stdev=item.get(stdev_key, ""),
                    n=item["n"],
                    unit=unit,
                    notes=f"backend={item['backend']}; build={item['build_number']} {item['build_commit']}",
                )
            )
    return out


def add_prompt_length_rows(paths: list[Path]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    by_key: dict[tuple[str, str, int], dict[str, dict[str, str]]] = {}
    for source in paths:
        for item in read_csv(source):
            prompt_len = int(item["requested_prompt_len"])
            workload = (
                f"prompt={prompt_len}; gen={item['requested_gen_len']}; "
                f"num_seqs={item['num_seqs']}; fixed-length decode"
            )
            for metric, mean_key, stdev_key, unit in [
                ("prefill_tps", "prefill_tps_mean", "prefill_tps_stdev", "tok/s"),
                ("decode_tps", "decode_tps_mean", "decode_tps_stdev", "tok/s"),
                ("wall_s", "wall_s_mean", "wall_s_stdev", "s"),
                ("peak_memory_gb", "peak_memory_gb_max", "", "GB"),
            ]:
                out.append(
                    row(
                        experiment_id="prompt_length_decode_scan",
                        source_file=source,
                        engine=item["engine"],
                        model=item["model"],
                        workload=workload,
                        metric=metric,
                        value=item[mean_key],
                        stdev=item.get(stdev_key, ""),
                        n=item["n"],
                        unit=unit,
                    )
                )
            by_key.setdefault((item["model"], item["requested_gen_len"], prompt_len), {})[
                item["engine"]
            ] = item

    for (model, gen_len, prompt_len), engines in by_key.items():
        if "mlx_lm" not in engines or "silicon_w1ref" not in engines:
            continue
        gap = float(engines["mlx_lm"]["decode_tps_mean"]) / float(
            engines["silicon_w1ref"]["decode_tps_mean"]
        )
        out.append(
            row(
                experiment_id="prompt_length_decode_scan",
                source_file=paths[0],
                engine="mlx_lm_over_silicon_w1ref",
                model=model,
                workload=f"prompt={prompt_len}; gen={gen_len}; num_seqs=2; fixed-length decode",
                metric="decode_gap",
                value=gap,
                unit="x",
                statistic="ratio",
                n="2",
                notes="mlx-lm decode_tps / SiliconLLM Stage 1 ref decode_tps",
            )
        )
    return out


def add_llama_cpp_mixed_workload_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    source = FILES["llama_cpp_mixed_workload"]
    for item in rows:
        workload = (
            f"llama-bench -pg {item['requested_prompt_len']},{item['requested_gen_len']}; "
            "mixed prompt+generation throughput"
        )
        for metric, mean_key, stdev_key, unit in [
            ("mixed_tps", "mixed_tps_mean", "mixed_tps_stdev", "tok/s"),
            ("avg_time_ns", "avg_ns", "stddev_ns", "ns"),
            ("model_size_gb", "model_size_gb", "", "GB"),
            ("model_n_params", "model_n_params", "", "params"),
        ]:
            out.append(
                row(
                    experiment_id="llama_cpp_mixed_workload_scan",
                    source_file=source,
                    engine=item["engine"],
                    model=item["model"],
                    workload=workload,
                    metric=metric,
                    value=item[mean_key],
                    stdev=item.get(stdev_key, ""),
                    n=item["n"],
                    unit=unit,
                    notes=f"backend={item['backend']}; build={item['build_number']} {item['build_commit']}",
                )
            )
    return out


def add_kv_cache_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    source = FILES["kv_cache_microbench"]
    path_to_engine = {
        "full_context_recompute": "synthetic_full_context",
        "cached_one_token": "synthetic_cached_decode",
    }
    for item in rows:
        engine = path_to_engine[item["path"]]
        workload = f"context={item['context_len']}; synthetic grouped attention"
        for metric, key, stat, unit in [
            ("latency_ms", "latency_ms_mean", "mean", "ms"),
            ("latency_ms", "latency_ms_median", "median", "ms"),
        ]:
            out.append(
                row(
                    experiment_id="kv_cache_microbenchmark",
                    source_file=source,
                    engine=engine,
                    model="synthetic-qwen2-0.5b-attn",
                    workload=workload,
                    metric=metric,
                    statistic=stat,
                    value=item[key],
                    stdev=item["latency_ms_stdev"] if stat == "mean" else "",
                    n=item["n"],
                    unit=unit,
                )
            )
        if item["path"] == "full_context_recompute":
            out.append(
                row(
                    experiment_id="kv_cache_microbenchmark",
                    source_file=source,
                    engine="synthetic_full_context_over_cached_decode",
                    model="synthetic-qwen2-0.5b-attn",
                    workload=workload,
                    metric="cached_speedup",
                    statistic="ratio",
                    value=item["cached_speedup_x"],
                    n=item["n"],
                    unit="x",
                    notes="full-context recompute median / cached decode median",
                )
            )
    return out


def add_decode_gap_scaling_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    source = FILES["decode_gap_scaling"]
    for item in rows:
        workload = "linear fit over prompt lengths 64, 128, 512; gen=64"
        for metric, key, unit in [
            ("intercept", "intercept_ms", "ms"),
            ("context_length_slope", "slope_ms_per_context_token", "ms/context_token"),
            ("r2", "r2", "unitless"),
            ("projected_decode_tps_prompt_1024", "predicted_decode_tps_at_prompt_1024", "tok/s"),
        ]:
            out.append(
                row(
                    experiment_id="decode_gap_scaling_fit",
                    source_file=source,
                    engine=item["engine"],
                    model=item["model"],
                    workload=workload,
                    metric=metric,
                    statistic="fit",
                    value=item[key],
                    unit=unit,
                    notes="Prompt-1024 values are projections, not measured results"
                    if metric == "projected_decode_tps_prompt_1024"
                    else "",
                )
            )
    return out


def build_unified_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(add_model_size_rows(read_csv(FILES["model_size_mlx"])))
    rows.extend(add_llama_cpp_baseline_rows(read_csv(FILES["model_size_llama_cpp"])))
    rows.extend(
        add_prompt_length_rows(
            [FILES["length_scan_0p5b"], FILES["length_scan_1p5b"]]
        )
    )
    rows.extend(add_llama_cpp_mixed_workload_rows(read_csv(FILES["llama_cpp_mixed_workload"])))
    rows.extend(add_kv_cache_rows(read_csv(FILES["kv_cache_microbench"])))
    rows.extend(add_decode_gap_scaling_rows(read_csv(FILES["decode_gap_scaling"])))
    return rows


def write_unified_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=UNIFIED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


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
        }
    )
    plt.rcParams.update(style)


def plot_coverage(rows: list[dict[str, str]]) -> None:
    apply_style()
    experiments = [
        "model_size_engine_comparison",
        "llama_cpp_metal_baseline",
        "prompt_length_decode_scan",
        "llama_cpp_mixed_workload_scan",
        "kv_cache_microbenchmark",
        "decode_gap_scaling_fit",
    ]
    artifact_groups = [
        "CSV",
        "PNG",
        "PDF/SVG",
        "Report",
    ]
    artifact_presence = {
        "model_size_engine_comparison": {"CSV", "PNG", "PDF/SVG", "Report"},
        "llama_cpp_metal_baseline": {"CSV", "PNG", "PDF/SVG", "Report"},
        "prompt_length_decode_scan": {"CSV", "PNG", "PDF/SVG", "Report"},
        "llama_cpp_mixed_workload_scan": {"CSV", "PNG", "PDF/SVG", "Report"},
        "kv_cache_microbenchmark": {"CSV", "PNG", "PDF/SVG", "Report"},
        "decode_gap_scaling_fit": {"CSV", "PNG", "PDF/SVG", "Report"},
    }
    data = np.array(
        [
            [1 if artifact in artifact_presence[exp] else 0 for artifact in artifact_groups]
            for exp in experiments
        ]
    )

    fig, ax = plt.subplots(figsize=(4.9, 2.9))
    fig.patch.set_facecolor("white")
    cmap = plt.matplotlib.colors.ListedColormap(["#ffffff", OKABE_ITO["bluish_green"]])
    ax.imshow(data, cmap=cmap, aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(artifact_groups)))
    ax.set_xticklabels(artifact_groups)
    ax.set_yticks(range(len(experiments)))
    ax.set_yticklabels([EXPERIMENT_LABEL[e] for e in experiments])
    ax.tick_params(length=0)

    for row_idx in range(data.shape[0]):
        for col_idx in range(data.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                "yes" if data[row_idx, col_idx] else "",
                ha="center",
                va="center",
                fontsize=7,
                color="white" if data[row_idx, col_idx] else "black",
            )

    ax.set_title("Phase 0 reproducibility coverage")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(artifact_groups), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(experiments), 1), minor=True)
    ax.grid(which="minor", color="#d9dde3", linewidth=0.6)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout(pad=0.8)
    save_publication_figure(
        fig,
        OUT_FIG,
        formats=["png", "pdf", "svg", "tiff"],
        dpi=600,
        transparent=False,
        facecolor="white",
    )


def print_summary(rows: list[dict[str, str]]) -> None:
    by_experiment: dict[str, int] = {}
    for item in rows:
        by_experiment[item["experiment_id"]] = by_experiment.get(item["experiment_id"], 0) + 1
    print("Unified Phase 0 summary")
    print("-" * 80)
    for experiment_id in sorted(by_experiment):
        print(f"{experiment_id:<34} {by_experiment[experiment_id]:>4} rows")
    print("-" * 80)
    print(f"CSV    -> {OUT_CSV}")
    print(f"Figure -> {OUT_FIG}.png/.pdf/.svg/.tiff")


def main() -> None:
    rows = build_unified_rows()
    write_unified_csv(rows)
    plot_coverage(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
