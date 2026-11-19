"""
Build a research-level long-format summary table.

This script does not run model inference. It combines the existing Phase 0
unified table with Phase 1 self-baseline checks and Phase 2 fallback checks, so
progress updates can point to one auditable CSV without renaming the Phase 0
manifest.

Outputs:
    benches/research_progress_unified_summary.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent
PHASE0_CSV = ROOT / "benches" / "phase0_unified_summary.csv"
SELF_BASELINE_CSV = ROOT / "benches" / "stage1_self_baseline_compare_summary.csv"
SELF_SANITY_CSV = ROOT / "benches" / "stage1_self_generation_sanity.csv"
SELF_1P5B_SMOKE_CSV = ROOT / "benches" / "stage1_self_baseline_compare_1p5b_smoke_summary.csv"
SELF_1P5B_SMOKE_SANITY_CSV = (
    ROOT / "benches" / "stage1_self_baseline_compare_1p5b_smoke_generation_sanity.csv"
)
KV_FALLBACK_CSV = ROOT / "benches" / "stage2_kv_fallback_benchmark_summary.csv"
KV_FALLBACK_1P5B_SMOKE_CSV = (
    ROOT / "benches" / "stage2_kv_fallback_benchmark_1p5b_smoke_summary.csv"
)
ATTRIBUTION_CSV = ROOT / "benches" / "stage2_attribution_benchmark_summary.csv"
ATTRIBUTION_RATIOS_CSV = ROOT / "benches" / "stage2_attribution_benchmark_ratios.csv"
STAGE2_PROFILE_CSV = ROOT / "benches" / "stage2_decode_component_profile_summary.csv"
STAGE3_SPEC_SELF_SUMMARY_CSV = (
    ROOT / "benches" / "stage3_speculative_accounting_summary.csv"
)
STAGE3_SPEC_SELF_RATIOS_CSV = (
    ROOT / "benches" / "stage3_speculative_accounting_ratios.csv"
)
STAGE3_SPEC_1P5B_SMOKE_SUMMARY_CSV = (
    ROOT / "benches" / "stage3_speculative_accounting_qwen2_1p5b_draft0p5b_smoke_summary.csv"
)
STAGE3_SPEC_1P5B_SMOKE_RATIOS_CSV = (
    ROOT / "benches" / "stage3_speculative_accounting_qwen2_1p5b_draft0p5b_smoke_ratios.csv"
)
STAGE3_SPEC_1P5B_N3_SUMMARY_CSV = (
    ROOT / "benches" / "stage3_speculative_accounting_qwen2_1p5b_draft0p5b_summary.csv"
)
STAGE3_SPEC_1P5B_N3_RATIOS_CSV = (
    ROOT / "benches" / "stage3_speculative_accounting_qwen2_1p5b_draft0p5b_ratios.csv"
)
STAGE3_SPEC_LENGTH_SCAN_SUMMARY_CSV = (
    ROOT / "benches" / "stage3_speculative_length_scan_summary.csv"
)
STAGE3_SPEC_LENGTH_SCAN_RATIOS_CSV = (
    ROOT / "benches" / "stage3_speculative_length_scan_ratios.csv"
)
OUT_CSV = ROOT / "benches" / "research_progress_unified_summary.csv"

BASE_FIELDS = [
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
FIELDS = ["phase_id", *BASE_FIELDS]

MODEL_LABEL = {
    "qwen2-0.5b": "Qwen2 0.5B",
    "qwen2-1.5b": "Qwen2 1.5B",
    "qwen2-7b": "Qwen2 7B",
    "synthetic-qwen2-0.5b-attn": "Synthetic Qwen2 0.5B attention",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def fmt(value: Any) -> str:
    if value in ("", None):
        return ""
    return f"{float(value):.6f}"


def research_row(
    *,
    phase_id: str,
    experiment_id: str,
    source_file: Path,
    engine: str,
    engine_label: str,
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
        "phase_id": phase_id,
        "experiment_id": experiment_id,
        "source_file": str(source_file.relative_to(ROOT)),
        "engine": engine,
        "engine_label": engine_label,
        "model": model,
        "model_label": MODEL_LABEL.get(model, model),
        "workload": workload,
        "metric": metric,
        "statistic": statistic,
        "value": fmt(value),
        "stdev": fmt(stdev),
        "n": str(n) if n not in ("", None) else "",
        "unit": unit,
        "notes": notes,
    }


def add_phase0_rows() -> list[dict[str, str]]:
    rows = []
    for item in read_csv(PHASE0_CSV):
        rows.append({"phase_id": "phase0", **{field: item[field] for field in BASE_FIELDS}})
    return rows


def add_self_baseline_rows(
    *,
    summary_path: Path,
    sanity_path: Path,
    experiment_id: str,
    workload: str,
    extra_notes: str = "",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    summary = read_csv(summary_path)
    metric_specs = [
        ("prefill_tps", "prefill_tps_mean", "prefill_tps_stdev", "tok/s"),
        ("decode_tps", "decode_tps_mean", "decode_tps_stdev", "tok/s"),
        ("wall_s", "wall_s_mean", "wall_s_stdev", "s"),
        ("peak_memory_gb", "peak_memory_gb_max", "", "GB"),
    ]
    for item in summary:
        for metric, mean_key, stdev_key, unit in metric_specs:
            rows.append(
                research_row(
                    phase_id="phase1",
                    experiment_id=experiment_id,
                    source_file=summary_path,
                    engine=item["engine"],
                    engine_label=item["engine_label"],
                    model=item["model"],
                    workload=workload,
                    metric=metric,
                    value=item[mean_key],
                    stdev=item.get(stdev_key, ""),
                    n=item["n"],
                    unit=unit,
                    notes=extra_notes,
                )
            )

    by_engine = {item["engine"]: item for item in summary}
    if {"tiny_llm_w1self", "tiny_llm_ref_w1"}.issubset(by_engine):
        self_row = by_engine["tiny_llm_w1self"]
        ref_row = by_engine["tiny_llm_ref_w1"]
        ratio_specs = [
            ("prefill_tps_ratio", "prefill_tps_mean", "Self/reference prefill throughput"),
            ("decode_tps_ratio", "decode_tps_mean", "Self/reference decode throughput"),
            ("wall_s_ratio", "wall_s_mean", "Self/reference wall time"),
        ]
        for metric, key, note in ratio_specs:
            rows.append(
                research_row(
                    phase_id="phase1",
                    experiment_id=experiment_id,
                    source_file=summary_path,
                    engine="tiny_llm_w1self_over_ref",
                    engine_label="Self Stage 1 / Reference Stage 1",
                    model=self_row["model"],
                    workload=workload,
                    metric=metric,
                    statistic="ratio",
                    value=float(self_row[key]) / float(ref_row[key]),
                    unit="x",
                    n=self_row["n"],
                    notes=f"{note}; {extra_notes}" if extra_notes else note,
                )
            )

    sanity = read_csv(sanity_path)
    if len(sanity) >= 2:
        token_match = sanity[0]["generated_token_ids"] == sanity[1]["generated_token_ids"]
        rows.append(
            research_row(
                phase_id="phase1",
                experiment_id=experiment_id,
                source_file=sanity_path,
                engine="tiny_llm_w1self_vs_ref",
                engine_label="Self Stage 1 vs Reference Stage 1",
                model=sanity[0]["model"],
                workload=(
                    f"prompt={sanity[0]['prompt']}; max_new_tokens="
                    f"{sanity[0]['max_new_tokens']}; greedy decode"
                ),
                metric="greedy_token_match",
                statistic="boolean",
                value=1 if token_match else 0,
                unit="boolean",
                n="1",
                notes=(
                    "1 means both engines produced identical generated token IDs"
                    + (f"; {extra_notes}" if extra_notes else "")
                ),
            )
        )
    return rows


def add_kv_fallback_rows(
    *,
    summary_path: Path,
    experiment_id: str,
    workload: str,
    extra_notes: str = "",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    summary = read_csv(summary_path)
    metric_specs = [
        ("prefill_tps", "prefill_tps_mean", "prefill_tps_stdev", "tok/s"),
        ("decode_tps", "decode_tps_mean", "decode_tps_stdev", "tok/s"),
        ("wall_s", "wall_s_mean", "wall_s_stdev", "s"),
        ("peak_memory_gb", "peak_memory_gb_max", "", "GB"),
    ]
    for item in summary:
        for metric, mean_key, stdev_key, unit in metric_specs:
            rows.append(
                research_row(
                    phase_id="phase2",
                    experiment_id=experiment_id,
                    source_file=summary_path,
                    engine=item["engine"],
                    engine_label=item["engine_label"],
                    model=item["model"],
                    workload=workload,
                    metric=metric,
                    value=item[mean_key],
                    stdev=item.get(stdev_key, ""),
                    n=item["n"],
                    unit=unit,
                    notes=extra_notes,
                )
            )

    by_engine = {item["engine"]: item for item in summary}
    if {"week1_full_context", "week2_kv_fallback"}.issubset(by_engine):
        week1 = by_engine["week1_full_context"]
        week2 = by_engine["week2_kv_fallback"]
        ratio_specs = [
            ("prefill_tps_ratio", "prefill_tps_mean", "Stage 2 / Stage 1 prefill throughput"),
            ("decode_tps_ratio", "decode_tps_mean", "Stage 2 / Stage 1 decode throughput"),
            ("wall_s_ratio", "wall_s_mean", "Stage 2 / Stage 1 wall time"),
            ("peak_memory_gb_ratio", "peak_memory_gb_max", "Stage 2 / Stage 1 peak memory"),
        ]
        for metric, key, note in ratio_specs:
            rows.append(
                research_row(
                    phase_id="phase2",
                    experiment_id=experiment_id,
                    source_file=summary_path,
                    engine="week2_kv_fallback_over_week1",
                    engine_label="Stage 2 KV fallback / Stage 1 full-context",
                    model=week1["model"],
                    workload=workload,
                    metric=metric,
                    statistic="ratio",
                    value=float(week2[key]) / float(week1[key]),
                    unit="x",
                    n=week2["n"],
                    notes=f"{note}; {extra_notes}" if extra_notes else note,
                )
            )
    return rows


def add_phase1_self_baseline_rows() -> list[dict[str, str]]:
    rows = add_self_baseline_rows(
        summary_path=SELF_BASELINE_CSV,
        sanity_path=SELF_SANITY_CSV,
        experiment_id="phase1_stage1_self_baseline_compare",
        workload=(
            "4 requests; input [64,256]; output [64,256]; seed 0; "
            "model loading not counted"
        ),
    )
    if SELF_1P5B_SMOKE_CSV.exists() and SELF_1P5B_SMOKE_SANITY_CSV.exists():
        rows.extend(
            add_self_baseline_rows(
                summary_path=SELF_1P5B_SMOKE_CSV,
                sanity_path=SELF_1P5B_SMOKE_SANITY_CSV,
                experiment_id="phase1_stage1_self_baseline_1p5b_smoke",
                workload=(
                    "1 request; input [64,256]; output [64,256]; seed 0; "
                    "model loading not counted; narrow smoke benchmark"
                ),
                extra_notes="N=2 narrow smoke benchmark, not a full N=3 sweep",
            )
        )
    return rows


def add_phase2_kv_fallback_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if KV_FALLBACK_CSV.exists():
        rows.extend(
            add_kv_fallback_rows(
                summary_path=KV_FALLBACK_CSV,
                experiment_id="phase2_stage2_kv_fallback_benchmark_0p5b",
                workload=(
                    "2 requests; input [64,256]; output [64,256]; seed 0; "
                    "model loading not counted; Stage 2 uses MLX built-in quantized matmul fallback"
                ),
                extra_notes="Stage 2 path is KV-cache fallback, not custom C++/Metal kernel",
            )
        )
    if KV_FALLBACK_1P5B_SMOKE_CSV.exists():
        rows.extend(
            add_kv_fallback_rows(
                summary_path=KV_FALLBACK_1P5B_SMOKE_CSV,
                experiment_id="phase2_stage2_kv_fallback_1p5b_smoke",
                workload=(
                    "1 request; input [64,256]; output [64,256]; seed 0; "
                    "model loading not counted; narrow smoke benchmark; "
                    "Stage 2 uses MLX built-in quantized matmul fallback"
                ),
                extra_notes=(
                    "N=2 narrow smoke benchmark; Stage 2 path is KV-cache fallback, "
                    "not custom C++/Metal kernel"
                ),
            )
        )
    return rows


def add_stage2_attribution_rows() -> list[dict[str, str]]:
    if not ATTRIBUTION_CSV.exists() or not ATTRIBUTION_RATIOS_CSV.exists():
        return []

    rows: list[dict[str, str]] = []
    workload = (
        "2 requests; input [64,256]; output [64,256]; seed 0; model loading not counted; "
        "four-arm attribution benchmark"
    )
    notes = (
        "Stage 2 full-context uses fresh per-call cache; Stage 2 KV-cache uses "
        "persistent decode cache; mlx-lm accounting follows stream_generate"
    )
    metric_specs = [
        ("prefill_tps", "prefill_tps_mean", "prefill_tps_stdev", "tok/s"),
        ("decode_tps", "decode_tps_mean", "decode_tps_stdev", "tok/s"),
        ("wall_s", "wall_s_mean", "wall_s_stdev", "s"),
        ("peak_memory_gb", "peak_memory_gb_max", "", "GB"),
    ]
    for item in read_csv(ATTRIBUTION_CSV):
        for metric, mean_key, stdev_key, unit in metric_specs:
            rows.append(
                research_row(
                    phase_id="phase2",
                    experiment_id="phase2_stage2_attribution_benchmark_0p5b",
                    source_file=ATTRIBUTION_CSV,
                    engine=item["engine"],
                    engine_label=item["engine_label"],
                    model=item["model"],
                    workload=workload,
                    metric=metric,
                    value=item[mean_key],
                    stdev=item.get(stdev_key, ""),
                    n=item["n"],
                    unit=unit,
                    notes=notes,
                )
            )

    ratio_metric_specs = [
        ("prefill_tps_ratio", "prefill_tps_ratio", "prefill throughput ratio"),
        ("decode_tps_ratio", "decode_tps_ratio", "decode throughput ratio"),
        ("wall_s_ratio", "wall_s_ratio", "wall-time ratio"),
        ("peak_memory_gb_ratio", "peak_memory_gb_ratio", "peak-memory ratio"),
    ]
    summary_by_engine = {item["engine"]: item for item in read_csv(ATTRIBUTION_CSV)}
    for item in read_csv(ATTRIBUTION_RATIOS_CSV):
        numerator = item["numerator_engine"]
        n = summary_by_engine.get(numerator, {}).get("n", "")
        for metric, value_key, note in ratio_metric_specs:
            if item.get(value_key, "") == "":
                continue
            rows.append(
                research_row(
                    phase_id="phase2",
                    experiment_id="phase2_stage2_attribution_benchmark_0p5b",
                    source_file=ATTRIBUTION_RATIOS_CSV,
                    engine=item["ratio_id"],
                    engine_label=item["ratio_label"],
                    model=item["model"],
                    workload=workload,
                    metric=metric,
                    statistic="ratio",
                    value=item[value_key],
                    unit="x",
                    n=n,
                    notes=f"{note}; {notes}",
                )
            )
    return rows


def add_stage2_decode_profile_rows() -> list[dict[str, str]]:
    if not STAGE2_PROFILE_CSV.exists():
        return []

    rows: list[dict[str, str]] = []
    for item in read_csv(STAGE2_PROFILE_CSV):
        workload = (
            f"one cached decode step after untimed prefill; prompt_len={item['prompt_len']}; "
            f"decode_offset={item['decode_offset']}; Qwen2 0.5B; synchronized MLX timings"
        )
        rows.append(
            research_row(
                phase_id="phase2",
                experiment_id="phase2_stage2_decode_component_profile_0p5b",
                source_file=STAGE2_PROFILE_CSV,
                engine=item["component"],
                engine_label=item["component"].replace("_", " "),
                model=item["model"],
                workload=workload,
                metric="decode_component_time_ms",
                value=item["mean_ms"],
                stdev=item["stdev_ms"],
                n=item["n"],
                unit="ms",
                notes="Stage 2 KV-cache decode component timing",
            )
        )
        rows.append(
            research_row(
                phase_id="phase2",
                experiment_id="phase2_stage2_decode_component_profile_0p5b",
                source_file=STAGE2_PROFILE_CSV,
                engine=item["component"],
                engine_label=item["component"].replace("_", " "),
                model=item["model"],
                workload=workload,
                metric="decode_component_share_pct",
                value=item["share_pct"],
                n=item["n"],
                unit="%",
                notes="Share of synchronized cached decode component time",
            )
        )
    return rows


def add_stage3_speculative_rows(
    *,
    summary_path: Path,
    ratios_path: Path,
    experiment_id: str,
    workload_note: str,
    extra_notes: str = "",
) -> list[dict[str, str]]:
    if not summary_path.exists() or not ratios_path.exists():
        return []

    rows: list[dict[str, str]] = []
    metric_specs = [
        ("decode_tps", "decode_tps_mean", "decode_tps_stdev", "tok/s"),
        ("wall_tokens_per_s", "wall_tokens_per_s_mean", "wall_tokens_per_s_stdev", "tok/s"),
        ("draft_time_s", "draft_time_s_mean", "draft_time_s_stdev", "s"),
        ("target_verify_time_s", "target_verify_time_s_mean", "target_verify_time_s_stdev", "s"),
        ("misc_time_s", "misc_time_s_mean", "misc_time_s_stdev", "s"),
        ("draft_time_share", "draft_time_share_mean", "draft_time_share_stdev", "fraction"),
        (
            "target_verify_time_share",
            "target_verify_time_share_mean",
            "target_verify_time_share_stdev",
            "fraction",
        ),
        ("misc_time_share", "misc_time_share_mean", "misc_time_share_stdev", "fraction"),
        (
            "target_calls_per_decode_token",
            "target_calls_per_decode_token_mean",
            "target_calls_per_decode_token_stdev",
            "calls/token",
        ),
        (
            "draft_calls_per_decode_token",
            "draft_calls_per_decode_token_mean",
            "draft_calls_per_decode_token_stdev",
            "calls/token",
        ),
        ("acceptance_rate", "acceptance_rate_mean", "acceptance_rate_stdev", "fraction"),
        ("wall_s", "wall_s_mean", "wall_s_stdev", "s"),
        ("peak_memory_gb", "peak_memory_gb_mean", "peak_memory_gb_stdev", "GB"),
    ]
    notes = (
        "Greedy speculative decoding accounting; exact-match sanity compares "
        "speculative output against target greedy output"
    )
    if extra_notes:
        notes = f"{notes}; {extra_notes}"

    for item in read_csv(summary_path):
        draft_length = item["draft_length"]
        workload = (
            f"{workload_note}; draft_length={draft_length}; "
            f"generated_tokens={item['generated_tokens']}; decode_tokens={item['decode_tokens']}"
        )
        engine = (
            "stage2_kv_cache_greedy"
            if item["mode"] == "baseline"
            else f"stage3_speculative_greedy_draft{draft_length}"
        )
        engine_label = (
            item["mode_label"]
            if item["mode"] == "baseline"
            else f"{item['mode_label']} (draft={draft_length})"
        )
        for metric, mean_key, stdev_key, unit in metric_specs:
            if item.get(mean_key, "") == "":
                continue
            rows.append(
                research_row(
                    phase_id="phase3",
                    experiment_id=experiment_id,
                    source_file=summary_path,
                    engine=engine,
                    engine_label=engine_label,
                    model=item["target_model"],
                    workload=workload,
                    metric=metric,
                    value=item[mean_key],
                    stdev=item.get(stdev_key, ""),
                    n=item["n"],
                    unit=unit,
                    notes=notes,
                )
            )
        rows.append(
            research_row(
                phase_id="phase3",
                experiment_id=experiment_id,
                source_file=summary_path,
                engine=engine,
                engine_label=engine_label,
                model=item["target_model"],
                workload=workload,
                metric="exact_match_baseline",
                statistic="boolean",
                value=item["matches_baseline_all"],
                n=item["n"],
                unit="boolean",
                notes=notes,
            )
        )

    ratio_metric_specs = [
        ("decode_tps_ratio", "decode_tps_ratio", "speculative/baseline decode throughput"),
        ("wall_s_ratio", "wall_s_ratio", "speculative/baseline wall time"),
        (
            "target_calls_per_decode_token_ratio",
            "target_calls_per_decode_token_ratio",
            "speculative/baseline target calls per decode token",
        ),
        (
            "wall_tokens_per_s_ratio",
            "wall_tokens_per_s_ratio",
            "speculative/baseline wall tokens per second",
        ),
    ]
    for item in read_csv(ratios_path):
        draft_length = item["draft_length"]
        workload = f"{workload_note}; draft_length={draft_length}; ratio vs Stage 2 KV-cache greedy"
        for metric, value_key, note in ratio_metric_specs:
            rows.append(
                research_row(
                    phase_id="phase3",
                    experiment_id=experiment_id,
                    source_file=ratios_path,
                    engine=f"stage3_speculative_draft{draft_length}_over_stage2_kv",
                    engine_label=f"Stage 3 speculative draft={draft_length} / Stage 2 KV-cache",
                    model=item["target_model"],
                    workload=workload,
                    metric=metric,
                    statistic="ratio",
                    value=item[value_key],
                    unit="x",
                    n="",
                    notes=f"{note}; {notes}",
                )
            )
    return rows


def add_phase3_speculative_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        add_stage3_speculative_rows(
            summary_path=STAGE3_SPEC_SELF_SUMMARY_CSV,
            ratios_path=STAGE3_SPEC_SELF_RATIOS_CSV,
            experiment_id="phase3_stage3_speculative_accounting_0p5b_selfpair",
            workload_note=(
                "Qwen2 0.5B target and Qwen2 0.5B draft self-pair; "
                "2 requests; prompt [64,96]; max_new_tokens=64; seed 0; N=2"
            ),
            extra_notes=(
                "Self-pair is an algorithmic upper-bound sanity case, not a "
                "speedup claim because draft and target have equal cost"
            ),
        )
    )
    rows.extend(
        add_stage3_speculative_rows(
            summary_path=STAGE3_SPEC_1P5B_N3_SUMMARY_CSV,
            ratios_path=STAGE3_SPEC_1P5B_N3_RATIOS_CSV,
            experiment_id="phase3_stage3_speculative_accounting_1p5b_target_0p5b_draft_n3",
            workload_note=(
                "Qwen2 1.5B target with Qwen2 0.5B draft; 2 requests; "
                "prompt [64,96]; max_new_tokens=64; seed 0; N=3"
            ),
            extra_notes="Formal Stage 3 N=3 sweep over draft lengths 2, 4, and 8",
        )
    )
    rows.extend(
        add_stage3_speculative_rows(
            summary_path=STAGE3_SPEC_1P5B_SMOKE_SUMMARY_CSV,
            ratios_path=STAGE3_SPEC_1P5B_SMOKE_RATIOS_CSV,
            experiment_id="phase3_stage3_speculative_accounting_1p5b_target_0p5b_draft_smoke",
            workload_note=(
                "Qwen2 1.5B target with Qwen2 0.5B draft; 1 request; "
                "prompt_len=64; max_new_tokens=32; seed 0; N=1 smoke"
            ),
            extra_notes="N=1 narrow smoke benchmark, not a full N=3 sweep",
        )
    )
    rows.extend(add_stage3_length_scan_rows())
    return rows


def add_stage3_length_scan_rows() -> list[dict[str, str]]:
    if not STAGE3_SPEC_LENGTH_SCAN_SUMMARY_CSV.exists() or not STAGE3_SPEC_LENGTH_SCAN_RATIOS_CSV.exists():
        return []

    rows: list[dict[str, str]] = []
    metric_specs = [
        ("decode_tps", "decode_tps_mean", "decode_tps_stdev", "tok/s"),
        ("wall_tokens_per_s", "wall_tokens_per_s_mean", "wall_tokens_per_s_stdev", "tok/s"),
        (
            "target_calls_per_decode_token",
            "target_calls_per_decode_token_mean",
            "target_calls_per_decode_token_stdev",
            "calls/token",
        ),
        (
            "draft_calls_per_decode_token",
            "draft_calls_per_decode_token_mean",
            "draft_calls_per_decode_token_stdev",
            "calls/token",
        ),
        ("acceptance_rate", "acceptance_rate_mean", "acceptance_rate_stdev", "fraction"),
        ("draft_time_share", "draft_time_share_mean", "draft_time_share_stdev", "fraction"),
        (
            "target_verify_time_share",
            "target_verify_time_share_mean",
            "target_verify_time_share_stdev",
            "fraction",
        ),
        ("misc_time_share", "misc_time_share_mean", "misc_time_share_stdev", "fraction"),
        ("wall_s", "wall_s_mean", "wall_s_stdev", "s"),
    ]
    notes = (
        "Stage 3 output-length scan; greedy speculative decoding; exact-match "
        "sanity compares speculative output against target greedy output"
    )
    for item in read_csv(STAGE3_SPEC_LENGTH_SCAN_SUMMARY_CSV):
        draft_length = item["draft_length"]
        output_len = item["output_len"]
        workload = (
            f"Qwen2 1.5B target with Qwen2 0.5B draft; 2 requests; "
            f"prompt [64,96]; max_new_tokens={output_len}; seed 0; N=2; "
            f"draft_length={draft_length}"
        )
        engine = (
            f"stage2_kv_cache_greedy_out{output_len}"
            if item["mode"] == "baseline"
            else f"stage3_speculative_out{output_len}_draft{draft_length}"
        )
        engine_label = (
            f"{item['mode_label']} (output={output_len})"
            if item["mode"] == "baseline"
            else f"{item['mode_label']} (output={output_len}, draft={draft_length})"
        )
        for metric, mean_key, stdev_key, unit in metric_specs:
            if item.get(mean_key, "") == "":
                continue
            rows.append(
                research_row(
                    phase_id="phase3",
                    experiment_id="phase3_stage3_speculative_output_length_scan",
                    source_file=STAGE3_SPEC_LENGTH_SCAN_SUMMARY_CSV,
                    engine=engine,
                    engine_label=engine_label,
                    model=item["target_model"],
                    workload=workload,
                    metric=metric,
                    value=item[mean_key],
                    stdev=item.get(stdev_key, ""),
                    n=item["n"],
                    unit=unit,
                    notes=notes,
                )
            )
        rows.append(
            research_row(
                phase_id="phase3",
                experiment_id="phase3_stage3_speculative_output_length_scan",
                source_file=STAGE3_SPEC_LENGTH_SCAN_SUMMARY_CSV,
                engine=engine,
                engine_label=engine_label,
                model=item["target_model"],
                workload=workload,
                metric="exact_match_baseline",
                statistic="boolean",
                value=item["matches_baseline_all"],
                n=item["n"],
                unit="boolean",
                notes=notes,
            )
        )

    ratio_metric_specs = [
        ("decode_tps_ratio", "decode_tps_ratio", "speculative/baseline decode throughput"),
        ("wall_s_ratio", "wall_s_ratio", "speculative/baseline wall time"),
        (
            "target_calls_per_decode_token_ratio",
            "target_calls_per_decode_token_ratio",
            "speculative/baseline target calls per decode token",
        ),
        (
            "wall_tokens_per_s_ratio",
            "wall_tokens_per_s_ratio",
            "speculative/baseline wall tokens per second",
        ),
    ]
    for item in read_csv(STAGE3_SPEC_LENGTH_SCAN_RATIOS_CSV):
        draft_length = item["draft_length"]
        output_len = item["output_len"]
        workload = (
            f"Qwen2 1.5B target with Qwen2 0.5B draft; "
            f"max_new_tokens={output_len}; draft_length={draft_length}; ratio vs Stage 2 KV-cache greedy"
        )
        for metric, value_key, note in ratio_metric_specs:
            rows.append(
                research_row(
                    phase_id="phase3",
                    experiment_id="phase3_stage3_speculative_output_length_scan",
                    source_file=STAGE3_SPEC_LENGTH_SCAN_RATIOS_CSV,
                    engine=f"stage3_speculative_out{output_len}_draft{draft_length}_over_stage2_kv",
                    engine_label=(
                        f"Stage 3 speculative output={output_len}, draft={draft_length} / "
                        "Stage 2 KV-cache"
                    ),
                    model=item["target_model"],
                    workload=workload,
                    metric=metric,
                    statistic="ratio",
                    value=item[value_key],
                    unit="x",
                    n="",
                    notes=f"{note}; {notes}",
                )
            )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict[str, str]]) -> None:
    counts: dict[tuple[str, str], int] = {}
    for item in rows:
        key = (item["phase_id"], item["experiment_id"])
        counts[key] = counts.get(key, 0) + 1

    print("Research progress unified summary")
    print("-" * 80)
    for (phase_id, experiment_id), count in sorted(counts.items()):
        print(f"{phase_id:<8} {experiment_id:<42} {count:>4} rows")
    print("-" * 80)
    print(f"CSV -> {OUT_CSV}")


def main() -> None:
    rows = [
        *add_phase0_rows(),
        *add_phase1_self_baseline_rows(),
        *add_phase2_kv_fallback_rows(),
        *add_stage2_attribution_rows(),
        *add_stage2_decode_profile_rows(),
        *add_phase3_speculative_rows(),
    ]
    write_csv(rows)
    print_summary(rows)


if __name__ == "__main__":
    main()
