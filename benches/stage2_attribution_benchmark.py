"""
Four-arm Stage 2 attribution benchmark.

This benchmark separates the total Stage 1 -> Stage 2 gain into:

1. Stage 1 full-context reference baseline
2. Stage 2 full-context self path without persistent decode KV cache
3. Stage 2 KV-cache fallback path
4. mlx-lm production reference

Default run:
    pdm run python benches/stage2_attribution_benchmark.py

Outputs:
    benches/stage2_attribution_benchmark_results.csv
    benches/stage2_attribution_benchmark_summary.csv
    benches/stage2_attribution_benchmark_ratios.csv
    benches/stage2_attribution_benchmark.png/.pdf/.svg/.tiff
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter

import matplotlib.pyplot as plt
import mlx.core as mx
from mlx_lm import load as mlx_lm_load
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from color_palettes import OKABE_ITO  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from style_presets import get_base_style  # noqa: E402
from tiny_llm import models as self_models  # noqa: E402
from tiny_llm.kv_cache import TinyKvFullCache  # noqa: E402
from tiny_llm_ref import models as ref_models  # noqa: E402


ENGINE_ORDER = [
    "stage1_ref_full_context",
    "stage2_full_context",
    "stage2_kv_cache",
    "mlx_lm",
]

ENGINE_LABEL = {
    "stage1_ref_full_context": "Stage 1 full-context ref",
    "stage2_full_context": "Stage 2 full-context",
    "stage2_kv_cache": "Stage 2 KV-cache",
    "mlx_lm": "mlx-lm",
}

ENGINE_COLOR = {
    "stage1_ref_full_context": OKABE_ITO["vermillion"],
    "stage2_full_context": OKABE_ITO["orange"],
    "stage2_kv_cache": OKABE_ITO["bluish_green"],
    "mlx_lm": OKABE_ITO["blue"],
}

RATIO_SPECS = [
    (
        "stage2_full_context_over_stage1_ref",
        "Stage 2 full-context / Stage 1 ref",
        "stage2_full_context",
        "stage1_ref_full_context",
    ),
    (
        "stage2_kv_cache_over_stage2_full_context",
        "Stage 2 KV-cache / Stage 2 full-context",
        "stage2_kv_cache",
        "stage2_full_context",
    ),
    (
        "mlx_lm_over_stage2_kv_cache",
        "mlx-lm / Stage 2 KV-cache",
        "mlx_lm",
        "stage2_kv_cache",
    ),
    (
        "stage2_kv_cache_over_stage1_ref",
        "Stage 2 KV-cache / Stage 1 ref",
        "stage2_kv_cache",
        "stage1_ref_full_context",
    ),
]


@dataclass
class BenchRequest:
    prompt_token_ids: list[int]
    max_new_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Stage 2 attribution benchmark.")
    parser.add_argument("--model", default="qwen2-0.5b")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--output-stem",
        default="stage2_attribution_benchmark",
        help="Output file stem under benches/ without suffixes.",
    )
    parser.add_argument(
        "--engines",
        nargs="+",
        choices=ENGINE_ORDER,
        default=ENGINE_ORDER,
        help="Subset of attribution arms to run.",
    )
    return parser.parse_args()


def output_paths(stem: str) -> tuple[Path, Path, Path, Path]:
    out_stem = Path(stem)
    if out_stem.name != stem or out_stem.suffix:
        raise ValueError("--output-stem must be a bare filename stem without directories or suffix")
    return (
        ROOT / "benches" / f"{stem}_results.csv",
        ROOT / "benches" / f"{stem}_summary.csv",
        ROOT / "benches" / f"{stem}_ratios.csv",
        ROOT / "benches" / stem,
    )


def build_requests(
    *,
    num_seqs: int,
    vocab_size: int,
    eos_token_id: int,
    seed: int,
) -> list[BenchRequest]:
    rng = Random(seed)
    low = 256 if vocab_size > 512 else 0
    high = vocab_size - 1
    requests: list[BenchRequest] = []
    for _ in range(num_seqs):
        prompt_len = rng.randint(64, 256)
        output_len = rng.randint(64, 256)
        ids: list[int] = []
        for _ in range(prompt_len):
            token_id = rng.randint(low, high)
            if token_id == eos_token_id:
                token_id = token_id - 1 if token_id > low else token_id + 1
            ids.append(token_id)
        requests.append(BenchRequest(ids, output_len))
    return requests


def reset_peak_memory() -> None:
    for ns in (getattr(mx, "metal", None), getattr(mx, "memory", None)):
        if ns is None:
            continue
        for fn_name in ("reset_peak_memory", "reset_peak_mem"):
            fn = getattr(ns, fn_name, None)
            if callable(fn):
                try:
                    fn()
                    return
                except Exception:
                    pass


def get_peak_memory_gb() -> float | None:
    for ns in (getattr(mx, "metal", None), getattr(mx, "memory", None)):
        if ns is None:
            continue
        for fn_name in ("get_peak_memory", "peak_memory"):
            fn = getattr(ns, fn_name, None)
            if callable(fn):
                try:
                    return float(fn()) / (1024**3)
                except Exception:
                    pass
    return None


def clear_mlx_cache() -> None:
    gc.collect()
    for ns in (getattr(mx, "metal", None), getattr(mx, "memory", None)):
        if ns is None:
            continue
        fn = getattr(ns, "clear_cache", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass


def sample_full_context(model, context: mx.array) -> mx.array:
    logits = model(context[None, :])[:, -1, :]
    return mx.argmax(logits, axis=-1)


def sample_stage2_fresh_cache(model, context: mx.array) -> mx.array:
    cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
    logits = model(context[None, :], 0, cache)[:, -1, :]
    return mx.argmax(logits, axis=-1)


def sample_stage2_kv_cache(
    model,
    tokens: mx.array,
    offset: int,
    cache: list[TinyKvFullCache],
) -> mx.array:
    logits = model(tokens[None, :], offset, cache)[:, -1, :]
    return mx.argmax(logits, axis=-1)


def run_one_request_full_context(
    model,
    request: BenchRequest,
    *,
    stage2_fresh_cache: bool,
) -> tuple[int, float, float]:
    context = mx.array(request.prompt_token_ids, dtype=mx.int32)
    sample = sample_stage2_fresh_cache if stage2_fresh_cache else sample_full_context

    t0 = perf_counter()
    token = sample(model, context)
    mx.eval(token)
    prefill_time = perf_counter() - t0

    generated_tokens = 1
    decode_time = 0.0
    for _ in range(request.max_new_tokens - 1):
        t1 = perf_counter()
        context = mx.concat([context, token])
        token = sample(model, context)
        mx.eval(token)
        decode_time += perf_counter() - t1
        generated_tokens += 1
    return generated_tokens, prefill_time, decode_time


def run_one_request_stage2_kv_cache(model, request: BenchRequest) -> tuple[int, float, float]:
    cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
    prompt = mx.array(request.prompt_token_ids, dtype=mx.int32)
    t0 = perf_counter()
    token = sample_stage2_kv_cache(model, prompt, 0, cache)
    mx.eval(token)
    prefill_time = perf_counter() - t0

    offset = len(request.prompt_token_ids)
    generated_tokens = 1
    decode_time = 0.0
    for _ in range(request.max_new_tokens - 1):
        t1 = perf_counter()
        token = sample_stage2_kv_cache(model, token.astype(mx.int32), offset, cache)
        mx.eval(token)
        decode_time += perf_counter() - t1
        offset += 1
        generated_tokens += 1
    return generated_tokens, prefill_time, decode_time


def aggregate_timing(
    requests: list[BenchRequest],
    runner,
) -> dict[str, float | int | None]:
    reset_peak_memory()
    runner(requests[0])

    total_prompt = 0
    total_generated = 0
    total_prefill_time = 0.0
    total_decode_time = 0.0
    t_all = perf_counter()
    for request in requests:
        generated, prefill_time, decode_time = runner(request)
        total_prompt += len(request.prompt_token_ids)
        total_generated += generated
        total_prefill_time += prefill_time
        total_decode_time += decode_time
    wall_s = perf_counter() - t_all
    decode_tokens = total_generated - len(requests)
    return {
        "prompt_tokens": total_prompt,
        "gen_tokens": total_generated,
        "decode_tokens": decode_tokens,
        "prefill_tps": total_prompt / total_prefill_time if total_prefill_time else 0.0,
        "decode_tps": decode_tokens / total_decode_time if total_decode_time else 0.0,
        "wall_s": wall_s,
        "peak_memory_gb": get_peak_memory_gb(),
    }


def measure_mlx_lm(model, tokenizer, requests: list[BenchRequest]) -> dict[str, float | int | None]:
    sampler = make_sampler(0.0)
    reset_peak_memory()

    warm_prompt = tokenizer.decode(requests[0].prompt_token_ids[:32])
    for _ in stream_generate(model, tokenizer, prompt=warm_prompt, max_tokens=8, sampler=sampler):
        pass

    total_prompt = 0
    total_gen = 0
    weighted_prefill_tps = 0.0
    weighted_decode_tps = 0.0
    peak_mem = 0.0
    t_all = perf_counter()
    for request in requests:
        prompt_text = tokenizer.decode(request.prompt_token_ids)
        last = None
        for response in stream_generate(
            model,
            tokenizer,
            prompt=prompt_text,
            max_tokens=request.max_new_tokens,
            sampler=sampler,
        ):
            last = response
        if last is None:
            continue
        prompt_tokens = getattr(last, "prompt_tokens", 0) or 0
        generation_tokens = getattr(last, "generation_tokens", 0) or 0
        prompt_tps = getattr(last, "prompt_tps", 0.0) or 0.0
        generation_tps = getattr(last, "generation_tps", 0.0) or 0.0
        peak_memory = getattr(last, "peak_memory", 0.0) or 0.0
        if prompt_tokens > 0:
            weighted_prefill_tps += prompt_tps * prompt_tokens
            total_prompt += prompt_tokens
        if generation_tokens > 0:
            weighted_decode_tps += generation_tps * generation_tokens
            total_gen += generation_tokens
        if peak_memory > peak_mem:
            peak_mem = peak_memory
    wall_s = perf_counter() - t_all
    return {
        "prompt_tokens": total_prompt,
        "gen_tokens": total_gen,
        "decode_tokens": total_gen,
        "prefill_tps": weighted_prefill_tps / total_prompt if total_prompt else 0.0,
        "decode_tps": weighted_decode_tps / total_gen if total_gen else 0.0,
        "wall_s": wall_s,
        "peak_memory_gb": max(peak_mem, get_peak_memory_gb() or 0.0),
    }


def measure_engine(
    *,
    engine: str,
    model_short: str,
    full_name: str,
    mlx_model,
    tokenizer,
    requests: list[BenchRequest],
    reps: int,
) -> list[dict[str, object]]:
    if engine == "stage1_ref_full_context":
        model = ref_models.dispatch_model(full_name, mlx_model, week=1)
        runner = lambda req: run_one_request_full_context(
            model, req, stage2_fresh_cache=False
        )
        run_one = lambda: aggregate_timing(requests, runner)
    elif engine == "stage2_full_context":
        model = self_models.dispatch_model(full_name, mlx_model, week=2)
        runner = lambda req: run_one_request_full_context(
            model, req, stage2_fresh_cache=True
        )
        run_one = lambda: aggregate_timing(requests, runner)
    elif engine == "stage2_kv_cache":
        model = self_models.dispatch_model(full_name, mlx_model, week=2)
        run_one = lambda: aggregate_timing(
            requests,
            lambda req: run_one_request_stage2_kv_cache(model, req),
        )
    elif engine == "mlx_lm":
        model = None
        run_one = lambda: measure_mlx_lm(mlx_model, tokenizer, requests)
    else:
        raise ValueError(engine)

    records: list[dict[str, object]] = []
    for rep in range(1, reps + 1):
        print(f"  {ENGINE_LABEL[engine]} rep {rep}/{reps}", flush=True)
        result = run_one()
        result.update({"engine": engine, "engine_label": ENGINE_LABEL[engine], "model": model_short, "rep": rep})
        records.append(result)
        peak = result.get("peak_memory_gb")
        peak_str = f"{float(peak):.2f} GB" if peak else "n/a"
        print(
            f"    prefill={float(result['prefill_tps']):.1f} tok/s "
            f"decode={float(result['decode_tps']):.2f} tok/s "
            f"wall={float(result['wall_s']):.2f}s peak={peak_str}",
            flush=True,
        )
    if model is not None:
        del model
    clear_mlx_cache()
    return records


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    if not rows:
        return
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_engine: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_engine.setdefault(str(record["engine"]), []).append(record)

    rows: list[dict[str, object]] = []
    for engine in ENGINE_ORDER:
        items = by_engine.get(engine)
        if not items:
            continue
        prefill = [float(x["prefill_tps"]) for x in items]
        decode = [float(x["decode_tps"]) for x in items]
        wall = [float(x["wall_s"]) for x in items]
        peaks = [float(x["peak_memory_gb"]) for x in items if x.get("peak_memory_gb")]
        rows.append(
            {
                "engine": engine,
                "engine_label": ENGINE_LABEL[engine],
                "model": items[0]["model"],
                "n": len(items),
                "prefill_tps_mean": statistics.mean(prefill),
                "prefill_tps_stdev": statistics.stdev(prefill) if len(prefill) > 1 else 0.0,
                "decode_tps_mean": statistics.mean(decode),
                "decode_tps_stdev": statistics.stdev(decode) if len(decode) > 1 else 0.0,
                "wall_s_mean": statistics.mean(wall),
                "wall_s_stdev": statistics.stdev(wall) if len(wall) > 1 else 0.0,
                "peak_memory_gb_max": max(peaks) if peaks else "",
            }
        )
    return rows


def build_ratio_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    by_engine = {str(row["engine"]): row for row in summary_rows}
    rows: list[dict[str, object]] = []
    for ratio_id, label, numerator, denominator in RATIO_SPECS:
        if numerator not in by_engine or denominator not in by_engine:
            continue
        num = by_engine[numerator]
        den = by_engine[denominator]
        row = {
            "ratio_id": ratio_id,
            "ratio_label": label,
            "model": num["model"],
            "numerator_engine": numerator,
            "denominator_engine": denominator,
            "prefill_tps_ratio": float(num["prefill_tps_mean"]) / float(den["prefill_tps_mean"]),
            "decode_tps_ratio": float(num["decode_tps_mean"]) / float(den["decode_tps_mean"]),
            "wall_s_ratio": float(num["wall_s_mean"]) / float(den["wall_s_mean"]),
        }
        num_peak = num.get("peak_memory_gb_max")
        den_peak = den.get("peak_memory_gb_max")
        row["peak_memory_gb_ratio"] = (
            float(num_peak) / float(den_peak) if num_peak and den_peak else ""
        )
        rows.append(row)
    return rows


def apply_style() -> None:
    style = get_base_style()
    style.update(
        {
            "font.size": 8,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    plt.rcParams.update(style)


def plot_summary(summary_rows: list[dict[str, object]], ratio_rows: list[dict[str, object]], out_fig: Path) -> None:
    apply_style()
    engines = [str(row["engine"]) for row in summary_rows]
    labels = [ENGINE_LABEL[engine] for engine in engines]
    colors = [ENGINE_COLOR[engine] for engine in engines]
    x = list(range(len(summary_rows)))

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.95))
    panels = [
        ("decode_tps_mean", "decode_tps_stdev", "Decode throughput", "tok/s"),
        ("wall_s_mean", "wall_s_stdev", "Wall time", "s"),
        ("peak_memory_gb_max", None, "Peak memory", "GB"),
    ]
    for ax, (mean_key, stdev_key, title, ylabel) in zip(axes, panels, strict=True):
        values = [float(row[mean_key]) if row[mean_key] != "" else 0.0 for row in summary_rows]
        errors = (
            [float(row[stdev_key]) for row in summary_rows]
            if stdev_key is not None
            else None
        )
        ax.bar(x, values, yerr=errors, color=colors, capsize=2.3, width=0.62)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=24, ha="right")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.75)

    ratio_by_id = {str(row["ratio_id"]): row for row in ratio_rows}
    kv_ratio = ratio_by_id.get("stage2_kv_cache_over_stage2_full_context")
    total_ratio = ratio_by_id.get("stage2_kv_cache_over_stage1_ref")
    subtitle = []
    if kv_ratio is not None:
        subtitle.append(f"KV-cache isolated decode: {float(kv_ratio['decode_tps_ratio']):.2f}x")
    if total_ratio is not None:
        subtitle.append(f"Stage 2 total decode: {float(total_ratio['decode_tps_ratio']):.2f}x")
    if subtitle:
        fig.text(0.99, 0.01, "; ".join(subtitle), ha="right", va="bottom", fontsize=7)

    fig.suptitle("Stage 2 attribution benchmark")
    fig.tight_layout(pad=0.8)
    save_publication_figure(
        fig,
        out_fig,
        formats=["png", "pdf", "svg", "tiff"],
        dpi=600,
        transparent=False,
        facecolor="white",
        pad_inches=0.08,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_results, out_summary, out_ratios, out_fig = output_paths(args.output_stem)
    full_name = self_models.shortcut_name_to_full_name(args.model)
    print(f"Loading {full_name}")
    mlx_model, tokenizer = mlx_lm_load(full_name)
    requests = build_requests(
        num_seqs=args.num_seqs,
        vocab_size=mlx_model.args.vocab_size,
        eos_token_id=tokenizer.eos_token_id,
        seed=args.seed,
    )
    print(
        f"Workload: {len(requests)} requests, "
        f"{sum(len(x.prompt_token_ids) for x in requests)} prompt tokens, "
        f"{sum(x.max_new_tokens for x in requests)} max generated tokens"
    )

    records: list[dict[str, object]] = []
    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        for engine in args.engines:
            print(f"\n=== {ENGINE_LABEL[engine]} ===")
            records.extend(
                measure_engine(
                    engine=engine,
                    model_short=args.model,
                    full_name=full_name,
                    mlx_model=mlx_model,
                    tokenizer=tokenizer,
                    requests=requests,
                    reps=args.reps,
                )
            )

    summary_rows = summarize(records)
    ratio_rows = build_ratio_rows(summary_rows)
    write_csv(out_results, records)
    write_csv(out_summary, summary_rows)
    write_csv(out_ratios, ratio_rows)
    plot_summary(summary_rows, ratio_rows, out_fig)

    print("\nStage 2 attribution benchmark")
    print("-" * 94)
    for row in summary_rows:
        print(
            f"{row['engine_label']:<28} "
            f"prefill={float(row['prefill_tps_mean']):8.1f} tok/s  "
            f"decode={float(row['decode_tps_mean']):8.2f} tok/s  "
            f"wall={float(row['wall_s_mean']):7.2f}s"
        )
    if ratio_rows:
        print("-" * 94)
        for row in ratio_rows:
            print(
                f"{row['ratio_label']:<45} "
                f"decode={float(row['decode_tps_ratio']):7.3f}x  "
                f"wall={float(row['wall_s_ratio']):7.3f}x"
            )
    print("-" * 94)
    print(f"Raw CSV     -> {out_results}")
    print(f"Summary CSV -> {out_summary}")
    print(f"Ratios CSV  -> {out_ratios}")
    print(f"Figure      -> {out_fig}.png/.pdf/.svg/.tiff")


if __name__ == "__main__":
    main()
