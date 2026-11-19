"""
Compare the self-implemented Stage 1 baseline against the Stage 1 reference.

This script is intentionally narrow and safe for the 18 GB Apple Silicon
machine. It only benchmarks Qwen2 0.5B by default, avoiding the known Stage 1
7B dequantization feasibility boundary.

Default run:

    pdm run python benches/week1_self_baseline_compare.py

Optional model-specific output:

    pdm run python benches/week1_self_baseline_compare.py \
        --model qwen2-1.5b \
        --output-stem stage1_self_baseline_compare_1p5b_smoke

Outputs:
    benches/stage1_self_baseline_compare_results.csv
    benches/stage1_self_baseline_compare_summary.csv
    benches/stage1_self_generation_sanity.csv
    benches/stage1_self_baseline_compare.png
    benches/stage1_self_baseline_compare.pdf
    benches/stage1_self_baseline_compare.svg
    benches/stage1_self_baseline_compare.tiff
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
from tiny_llm_ref import models as ref_models  # noqa: E402


OUT_RESULTS = ROOT / "benches" / "stage1_self_baseline_compare_results.csv"
OUT_SUMMARY = ROOT / "benches" / "stage1_self_baseline_compare_summary.csv"
OUT_SANITY = ROOT / "benches" / "stage1_self_generation_sanity.csv"
OUT_FIG = ROOT / "benches" / "stage1_self_baseline_compare"

NUM_SEQS = 4
SEED = 0

ENGINE_LABEL = {
    "tiny_llm_w1self": "Self Stage 1",
    "tiny_llm_ref_w1": "Reference Stage 1",
}

ENGINE_COLOR = {
    "tiny_llm_w1self": OKABE_ITO["bluish_green"],
    "tiny_llm_ref_w1": OKABE_ITO["vermillion"],
}


@dataclass
class BenchRequest:
    prompt_token_ids: list[int]
    max_new_tokens: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare self Stage 1 baseline with reference Stage 1 baseline."
    )
    parser.add_argument("--model", default="qwen2-0.5b")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--num-seqs", type=int, default=NUM_SEQS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--sanity-prompt",
        default="Give me a short introduction to large language models.",
    )
    parser.add_argument("--sanity-tokens", type=int, default=16)
    parser.add_argument(
        "--output-stem",
        default="stage1_self_baseline_compare",
        help="Output file stem under benches/ without suffixes.",
    )
    return parser.parse_args()


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


def sample_next_week1(model, context: mx.array) -> mx.array:
    logits = model(context[None, :])[:, -1, :]
    return mx.argmax(logits, axis=-1)


def run_one_request(model, request: BenchRequest) -> tuple[int, float, float]:
    context = mx.array(request.prompt_token_ids, dtype=mx.int32)
    t0 = perf_counter()
    token = sample_next_week1(model, context)
    mx.eval(token)
    prefill_time = perf_counter() - t0

    generated_tokens = 1
    decode_time = 0.0
    for _ in range(request.max_new_tokens - 1):
        t1 = perf_counter()
        context = mx.concat([context, token])
        token = sample_next_week1(model, context)
        mx.eval(token)
        decode_time += perf_counter() - t1
        generated_tokens += 1
    return generated_tokens, prefill_time, decode_time


def measure_model(model, requests: list[BenchRequest]) -> dict[str, float | int | None]:
    reset_peak_memory()
    run_one_request(model, requests[0])

    total_prompt = 0
    total_generated = 0
    total_prefill_time = 0.0
    total_decode_time = 0.0
    t_all = perf_counter()
    for request in requests:
        generated, prefill_time, decode_time = run_one_request(model, request)
        total_prompt += len(request.prompt_token_ids)
        total_generated += generated
        total_prefill_time += prefill_time
        total_decode_time += decode_time
    wall_s = perf_counter() - t_all

    decode_tokens = total_generated - len(requests)
    return {
        "prompt_tokens": total_prompt,
        "gen_tokens": total_generated,
        "prefill_tps": total_prompt / total_prefill_time if total_prefill_time else 0.0,
        "decode_tps": decode_tokens / total_decode_time if total_decode_time else 0.0,
        "wall_s": wall_s,
        "peak_memory_gb": get_peak_memory_gb(),
    }


def greedy_sanity(
    *,
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
) -> tuple[list[int], str]:
    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    context = mx.array(token_ids, dtype=mx.int32)
    generated: list[int] = []
    for _ in range(max_new_tokens):
        token = sample_next_week1(model, context)
        mx.eval(token)
        token_id = int(token.item())
        if token_id == tokenizer.eos_token_id:
            break
        generated.append(token_id)
        context = mx.concat([context, token])
    return generated, tokenizer.decode(generated)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def summarize(records: list[dict[str, object]]) -> list[dict[str, object]]:
    by_engine: dict[str, list[dict[str, object]]] = {}
    for record in records:
        by_engine.setdefault(str(record["engine"]), []).append(record)

    rows = []
    for engine, items in by_engine.items():
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


def plot_summary(summary_rows: list[dict[str, object]], out_fig: Path) -> None:
    apply_style()
    metrics = [
        ("prefill_tps_mean", "prefill_tps_stdev", "Prefill throughput", "tok/s"),
        ("decode_tps_mean", "decode_tps_stdev", "Decode throughput", "tok/s"),
        ("wall_s_mean", "wall_s_stdev", "Wall time", "s"),
    ]
    engines = [str(row["engine"]) for row in summary_rows]
    labels = [ENGINE_LABEL[e] for e in engines]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    for ax, (mean_key, stdev_key, title, ylabel) in zip(axes, metrics, strict=True):
        values = [float(row[mean_key]) for row in summary_rows]
        errors = [float(row[stdev_key]) for row in summary_rows]
        colors = [ENGINE_COLOR[e] for e in engines]
        ax.bar(labels, values, yerr=errors, color=colors, capsize=2.5, width=0.58)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=20)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.75)

    fig.suptitle("Stage 1 self baseline vs reference")
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


def run_engine(
    *,
    engine: str,
    model_short: str,
    mlx_model,
    requests: list[BenchRequest],
    reps: int,
) -> list[dict[str, object]]:
    full_name = self_models.shortcut_name_to_full_name(model_short)
    if engine == "tiny_llm_w1self":
        model = self_models.dispatch_model(full_name, mlx_model, week=1)
    elif engine == "tiny_llm_ref_w1":
        model = ref_models.dispatch_model(full_name, mlx_model, week=1)
    else:
        raise ValueError(engine)

    records: list[dict[str, object]] = []
    for rep in range(1, reps + 1):
        print(f"  {ENGINE_LABEL[engine]} rep {rep}/{reps}", flush=True)
        result = measure_model(model, requests)
        result.update({"engine": engine, "model": model_short, "rep": rep})
        records.append(result)
        peak = result.get("peak_memory_gb")
        peak_str = f"{float(peak):.2f} GB" if peak else "n/a"
        print(
            f"    prefill={float(result['prefill_tps']):.1f} tok/s "
            f"decode={float(result['decode_tps']):.2f} tok/s "
            f"wall={float(result['wall_s']):.2f}s peak={peak_str}",
            flush=True,
        )
    del model
    clear_mlx_cache()
    return records


def main() -> None:
    args = parse_args()
    out_stem = Path(args.output_stem)
    if out_stem.name != args.output_stem or out_stem.suffix:
        raise ValueError("--output-stem must be a bare filename stem without directories or suffix")
    if args.output_stem == "week1_self_baseline_compare":
        out_results = OUT_RESULTS
        out_summary = OUT_SUMMARY
        out_sanity = OUT_SANITY
        out_fig = OUT_FIG
    else:
        out_results = ROOT / "benches" / f"{args.output_stem}_results.csv"
        out_summary = ROOT / "benches" / f"{args.output_stem}_summary.csv"
        out_sanity = ROOT / "benches" / f"{args.output_stem}_generation_sanity.csv"
        out_fig = ROOT / "benches" / args.output_stem

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

    all_records: list[dict[str, object]] = []
    sanity_rows: list[dict[str, object]] = []
    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        for engine in ["tiny_llm_w1self", "tiny_llm_ref_w1"]:
            print(f"\n=== {ENGINE_LABEL[engine]} ===")
            if engine == "tiny_llm_w1self":
                model = self_models.dispatch_model(full_name, mlx_model, week=1)
            else:
                model = ref_models.dispatch_model(full_name, mlx_model, week=1)
            token_ids, text = greedy_sanity(
                model=model,
                tokenizer=tokenizer,
                prompt=args.sanity_prompt,
                max_new_tokens=args.sanity_tokens,
            )
            sanity_rows.append(
                {
                    "engine": engine,
                    "engine_label": ENGINE_LABEL[engine],
                    "model": args.model,
                    "prompt": args.sanity_prompt,
                    "max_new_tokens": args.sanity_tokens,
                    "generated_token_ids": " ".join(str(x) for x in token_ids),
                    "generated_text": text,
                }
            )
            del model
            clear_mlx_cache()

            all_records.extend(
                run_engine(
                    engine=engine,
                    model_short=args.model,
                    mlx_model=mlx_model,
                    requests=requests,
                    reps=args.reps,
                )
            )

    summary_rows = summarize(all_records)
    write_csv(out_results, all_records, list(all_records[0].keys()))
    write_csv(out_summary, summary_rows, list(summary_rows[0].keys()))
    write_csv(
        out_sanity,
        sanity_rows,
        [
            "engine",
            "engine_label",
            "model",
            "prompt",
            "max_new_tokens",
            "generated_token_ids",
            "generated_text",
        ],
    )
    plot_summary(summary_rows, out_fig)

    print("\nStage 1 self baseline comparison")
    print("-" * 80)
    for row in summary_rows:
        print(
            f"{row['engine_label']:<18} "
            f"prefill={float(row['prefill_tps_mean']):8.1f} ± {float(row['prefill_tps_stdev']):5.1f} tok/s  "
            f"decode={float(row['decode_tps_mean']):7.2f} ± {float(row['decode_tps_stdev']):5.2f} tok/s  "
            f"wall={float(row['wall_s_mean']):7.2f}s"
        )
    if len(summary_rows) == 2:
        by_engine = {str(row["engine"]): row for row in summary_rows}
        self_decode = float(by_engine["tiny_llm_w1self"]["decode_tps_mean"])
        ref_decode = float(by_engine["tiny_llm_ref_w1"]["decode_tps_mean"])
        self_prefill = float(by_engine["tiny_llm_w1self"]["prefill_tps_mean"])
        ref_prefill = float(by_engine["tiny_llm_ref_w1"]["prefill_tps_mean"])
        print(
            f"Self/ref ratio: prefill={self_prefill / ref_prefill:.3f}x, "
            f"decode={self_decode / ref_decode:.3f}x"
        )
        token_match = (
            sanity_rows[0]["generated_token_ids"] == sanity_rows[1]["generated_token_ids"]
        )
        print(f"Greedy sanity token match: {token_match}")
    print("-" * 80)
    print(f"Raw CSV     -> {out_results}")
    print(f"Summary CSV -> {out_summary}")
    print(f"Sanity CSV  -> {out_sanity}")
    print(f"Figure      -> {out_fig}.png/.pdf/.svg/.tiff")


if __name__ == "__main__":
    main()
