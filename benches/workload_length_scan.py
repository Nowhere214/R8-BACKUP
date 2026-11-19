"""
Workload length scan for Phase 0.

This script varies prompt length while keeping generation length fixed. The
goal is to test whether the decode gap changes as the context becomes longer.

Default run is intentionally small enough for a quick group-meeting datapoint:

    pdm run python benches/workload_length_scan.py

Outputs:
    benches/workload_length_scan_results.csv
    benches/workload_length_scan_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import gc
import statistics
import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter


SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


@dataclass
class BenchRequest:
    prompt_token_ids: list[int]
    max_new_tokens: int


def build_requests(
    *,
    num_seqs: int,
    prompt_len: int,
    gen_len: int,
    vocab_size: int,
    eos_token_id: int,
    seed: int,
) -> list[BenchRequest]:
    rng = Random(seed + prompt_len * 1009 + gen_len * 917)
    low = 256 if vocab_size > 512 else 0
    high = vocab_size - 1
    requests: list[BenchRequest] = []
    for _ in range(num_seqs):
        ids: list[int] = []
        for _ in range(prompt_len):
            token_id = rng.randint(low, high)
            if token_id == eos_token_id:
                token_id = token_id - 1 if token_id > low else token_id + 1
            ids.append(token_id)
        requests.append(BenchRequest(ids, gen_len))
    return requests


def reset_peak_memory(mx) -> None:
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


def get_peak_memory_gb(mx) -> float | None:
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


def clear_mlx_cache(mx) -> None:
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


def sample_next_w1(mx, model, context):
    logits = model(context[None, :])[:, -1, :]
    return mx.argmax(logits, axis=-1)


def run_one_request_w1(mx, model, request: BenchRequest) -> tuple[int, float, float]:
    context = mx.array(request.prompt_token_ids, dtype=mx.int32)
    t0 = perf_counter()
    token = sample_next_w1(mx, model, context)
    mx.eval(token)
    prefill_time = perf_counter() - t0

    generated = 1
    decode_time = 0.0
    for _ in range(request.max_new_tokens - 1):
        t1 = perf_counter()
        context = mx.concat([context, token])
        token = sample_next_w1(mx, model, context)
        mx.eval(token)
        decode_time += perf_counter() - t1
        generated += 1
    return generated, prefill_time, decode_time


def measure_week1_ref(mx, model, requests: list[BenchRequest]) -> dict:
    reset_peak_memory(mx)
    run_one_request_w1(mx, model, requests[0])

    total_prompt = 0
    total_gen = 0
    total_prefill = 0.0
    total_decode = 0.0
    t_all = perf_counter()
    for request in requests:
        generated, prefill_time, decode_time = run_one_request_w1(mx, model, request)
        total_prompt += len(request.prompt_token_ids)
        total_gen += generated
        total_prefill += prefill_time
        total_decode += decode_time
    wall = perf_counter() - t_all

    return {
        "prompt_tokens": total_prompt,
        "gen_tokens": total_gen,
        "prefill_tps": total_prompt / total_prefill if total_prefill > 0 else 0.0,
        "decode_tps": (total_gen - len(requests)) / total_decode if total_decode > 0 else 0.0,
        "wall_s": wall,
        "peak_memory_gb": get_peak_memory_gb(mx),
    }


@contextmanager
def ignore_tokenizer_eos(tokenizer, enabled: bool):
    if not enabled:
        yield
        return
    old_value = getattr(tokenizer, "eos_token_ids", None)
    try:
        setattr(tokenizer, "eos_token_ids", [])
    except Exception:
        yield
        return
    try:
        yield
    finally:
        try:
            setattr(tokenizer, "eos_token_ids", old_value)
        except Exception:
            pass


def measure_mlx_lm(
    mx,
    stream_generate,
    make_sampler,
    model,
    tokenizer,
    requests: list[BenchRequest],
    *,
    ignore_eos: bool,
) -> dict:
    sampler = make_sampler(0.0)
    reset_peak_memory(mx)

    with ignore_tokenizer_eos(tokenizer, ignore_eos):
        warm_prompt = requests[0].prompt_token_ids[: min(32, len(requests[0].prompt_token_ids))]
        for _ in stream_generate(model, tokenizer, prompt=warm_prompt, max_tokens=8, sampler=sampler):
            pass

        total_prompt = 0
        total_gen = 0
        weighted_prefill_tps = 0.0
        weighted_decode_tps = 0.0
        peak_mem = 0.0
        t_all = perf_counter()
        for request in requests:
            last = None
            for result in stream_generate(
                model,
                tokenizer,
                prompt=request.prompt_token_ids,
                max_tokens=request.max_new_tokens,
                sampler=sampler,
            ):
                last = result
            if last is None:
                continue
            prompt_tokens = getattr(last, "prompt_tokens", 0) or 0
            gen_tokens = getattr(last, "generation_tokens", 0) or 0
            prompt_tps = getattr(last, "prompt_tps", 0.0) or 0.0
            gen_tps = getattr(last, "generation_tps", 0.0) or 0.0
            peak = getattr(last, "peak_memory", 0.0) or 0.0
            if prompt_tokens > 0:
                weighted_prefill_tps += prompt_tps * prompt_tokens
                total_prompt += prompt_tokens
            if gen_tokens > 0:
                weighted_decode_tps += gen_tps * gen_tokens
                total_gen += gen_tokens
            peak_mem = max(peak_mem, peak)
        wall = perf_counter() - t_all

    return {
        "prompt_tokens": total_prompt,
        "gen_tokens": total_gen,
        "prefill_tps": weighted_prefill_tps / total_prompt if total_prompt > 0 else 0.0,
        "decode_tps": weighted_decode_tps / total_gen if total_gen > 0 else 0.0,
        "wall_s": wall,
        "peak_memory_gb": max(peak_mem, get_peak_memory_gb(mx) or 0.0),
    }


def run_combo(
    *,
    engine: str,
    model_short: str,
    prompt_lens: list[int],
    gen_len: int,
    num_seqs: int,
    reps: int,
    seed: int,
    ignore_eos: bool,
) -> list[dict]:
    import mlx.core as mx
    from mlx_lm import load as mlx_lm_load
    from mlx_lm import stream_generate
    from mlx_lm.sample_utils import make_sampler
    from tiny_llm_ref import models as ref_models

    full_name = ref_models.shortcut_name_to_full_name(model_short)
    print(f"\n=== [{engine}] loading {full_name} ===", flush=True)
    t0 = perf_counter()
    mlx_model, tokenizer = mlx_lm_load(full_name)
    print(f"    load took {perf_counter() - t0:.1f}s", flush=True)

    ref_model = None
    if engine == "silicon_w1ref":
        ref_model = ref_models.dispatch_model(full_name, mlx_model, week=1)
    elif engine != "mlx_lm":
        raise ValueError(f"unknown engine: {engine}")

    records: list[dict] = []
    for prompt_len in prompt_lens:
        requests = build_requests(
            num_seqs=num_seqs,
            prompt_len=prompt_len,
            gen_len=gen_len,
            vocab_size=mlx_model.args.vocab_size,
            eos_token_id=tokenizer.eos_token_id,
            seed=seed,
        )
        print(
            f"    workload: prompt_len={prompt_len}, gen_len={gen_len}, "
            f"num_seqs={num_seqs}, reps={reps}",
            flush=True,
        )
        for rep in range(1, reps + 1):
            print(f"      rep {rep}/{reps} ...", end="", flush=True)
            with mx.stream(mx.gpu):
                if engine == "silicon_w1ref":
                    result = measure_week1_ref(mx, ref_model, requests)
                else:
                    result = measure_mlx_lm(
                        mx,
                        stream_generate,
                        make_sampler,
                        mlx_model,
                        tokenizer,
                        requests,
                        ignore_eos=ignore_eos,
                    )
            result.update(
                {
                    "engine": engine,
                    "model": model_short,
                    "requested_prompt_len": prompt_len,
                    "requested_gen_len": gen_len,
                    "num_seqs": num_seqs,
                    "rep": rep,
                }
            )
            records.append(result)
            peak = result.get("peak_memory_gb")
            peak_str = f"{peak:.2f} GB" if peak else "?"
            print(
                f" prefill={result['prefill_tps']:8.1f} tps "
                f"decode={result['decode_tps']:7.2f} tps "
                f"wall={result['wall_s']:7.2f}s peak={peak_str}",
                flush=True,
            )

    del mlx_model
    del tokenizer
    if ref_model is not None:
        del ref_model
    clear_mlx_cache(mx)
    return records


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[str, str, int, int, int], list[dict]] = {}
    for record in records:
        key = (
            record["engine"],
            record["model"],
            int(record["requested_prompt_len"]),
            int(record["requested_gen_len"]),
            int(record["num_seqs"]),
        )
        grouped.setdefault(key, []).append(record)

    rows: list[dict] = []
    for (engine, model, prompt_len, gen_len, num_seqs), group in grouped.items():
        prefill = [float(r["prefill_tps"]) for r in group]
        decode = [float(r["decode_tps"]) for r in group]
        wall = [float(r["wall_s"]) for r in group]
        peak = [float(r["peak_memory_gb"]) for r in group if r.get("peak_memory_gb")]
        prompt_tokens = [float(r["prompt_tokens"]) for r in group]
        gen_tokens = [float(r["gen_tokens"]) for r in group]
        rows.append(
            {
                "engine": engine,
                "model": model,
                "requested_prompt_len": prompt_len,
                "requested_gen_len": gen_len,
                "num_seqs": num_seqs,
                "n": len(group),
                "prompt_tokens_mean": statistics.mean(prompt_tokens),
                "gen_tokens_mean": statistics.mean(gen_tokens),
                "prefill_tps_mean": statistics.mean(prefill),
                "prefill_tps_stdev": statistics.stdev(prefill) if len(prefill) > 1 else 0.0,
                "decode_tps_mean": statistics.mean(decode),
                "decode_tps_stdev": statistics.stdev(decode) if len(decode) > 1 else 0.0,
                "wall_s_mean": statistics.mean(wall),
                "wall_s_stdev": statistics.stdev(wall) if len(wall) > 1 else 0.0,
                "peak_memory_gb_max": max(peak) if peak else None,
            }
        )
    return sorted(rows, key=lambda r: (r["model"], r["engine"], int(r["requested_prompt_len"])))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows: list[dict]) -> None:
    print()
    print("=" * 118)
    print("WORKLOAD LENGTH SCAN SUMMARY")
    print("=" * 118)
    print(
        f"{'Engine':<16} {'Model':<12} {'Prompt':>7} {'Gen':>5} {'N':>2} "
        f"{'Prefill tok/s':>20} {'Decode tok/s':>20} {'Wall s':>13} {'Peak GB':>9}"
    )
    print("-" * 118)
    for row in rows:
        peak = row["peak_memory_gb_max"]
        peak_str = f"{peak:7.2f}" if peak else "      -"
        print(
            f"{row['engine']:<16} {row['model']:<12} "
            f"{int(row['requested_prompt_len']):>7} {int(row['requested_gen_len']):>5} {int(row['n']):>2} "
            f"{row['prefill_tps_mean']:9.1f} ± {row['prefill_tps_stdev']:<7.2f} "
            f"{row['decode_tps_mean']:9.2f} ± {row['decode_tps_stdev']:<7.3f} "
            f"{row['wall_s_mean']:7.2f} ± {row['wall_s_stdev']:<4.2f} {peak_str:>9}"
        )
    print("=" * 118)


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", default="silicon_w1ref,mlx_lm")
    parser.add_argument("--models", default="qwen2-0.5b")
    parser.add_argument("--prompt-lens", default="64,128,512")
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--respect-eos", action="store_true")
    parser.add_argument("--out-prefix", default="benches/workload_length_scan")
    args = parser.parse_args()

    engines = parse_csv_list(args.engines)
    models = parse_csv_list(args.models)
    prompt_lens = parse_int_csv_list(args.prompt_lens)

    all_records: list[dict] = []
    t0 = perf_counter()
    for model in models:
        for engine in engines:
            try:
                all_records.extend(
                    run_combo(
                        engine=engine,
                        model_short=model,
                        prompt_lens=prompt_lens,
                        gen_len=args.gen_len,
                        num_seqs=args.num_seqs,
                        reps=args.reps,
                        seed=args.seed,
                        ignore_eos=not args.respect_eos,
                    )
                )
            except Exception as exc:
                print(f"\n[FAILED] {engine} @ {model}: {exc}")
                traceback.print_exc()
                print("(continuing with the next combo)\n")

    raw_path = Path(f"{args.out_prefix}_results.csv")
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    write_csv(raw_path, all_records)
    summary = summarize(all_records)
    write_csv(summary_path, summary)
    print_summary(summary)
    elapsed = perf_counter() - t0
    print(f"\nTotal wall time: {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Raw     -> {raw_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
