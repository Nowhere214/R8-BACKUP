"""
benches/biweekly_n3.py
======================

Phase 0 N=3 statistical-rigor benchmark for the bi-weekly report (2026-05-14).

Runs each (engine, model) combination 3 times in one Python session to
capture system-noise variance, then reports mean +- stdev. Each iteration
reuses the same loaded model object, so weights-load time is NOT counted.

Coverage (5 combinations, 15 timed measurements + warmups):
  SiliconLLM Stage 1 reference @ {qwen2-0.5b, qwen2-1.5b}
  mlx-lm                       @ {qwen2-0.5b, qwen2-1.5b, qwen2-7b}
  (Stage 1 ref @ 7B skipped: dequantized weights ~17 GB > 18 GB unified memory)

Outputs:
  benches/biweekly_n3_results.csv  -- one row per timed run
  benches/biweekly_n3_summary.csv  -- one row per (engine, model) with mean+-stdev
  stdout                          -- pretty summary table

Usage:
  cd ~/Documents/SiliconLLM
  pdm run python benches/biweekly_n3.py

Estimated wall time: ~15-20 minutes.
"""

from __future__ import annotations

import csv
import gc
import statistics
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter

import mlx.core as mx
from mlx_lm import load as mlx_lm_load
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

# Make `tiny_llm_ref` importable when running this file directly
# (pyproject.toml's pythonpath="src" only applies under pytest).
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))
from tiny_llm_ref import models as ref_models  # noqa: E402

# ---------- experiment parameters ----------
NUM_REPS = 3
NUM_SEQS = 4
SEED = 0


# ---------- workload generator (mirrors upstream bench.py) ----------

@dataclass
class BenchRequest:
    prompt_token_ids: list[int]
    max_new_tokens: int


def build_requests(num_seqs: int, vocab_size: int, eos_token_id: int, seed: int) -> list[BenchRequest]:
    rng = Random(seed)
    low = 256 if vocab_size > 512 else 0
    high = vocab_size - 1
    requests: list[BenchRequest] = []
    for _ in range(num_seqs):
        plen = rng.randint(64, 256)
        olen = rng.randint(64, 256)
        ids: list[int] = []
        for _ in range(plen):
            t = rng.randint(low, high)
            if t == eos_token_id:
                t = (t - 1) if t > low else (t + 1)
            ids.append(t)
        requests.append(BenchRequest(ids, olen))
    return requests


# ---------- SiliconLLM Stage 1 reference path ----------

def _sample_next_w1(model, y: mx.array) -> mx.array:
    logits = model(y[None, :])[:, -1, :]
    return mx.argmax(logits, axis=-1)


def _run_one_request_w1(model, request: BenchRequest) -> tuple[int, float, float]:
    context = mx.array(request.prompt_token_ids, dtype=mx.int32)
    t0 = perf_counter()
    token = _sample_next_w1(model, context)
    mx.eval(token)
    prefill_time = perf_counter() - t0
    generated = 1
    decode_time = 0.0
    for _ in range(request.max_new_tokens - 1):
        t1 = perf_counter()
        context = mx.concat([context, token])
        token = _sample_next_w1(model, context)
        mx.eval(token)
        decode_time += perf_counter() - t1
        generated += 1
    return generated, prefill_time, decode_time


def measure_week1_ref(model, requests: list[BenchRequest]) -> dict:
    _reset_peak_memory()
    # one warmup request
    _run_one_request_w1(model, requests[0])

    total_prompt = total_gen = 0
    total_prefill = total_decode = 0.0
    t_all = perf_counter()
    for req in requests:
        gen, pf, dc = _run_one_request_w1(model, req)
        total_prompt += len(req.prompt_token_ids)
        total_gen += gen
        total_prefill += pf
        total_decode += dc
    wall = perf_counter() - t_all

    return {
        "prompt_tokens": total_prompt,
        "gen_tokens": total_gen,
        "prefill_tps": total_prompt / total_prefill if total_prefill > 0 else 0.0,
        # decode_tps excludes the first generated token of each request (counted in prefill)
        "decode_tps": (total_gen - len(requests)) / total_decode if total_decode > 0 else 0.0,
        "wall_s": wall,
        "peak_memory_gb": _get_peak_memory_gb(),
    }


# ---------- mlx-lm path ----------

def measure_mlx_lm(model, tokenizer, requests: list[BenchRequest]) -> dict:
    sampler = make_sampler(0.0)
    _reset_peak_memory()

    # warmup
    warm_prompt = tokenizer.decode(requests[0].prompt_token_ids[:32])
    for _ in stream_generate(model, tokenizer, prompt=warm_prompt, max_tokens=8, sampler=sampler):
        pass

    total_prompt = total_gen = 0
    weighted_prefill_tps = 0.0
    weighted_decode_tps = 0.0
    peak_mem = 0.0
    t_all = perf_counter()
    for req in requests:
        prompt_text = tokenizer.decode(req.prompt_token_ids)
        last = None
        for r in stream_generate(
            model, tokenizer,
            prompt=prompt_text,
            max_tokens=req.max_new_tokens,
            sampler=sampler,
        ):
            last = r
        if last is None:
            continue
        pt = getattr(last, "prompt_tokens", 0) or 0
        gt = getattr(last, "generation_tokens", 0) or 0
        ptps = getattr(last, "prompt_tps", 0.0) or 0.0
        gtps = getattr(last, "generation_tps", 0.0) or 0.0
        pm = getattr(last, "peak_memory", 0.0) or 0.0
        if pt > 0:
            weighted_prefill_tps += ptps * pt
            total_prompt += pt
        if gt > 0:
            weighted_decode_tps += gtps * gt
            total_gen += gt
        if pm > peak_mem:
            peak_mem = pm
    wall = perf_counter() - t_all

    return {
        "prompt_tokens": total_prompt,
        "gen_tokens": total_gen,
        "prefill_tps": weighted_prefill_tps / total_prompt if total_prompt > 0 else 0.0,
        "decode_tps": weighted_decode_tps / total_gen if total_gen > 0 else 0.0,
        "wall_s": wall,
        "peak_memory_gb": max(peak_mem, _get_peak_memory_gb() or 0.0),
    }


# ---------- MLX memory helpers (API has changed across MLX versions) ----------

def _reset_peak_memory() -> None:
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


def _get_peak_memory_gb() -> float | None:
    for ns in (getattr(mx, "metal", None), getattr(mx, "memory", None)):
        if ns is None:
            continue
        for fn_name in ("get_peak_memory", "peak_memory"):
            fn = getattr(ns, fn_name, None)
            if callable(fn):
                try:
                    bytes_val = fn()
                    return float(bytes_val) / (1024 ** 3)
                except Exception:
                    pass
    return None


def _free_model(*objs) -> None:
    for o in objs:
        try:
            del o
        except Exception:
            pass
    gc.collect()
    for ns in (getattr(mx, "metal", None), getattr(mx, "memory", None)):
        if ns is None:
            continue
        for fn_name in ("clear_cache",):
            fn = getattr(ns, fn_name, None)
            if callable(fn):
                try:
                    fn()
                except Exception:
                    pass


# ---------- driver ----------

def run_combo(engine: str, model_short: str, num_reps: int) -> list[dict]:
    full_name = ref_models.shortcut_name_to_full_name(model_short)
    print(f"\n=== [{engine}] loading {full_name} ===", flush=True)
    t0 = perf_counter()
    mlx_model, tokenizer = mlx_lm_load(full_name)
    load_time = perf_counter() - t0
    print(f"    load took {load_time:.1f}s")

    requests = build_requests(NUM_SEQS, mlx_model.args.vocab_size, tokenizer.eos_token_id, SEED)
    total_prompt_tokens = sum(len(r.prompt_token_ids) for r in requests)
    total_max_gen = sum(r.max_new_tokens for r in requests)
    print(f"    workload: {len(requests)} requests, {total_prompt_tokens} prompt tok, "
          f"{total_max_gen} max-gen tok (seed={SEED})")

    ref_model = None
    if engine == "silicon_w1ref":
        ref_model = ref_models.dispatch_model(full_name, mlx_model, week=1)
        run_one = lambda: measure_week1_ref(ref_model, requests)
    elif engine == "mlx_lm":
        run_one = lambda: measure_mlx_lm(mlx_model, tokenizer, requests)
    else:
        raise ValueError(f"unknown engine: {engine}")

    results: list[dict] = []
    for rep in range(num_reps):
        print(f"    rep {rep+1}/{num_reps} ...", end="", flush=True)
        with mx.stream(mx.gpu):
            r = run_one()
        r["engine"] = engine
        r["model"] = model_short
        r["rep"] = rep + 1
        results.append(r)
        peak_str = f"{r['peak_memory_gb']:.2f} GB" if r.get("peak_memory_gb") else "?"
        print(f" prefill={r['prefill_tps']:7.1f} tps  decode={r['decode_tps']:6.2f} tps  "
              f"wall={r['wall_s']:6.2f}s  peak={peak_str}")

    _free_model(mlx_model, tokenizer, ref_model)
    return results


def summarize(records: list[dict]) -> list[dict]:
    by_combo: dict[tuple[str, str], list[dict]] = {}
    for r in records:
        by_combo.setdefault((r["engine"], r["model"]), []).append(r)

    summary: list[dict] = []
    for (engine, model_short), rs in by_combo.items():
        prefill = [r["prefill_tps"] for r in rs]
        decode = [r["decode_tps"] for r in rs]
        wall = [r["wall_s"] for r in rs]
        peaks = [r["peak_memory_gb"] for r in rs if r.get("peak_memory_gb")]
        summary.append({
            "engine": engine,
            "model": model_short,
            "n": len(rs),
            "prefill_tps_mean": statistics.mean(prefill),
            "prefill_tps_stdev": statistics.stdev(prefill) if len(prefill) > 1 else 0.0,
            "prefill_tps_cv_pct": (statistics.stdev(prefill) / statistics.mean(prefill) * 100) if len(prefill) > 1 and statistics.mean(prefill) > 0 else 0.0,
            "decode_tps_mean": statistics.mean(decode),
            "decode_tps_stdev": statistics.stdev(decode) if len(decode) > 1 else 0.0,
            "decode_tps_cv_pct": (statistics.stdev(decode) / statistics.mean(decode) * 100) if len(decode) > 1 and statistics.mean(decode) > 0 else 0.0,
            "wall_s_mean": statistics.mean(wall),
            "wall_s_stdev": statistics.stdev(wall) if len(wall) > 1 else 0.0,
            "peak_memory_gb_max": max(peaks) if peaks else None,
        })
    return summary


def write_csv(path: Path, records: list[dict]) -> None:
    if not records:
        return
    fieldnames: list[str] = []
    for r in records:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in records:
            w.writerow(r)


def pretty_print_summary(summary: list[dict]) -> None:
    print()
    print("=" * 110)
    print("PHASE 0 N=3 SUMMARY  (mean ± stdev across 3 in-session reps; same seed, same workload)")
    print("=" * 110)
    print(f"{'Engine':<16} {'Model':<12} {'N':>2}  "
          f"{'Prefill tok/s':>22}  {'Decode tok/s':>22}  "
          f"{'Wall (s)':>14}  {'Peak mem GB':>11}")
    print("-" * 110)
    for s in summary:
        prefill_str = f"{s['prefill_tps_mean']:8.1f} ± {s['prefill_tps_stdev']:6.2f}  ({s['prefill_tps_cv_pct']:4.1f}%)"
        decode_str = f"{s['decode_tps_mean']:8.2f} ± {s['decode_tps_stdev']:6.3f}  ({s['decode_tps_cv_pct']:4.1f}%)"
        wall_str = f"{s['wall_s_mean']:7.2f} ± {s['wall_s_stdev']:5.2f}"
        peak_str = f"{s['peak_memory_gb_max']:7.2f}" if s["peak_memory_gb_max"] else "      —"
        print(f"{s['engine']:<16} {s['model']:<12} {s['n']:>2}  "
              f"{prefill_str:>22}  {decode_str:>22}  {wall_str:>14}  {peak_str:>11}")
    print("=" * 110)


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    raw_csv = out_dir / "biweekly_n3_results.csv"
    sum_csv = out_dir / "biweekly_n3_summary.csv"

    # ordered to keep memory low: small models first, biggest last
    schedule: list[tuple[str, str]] = [
        ("silicon_w1ref", "qwen2-0.5b"),
        ("mlx_lm",         "qwen2-0.5b"),
        ("silicon_w1ref", "qwen2-1.5b"),
        ("mlx_lm",         "qwen2-1.5b"),
        ("mlx_lm",         "qwen2-7b"),
    ]

    all_records: list[dict] = []
    t_start = perf_counter()
    for engine, model_short in schedule:
        try:
            all_records.extend(run_combo(engine, model_short, NUM_REPS))
        except Exception as exc:
            print(f"\n[FAILED] {engine} @ {model_short}: {exc}")
            traceback.print_exc()
            print("(continuing with the next combo)\n")

    write_csv(raw_csv, all_records)
    summary = summarize(all_records)
    write_csv(sum_csv, summary)
    pretty_print_summary(summary)

    elapsed = perf_counter() - t_start
    print(f"\nTotal wall time: {elapsed:.1f} s ({elapsed/60:.1f} min)")
    print(f"Raw    -> {raw_csv}")
    print(f"Summary -> {sum_csv}")


if __name__ == "__main__":
    main()
