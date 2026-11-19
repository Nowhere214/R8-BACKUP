"""
KV-cache microbenchmark for a Qwen2-style grouped attention block.

This is a controlled synthetic benchmark. It does not load model weights.
Instead, it uses Qwen2 0.5B-like attention dimensions and compares:

1. full_context_recompute:
   project Q/K/V for the whole context and compute causal attention for all
   positions, then read the last token output.

2. cached_one_token:
   assume previous K/V are already cached, project only the new token, append it
   to the cache, and compute attention for that one token.

The goal is to isolate the shape-level effect of KV cache before running a
full model implementation.

Usage:
    pdm run python benches/kv_cache_microbench.py

Outputs:
    benches/kv_cache_microbench_results.csv
    benches/kv_cache_microbench_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from pathlib import Path
from time import perf_counter


SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))


def build_case(mx, *, context_len: int, hidden_size: int, num_heads: int, num_kv_heads: int, seed: int):
    mx.random.seed(seed + context_len)
    head_dim = hidden_size // num_heads
    kv_hidden = num_kv_heads * head_dim

    x_context = mx.random.normal((1, context_len + 1, hidden_size), dtype=mx.float16)
    x_prev = x_context[:, :context_len, :]
    x_last = x_context[:, context_len:, :]

    # Weight layout follows tiny_llm_ref.basics.linear: output = x @ w.T.
    wq = mx.random.normal((hidden_size, hidden_size), dtype=mx.float16) * 0.02
    wk = mx.random.normal((kv_hidden, hidden_size), dtype=mx.float16) * 0.02
    wv = mx.random.normal((kv_hidden, hidden_size), dtype=mx.float16) * 0.02
    wo = mx.random.normal((hidden_size, hidden_size), dtype=mx.float16) * 0.02

    k_cache = mx.matmul(x_prev, wk.T).reshape(1, context_len, num_kv_heads, head_dim)
    v_cache = mx.matmul(x_prev, wv.T).reshape(1, context_len, num_kv_heads, head_dim)
    k_cache = k_cache.transpose(0, 2, 1, 3)
    v_cache = v_cache.transpose(0, 2, 1, 3)
    mx.eval(x_context, x_last, wq, wk, wv, wo, k_cache, v_cache)
    return {
        "x_context": x_context,
        "x_last": x_last,
        "wq": wq,
        "wk": wk,
        "wv": wv,
        "wo": wo,
        "k_cache": k_cache,
        "v_cache": v_cache,
        "head_dim": head_dim,
    }


def full_context_recompute(mx, attention_fn, case: dict, *, num_heads: int, num_kv_heads: int):
    x = case["x_context"]
    head_dim = case["head_dim"]
    wq = case["wq"]
    wk = case["wk"]
    wv = case["wv"]
    wo = case["wo"]
    _, length, hidden_size = x.shape

    q = mx.matmul(x, wq.T).reshape(1, length, num_heads, head_dim).transpose(0, 2, 1, 3)
    k = mx.matmul(x, wk.T).reshape(1, length, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v = mx.matmul(x, wv.T).reshape(1, length, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    out = attention_fn(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), mask="causal")
    out = out.astype(x.dtype).transpose(0, 2, 1, 3).reshape(1, length, hidden_size)
    last = out[:, -1:, :]
    return mx.matmul(last, wo.T)


def cached_one_token(mx, attention_fn, case: dict, *, num_heads: int, num_kv_heads: int):
    x = case["x_last"]
    head_dim = case["head_dim"]
    wq = case["wq"]
    wk = case["wk"]
    wv = case["wv"]
    wo = case["wo"]
    k_cache = case["k_cache"]
    v_cache = case["v_cache"]
    hidden_size = x.shape[-1]

    q = mx.matmul(x, wq.T).reshape(1, 1, num_heads, head_dim).transpose(0, 2, 1, 3)
    k_new = mx.matmul(x, wk.T).reshape(1, 1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    v_new = mx.matmul(x, wv.T).reshape(1, 1, num_kv_heads, head_dim).transpose(0, 2, 1, 3)
    k = mx.concat([k_cache, k_new], axis=2)
    v = mx.concat([v_cache, v_new], axis=2)
    out = attention_fn(q.astype(mx.float32), k.astype(mx.float32), v.astype(mx.float32), mask=None)
    out = out.astype(x.dtype).transpose(0, 2, 1, 3).reshape(1, 1, hidden_size)
    return mx.matmul(out, wo.T)


def time_path(mx, fn, *, warmup: int, reps: int) -> list[float]:
    for _ in range(warmup):
        out = fn()
        mx.eval(out)

    times = []
    for _ in range(reps):
        t0 = perf_counter()
        out = fn()
        mx.eval(out)
        times.append(perf_counter() - t0)
    return times


def summarize(records: list[dict]) -> list[dict]:
    grouped: dict[tuple[int, str], list[dict]] = {}
    for record in records:
        grouped.setdefault((record["context_len"], record["path"]), []).append(record)

    rows = []
    for (context_len, path), group in grouped.items():
        latencies = [r["latency_ms"] for r in group]
        rows.append(
            {
                "context_len": context_len,
                "path": path,
                "n": len(group),
                "latency_ms_mean": statistics.mean(latencies),
                "latency_ms_median": statistics.median(latencies),
                "latency_ms_stdev": statistics.stdev(latencies) if len(latencies) > 1 else 0.0,
            }
        )

    by_context: dict[int, dict[str, dict]] = {}
    for row in rows:
        by_context.setdefault(row["context_len"], {})[row["path"]] = row
    for context_len, values in by_context.items():
        if "full_context_recompute" in values and "cached_one_token" in values:
            speedup = (
                values["full_context_recompute"]["latency_ms_median"]
                / values["cached_one_token"]["latency_ms_median"]
            )
            values["full_context_recompute"]["cached_speedup_x"] = speedup
            values["cached_one_token"]["cached_speedup_x"] = speedup

    for row in rows:
        row.setdefault("cached_speedup_x", "")
    return sorted(rows, key=lambda r: (r["context_len"], r["path"]))


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
    print("=" * 94)
    print("KV CACHE MICROBENCH SUMMARY")
    print("=" * 94)
    print(f"{'Context':>8} {'Path':<24} {'N':>2} {'Median ms':>18} {'Mean ms':>18} {'Cached speedup':>16}")
    print("-" * 94)
    for row in rows:
        speedup = row["cached_speedup_x"]
        speedup_str = f"{speedup:.1f}x" if isinstance(speedup, float) else ""
        print(
            f"{row['context_len']:>8} {row['path']:<24} {row['n']:>2} "
            f"{row['latency_ms_median']:9.3f} "
            f"{row['latency_ms_mean']:9.3f} ± {row['latency_ms_stdev']:<8.3f} "
            f"{speedup_str:>16}"
        )
    print("=" * 94)


def parse_int_csv_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--context-lens", default="64,128,256,512,1024")
    parser.add_argument("--hidden-size", type=int, default=896)
    parser.add_argument("--num-heads", type=int, default=14)
    parser.add_argument("--num-kv-heads", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--reps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-prefix", default="benches/kv_cache_microbench")
    args = parser.parse_args()

    import mlx.core as mx
    from tiny_llm_ref.attention import scaled_dot_product_attention_grouped

    context_lens = parse_int_csv_list(args.context_lens)
    records: list[dict] = []
    for context_len in context_lens:
        print(f"Running context_len={context_len}", flush=True)
        case = build_case(
            mx,
            context_len=context_len,
            hidden_size=args.hidden_size,
            num_heads=args.num_heads,
            num_kv_heads=args.num_kv_heads,
            seed=args.seed,
        )
        paths = {
            "full_context_recompute": lambda: full_context_recompute(
                mx,
                scaled_dot_product_attention_grouped,
                case,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
            ),
            "cached_one_token": lambda: cached_one_token(
                mx,
                scaled_dot_product_attention_grouped,
                case,
                num_heads=args.num_heads,
                num_kv_heads=args.num_kv_heads,
            ),
        }
        for path, fn in paths.items():
            times = time_path(mx, fn, warmup=args.warmup, reps=args.reps)
            for rep, seconds in enumerate(times, start=1):
                records.append(
                    {
                        "context_len": context_len,
                        "path": path,
                        "rep": rep,
                        "latency_ms": seconds * 1000,
                        "hidden_size": args.hidden_size,
                        "num_heads": args.num_heads,
                        "num_kv_heads": args.num_kv_heads,
                        "head_dim": args.hidden_size // args.num_heads,
                    }
                )
            print(f"  {path}: {statistics.mean(times) * 1000:.3f} ms")

    raw_path = Path(f"{args.out_prefix}_results.csv")
    summary_path = Path(f"{args.out_prefix}_summary.csv")
    write_csv(raw_path, records)
    summary = summarize(records)
    write_csv(summary_path, summary)
    print_summary(summary)
    print(f"\nRaw     -> {raw_path}")
    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
