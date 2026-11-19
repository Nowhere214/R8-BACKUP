"""
Stage 3 greedy speculative decoding accounting benchmark.

This benchmark is intentionally narrow. It measures whether a draft model can
reduce target-model decode calls under the existing Stage 2 KV-cache path, while
checking that speculative greedy output exactly matches the target greedy
baseline.

Default run:
    pdm run python benches/stage3_speculative_accounting.py

Outputs:
    benches/stage3_speculative_accounting_results.csv
    benches/stage3_speculative_accounting_summary.csv
    benches/stage3_speculative_accounting_ratios.csv
    benches/stage3_speculative_accounting.png/.pdf/.svg/.tiff
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
from typing import Iterable

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
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
from tiny_llm.kv_cache import TinyKvFullCache  # noqa: E402


MODE_LABEL = {
    "baseline": "Stage 2 KV-cache greedy",
    "speculative": "Stage 3 speculative greedy",
}


@dataclass
class BenchRequest:
    prompt_token_ids: list[int]
    max_new_tokens: int


@dataclass
class RunStats:
    generated_token_ids: list[int]
    prompt_tokens: int
    prefill_time_s: float
    decode_time_s: float
    wall_s: float
    target_calls: int
    draft_calls: int
    proposed_draft_tokens: int
    accepted_draft_tokens: int
    draft_time_s: float = 0.0
    target_verify_time_s: float = 0.0
    misc_time_s: float = 0.0

    @property
    def generated_tokens(self) -> int:
        return len(self.generated_token_ids)

    @property
    def decode_tokens(self) -> int:
        return max(0, self.generated_tokens - 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run greedy speculative decoding accounting benchmark."
    )
    parser.add_argument("--target-model", default="qwen2-0.5b")
    parser.add_argument("--draft-model", default="qwen2-0.5b")
    parser.add_argument("--draft-lengths", default="1,2,4,8")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--min-prompt-len", type=int, default=64)
    parser.add_argument("--max-prompt-len", type=int, default=96)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--output-stem",
        default="stage3_speculative_accounting",
        help="Output file stem under benches/ without suffixes.",
    )
    return parser.parse_args()


def parse_int_list(value: str) -> list[int]:
    items = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not items or any(item < 1 for item in items):
        raise ValueError("draft lengths must be positive integers")
    return sorted(set(items))


def output_paths(stem: str) -> tuple[Path, Path, Path, Path]:
    out_stem = Path(stem)
    if out_stem.name != stem or out_stem.suffix:
        raise ValueError("--output-stem must be a bare filename stem")
    return (
        ROOT / "benches" / f"{stem}_results.csv",
        ROOT / "benches" / f"{stem}_summary.csv",
        ROOT / "benches" / f"{stem}_ratios.csv",
        ROOT / "benches" / stem,
    )


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


def build_requests(
    *,
    num_seqs: int,
    min_prompt_len: int,
    max_prompt_len: int,
    max_new_tokens: int,
    vocab_size: int,
    eos_token_id: int,
    seed: int,
) -> list[BenchRequest]:
    rng = Random(seed)
    low = 256 if vocab_size > 512 else 0
    high = vocab_size - 1
    requests: list[BenchRequest] = []
    for _ in range(num_seqs):
        prompt_len = rng.randint(min_prompt_len, max_prompt_len)
        ids: list[int] = []
        for _ in range(prompt_len):
            token_id = rng.randint(low, high)
            if token_id == eos_token_id:
                token_id = token_id - 1 if token_id > low else token_id + 1
            ids.append(token_id)
        requests.append(BenchRequest(ids, max_new_tokens))
    return requests


def make_cache(model) -> list[TinyKvFullCache]:
    return [TinyKvFullCache() for _ in range(model.num_hidden_layers)]


def next_token(model, token_ids: Iterable[int], offset: int, cache) -> int:
    tokens = mx.array(list(token_ids), dtype=mx.int32)
    logits = model(tokens[None, :], offset, cache)[:, -1, :]
    token = mx.argmax(logits, axis=-1)
    mx.eval(token)
    return int(token.item())


def predict_tokens(model, token_ids: list[int], offset: int, cache) -> list[int]:
    tokens = mx.array(token_ids, dtype=mx.int32)
    logits = model(tokens[None, :], offset, cache)
    preds = mx.argmax(logits, axis=-1)
    mx.eval(preds)
    return [int(x) for x in np.array(preds).reshape(-1)]


def rewind_cache(cache: list[TinyKvFullCache], n: int) -> None:
    if n <= 0:
        return
    for layer in cache:
        layer.rewind(n)


def run_target_greedy(model, request: BenchRequest) -> RunStats:
    cache = make_cache(model)
    prompt = request.prompt_token_ids

    t_all = perf_counter()
    t0 = perf_counter()
    token = next_token(model, prompt, 0, cache)
    prefill_time = perf_counter() - t0

    generated = [token]
    offset = len(prompt)
    decode_time = 0.0
    target_calls = 0
    while len(generated) < request.max_new_tokens:
        t1 = perf_counter()
        token = next_token(model, [token], offset, cache)
        decode_time += perf_counter() - t1
        offset += 1
        target_calls += 1
        generated.append(token)

    return RunStats(
        generated_token_ids=generated,
        prompt_tokens=len(prompt),
        prefill_time_s=prefill_time,
        decode_time_s=decode_time,
        wall_s=perf_counter() - t_all,
        target_calls=target_calls,
        draft_calls=0,
        proposed_draft_tokens=0,
        accepted_draft_tokens=0,
        target_verify_time_s=decode_time,
    )


def run_speculative_greedy(
    *,
    target_model,
    draft_model,
    request: BenchRequest,
    draft_length: int,
) -> RunStats:
    target_cache = make_cache(target_model)
    draft_cache = make_cache(draft_model)
    prompt = request.prompt_token_ids

    t_all = perf_counter()
    t0 = perf_counter()
    token = next_token(target_model, prompt, 0, target_cache)
    next_token(draft_model, prompt, 0, draft_cache)
    prefill_time = perf_counter() - t0

    generated = [token]
    target_offset = len(prompt)
    draft_offset = len(prompt)
    decode_time = 0.0
    target_calls = 0
    draft_calls = 0
    proposed_draft_tokens = 0
    accepted_draft_tokens = 0
    draft_time = 0.0
    target_verify_time = 0.0

    while len(generated) < request.max_new_tokens:
        remaining = request.max_new_tokens - len(generated)
        if remaining <= draft_length:
            for _ in range(remaining):
                t_tail = perf_counter()
                token = next_token(target_model, [token], target_offset, target_cache)
                elapsed = perf_counter() - t_tail
                decode_time += elapsed
                target_verify_time += elapsed
                target_offset += 1
                target_calls += 1
                generated.append(token)
            break

        iter_start = perf_counter()
        draft_tokens: list[int] = []
        draft_input = token
        for _ in range(draft_length):
            t_draft = perf_counter()
            draft_input = next_token(draft_model, [draft_input], draft_offset, draft_cache)
            draft_time += perf_counter() - t_draft
            draft_offset += 1
            draft_calls += 1
            proposed_draft_tokens += 1
            draft_tokens.append(draft_input)

        verify_input = [token, *draft_tokens]
        t_verify = perf_counter()
        target_preds = predict_tokens(target_model, verify_input, target_offset, target_cache)
        target_verify_time += perf_counter() - t_verify
        target_offset += len(verify_input)
        target_calls += 1

        accepted = 0
        mismatch_token: int | None = None
        for idx, draft_token in enumerate(draft_tokens):
            if target_preds[idx] == draft_token:
                accepted += 1
                continue
            mismatch_token = target_preds[idx]
            break

        accepted_draft_tokens += accepted
        if accepted < draft_length:
            rewind_cache(target_cache, draft_length - accepted)
            target_offset -= draft_length - accepted
            rewind_cache(draft_cache, draft_length - accepted - 1)
            draft_offset -= draft_length - accepted - 1
            generated.extend(draft_tokens[:accepted])
            token = int(mismatch_token)
            generated.append(token)
        else:
            # Align the draft cache with the accepted target prefix. The call's
            # prediction is discarded; only the cache update is needed.
            t_align = perf_counter()
            next_token(draft_model, [draft_tokens[-1]], draft_offset, draft_cache)
            draft_time += perf_counter() - t_align
            draft_offset += 1
            draft_calls += 1
            token = target_preds[draft_length]
            generated.extend(draft_tokens)
            generated.append(token)
        decode_time += perf_counter() - iter_start

    return RunStats(
        generated_token_ids=generated[: request.max_new_tokens],
        prompt_tokens=len(prompt),
        prefill_time_s=prefill_time,
        decode_time_s=decode_time,
        wall_s=perf_counter() - t_all,
        target_calls=target_calls,
        draft_calls=draft_calls,
        proposed_draft_tokens=proposed_draft_tokens,
        accepted_draft_tokens=accepted_draft_tokens,
        draft_time_s=draft_time,
        target_verify_time_s=target_verify_time,
        misc_time_s=max(0.0, decode_time - draft_time - target_verify_time),
    )


def aggregate_request_stats(items: list[RunStats]) -> dict[str, float | int | None]:
    prompt_tokens = sum(item.prompt_tokens for item in items)
    generated_tokens = sum(item.generated_tokens for item in items)
    decode_tokens = sum(item.decode_tokens for item in items)
    prefill_time = sum(item.prefill_time_s for item in items)
    decode_time = sum(item.decode_time_s for item in items)
    wall_s = sum(item.wall_s for item in items)
    target_calls = sum(item.target_calls for item in items)
    draft_calls = sum(item.draft_calls for item in items)
    proposed = sum(item.proposed_draft_tokens for item in items)
    accepted = sum(item.accepted_draft_tokens for item in items)
    draft_time = sum(item.draft_time_s for item in items)
    target_verify_time = sum(item.target_verify_time_s for item in items)
    misc_time = sum(item.misc_time_s for item in items)
    return {
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "decode_tokens": decode_tokens,
        "prefill_time_s": prefill_time,
        "decode_time_s": decode_time,
        "wall_s": wall_s,
        "target_calls": target_calls,
        "draft_calls": draft_calls,
        "proposed_draft_tokens": proposed,
        "accepted_draft_tokens": accepted,
        "draft_time_s": draft_time,
        "target_verify_time_s": target_verify_time,
        "misc_time_s": misc_time,
        "draft_time_share": draft_time / decode_time if decode_time else "",
        "target_verify_time_share": target_verify_time / decode_time if decode_time else "",
        "misc_time_share": misc_time / decode_time if decode_time else "",
        "acceptance_rate": accepted / proposed if proposed else "",
        "target_calls_per_decode_token": target_calls / decode_tokens if decode_tokens else "",
        "draft_calls_per_decode_token": draft_calls / decode_tokens if decode_tokens else "",
        "decode_tps": decode_tokens / decode_time if decode_time else 0.0,
        "wall_tokens_per_s": generated_tokens / wall_s if wall_s else 0.0,
    }


def rows_for_rep(
    *,
    rep: int,
    target_model_short: str,
    draft_model_short: str,
    target_model,
    draft_model,
    requests: list[BenchRequest],
    draft_lengths: list[int],
) -> list[dict[str, object]]:
    reset_peak_memory()

    baseline_outputs: list[list[int]] = []
    baseline_stats: list[RunStats] = []
    for request in requests:
        stats = run_target_greedy(target_model, request)
        baseline_outputs.append(stats.generated_token_ids)
        baseline_stats.append(stats)

    base_row = aggregate_request_stats(baseline_stats)
    base_row.update(
        {
            "mode": "baseline",
            "mode_label": MODE_LABEL["baseline"],
            "draft_length": 0,
            "target_model": target_model_short,
            "draft_model": "",
            "model_pair": f"{target_model_short}/none",
            "rep": rep,
            "matches_baseline": 1,
            "peak_memory_gb": get_peak_memory_gb(),
        }
    )

    rows: list[dict[str, object]] = [base_row]
    for draft_length in draft_lengths:
        spec_stats: list[RunStats] = []
        matches = True
        for idx, request in enumerate(requests):
            stats = run_speculative_greedy(
                target_model=target_model,
                draft_model=draft_model,
                request=request,
                draft_length=draft_length,
            )
            matches = matches and stats.generated_token_ids == baseline_outputs[idx]
            spec_stats.append(stats)

        row = aggregate_request_stats(spec_stats)
        row.update(
            {
                "mode": "speculative",
                "mode_label": MODE_LABEL["speculative"],
                "draft_length": draft_length,
                "target_model": target_model_short,
                "draft_model": draft_model_short,
                "model_pair": f"{target_model_short}/{draft_model_short}",
                "rep": rep,
                "matches_baseline": 1 if matches else 0,
                "peak_memory_gb": get_peak_memory_gb(),
            }
        )
        rows.append(row)
    return rows


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


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["mode"]), int(row["draft_length"])), []).append(row)

    summary: list[dict[str, object]] = []
    metric_keys = [
        "prefill_time_s",
        "decode_time_s",
        "draft_time_s",
        "target_verify_time_s",
        "misc_time_s",
        "draft_time_share",
        "target_verify_time_share",
        "misc_time_share",
        "wall_s",
        "acceptance_rate",
        "target_calls_per_decode_token",
        "draft_calls_per_decode_token",
        "decode_tps",
        "wall_tokens_per_s",
        "peak_memory_gb",
    ]
    for (mode, draft_length), items in sorted(groups.items(), key=lambda x: (x[0][0], x[0][1])):
        item0 = items[0]
        out: dict[str, object] = {
            "mode": mode,
            "mode_label": item0["mode_label"],
            "draft_length": draft_length,
            "target_model": item0["target_model"],
            "draft_model": item0["draft_model"],
            "model_pair": item0["model_pair"],
            "n": len(items),
            "prompt_tokens": item0["prompt_tokens"],
            "generated_tokens": item0["generated_tokens"],
            "decode_tokens": item0["decode_tokens"],
            "target_calls_mean": statistics.mean(float(x["target_calls"]) for x in items),
            "draft_calls_mean": statistics.mean(float(x["draft_calls"]) for x in items),
            "proposed_draft_tokens_mean": statistics.mean(
                float(x["proposed_draft_tokens"]) for x in items
            ),
            "accepted_draft_tokens_mean": statistics.mean(
                float(x["accepted_draft_tokens"]) for x in items
            ),
            "matches_baseline_all": min(int(x["matches_baseline"]) for x in items),
        }
        for key in metric_keys:
            values = [float(x[key]) for x in items if x.get(key, "") != ""]
            if not values:
                out[f"{key}_mean"] = ""
                out[f"{key}_stdev"] = ""
                continue
            out[f"{key}_mean"] = statistics.mean(values)
            out[f"{key}_stdev"] = statistics.stdev(values) if len(values) > 1 else 0.0
        summary.append(out)
    return summary


def build_ratio_rows(summary_rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline = next((row for row in summary_rows if row["mode"] == "baseline"), None)
    if baseline is None:
        return []

    rows: list[dict[str, object]] = []
    for row in summary_rows:
        if row["mode"] != "speculative":
            continue
        rows.append(
            {
                "target_model": row["target_model"],
                "draft_model": row["draft_model"],
                "model_pair": row["model_pair"],
                "draft_length": row["draft_length"],
                "decode_tps_ratio": float(row["decode_tps_mean"])
                / float(baseline["decode_tps_mean"]),
                "wall_s_ratio": float(row["wall_s_mean"]) / float(baseline["wall_s_mean"]),
                "target_calls_per_decode_token_ratio": float(
                    row["target_calls_per_decode_token_mean"]
                )
                / float(baseline["target_calls_per_decode_token_mean"]),
                "wall_tokens_per_s_ratio": float(row["wall_tokens_per_s_mean"])
                / float(baseline["wall_tokens_per_s_mean"]),
                "acceptance_rate": row["acceptance_rate_mean"],
                "matches_baseline_all": row["matches_baseline_all"],
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


def plot_summary(summary_rows: list[dict[str, object]], ratio_rows: list[dict[str, object]], out_fig: Path) -> None:
    apply_style()
    spec_rows = [row for row in summary_rows if row["mode"] == "speculative"]
    baseline = next(row for row in summary_rows if row["mode"] == "baseline")
    x = np.arange(len(spec_rows))
    labels = [str(row["draft_length"]) for row in spec_rows]

    fig, axes = plt.subplots(1, 3, figsize=(7.4, 2.9))
    color = OKABE_ITO["bluish_green"]
    baseline_color = OKABE_ITO["vermillion"]

    acceptance = [100.0 * float(row["acceptance_rate_mean"]) for row in spec_rows]
    axes[0].bar(x, acceptance, color=color, width=0.62)
    axes[0].set_title("Draft acceptance")
    axes[0].set_ylabel("accepted draft tokens (%)")
    axes[0].set_ylim(0, max(105, max(acceptance, default=0) * 1.15))

    target_calls = [float(row["target_calls_per_decode_token_mean"]) for row in spec_rows]
    axes[1].bar(x, target_calls, color=color, width=0.62, label="speculative")
    axes[1].axhline(
        float(baseline["target_calls_per_decode_token_mean"]),
        color=baseline_color,
        linewidth=1.2,
        linestyle="--",
        label="baseline",
    )
    axes[1].set_title("Target calls")
    axes[1].set_ylabel("calls per decode token")
    axes[1].legend(frameon=False)

    ratios_by_len = {int(row["draft_length"]): row for row in ratio_rows}
    decode_ratios = [
        float(ratios_by_len[int(row["draft_length"])]["decode_tps_ratio"])
        for row in spec_rows
    ]
    axes[2].bar(x, decode_ratios, color=color, width=0.62)
    axes[2].axhline(1.0, color=baseline_color, linewidth=1.2, linestyle="--")
    axes[2].set_title("Decode throughput")
    axes[2].set_ylabel("ratio vs baseline")

    for ax in axes:
        ax.set_xlabel("draft length")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.75)

    model_pair = str(spec_rows[0]["model_pair"]) if spec_rows else str(baseline["model_pair"])
    exact = all(int(row["matches_baseline_all"]) == 1 for row in spec_rows)
    fig.suptitle(f"Stage 3 speculative decoding accounting ({model_pair})")
    fig.text(
        0.99,
        0.015,
        f"Exact greedy match: {'yes' if exact else 'no'}",
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout(pad=0.8, rect=[0.0, 0.08, 1.0, 0.93])
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
    draft_lengths = parse_int_list(args.draft_lengths)
    out_results, out_summary, out_ratios, out_fig = output_paths(args.output_stem)
    target_full = self_models.shortcut_name_to_full_name(args.target_model)
    draft_full = self_models.shortcut_name_to_full_name(args.draft_model)

    print(f"Loading target: {target_full}")
    target_mlx_model, target_tokenizer = mlx_lm_load(target_full)
    if target_full == draft_full:
        print("Draft equals target: reusing one model object with separate KV caches")
        draft_mlx_model = target_mlx_model
    else:
        print(f"Loading draft:  {draft_full}")
        draft_mlx_model, _ = mlx_lm_load(draft_full)

    target_model = self_models.dispatch_model(target_full, target_mlx_model, week=2)
    draft_model = (
        target_model
        if target_full == draft_full
        else self_models.dispatch_model(draft_full, draft_mlx_model, week=2)
    )

    vocab_size = min(target_mlx_model.args.vocab_size, draft_mlx_model.args.vocab_size)
    requests = build_requests(
        num_seqs=args.num_seqs,
        min_prompt_len=args.min_prompt_len,
        max_prompt_len=args.max_prompt_len,
        max_new_tokens=args.max_new_tokens,
        vocab_size=vocab_size,
        eos_token_id=target_tokenizer.eos_token_id,
        seed=args.seed,
    )
    print(
        f"Workload: {len(requests)} requests, "
        f"{sum(len(x.prompt_token_ids) for x in requests)} prompt tokens, "
        f"{sum(x.max_new_tokens for x in requests)} generated tokens target, "
        f"draft lengths={draft_lengths}"
    )

    records: list[dict[str, object]] = []
    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        # Warm up the target path and the longest speculative path on a tiny request.
        warm = BenchRequest(requests[0].prompt_token_ids[:16], min(8, args.max_new_tokens))
        run_target_greedy(target_model, warm)
        run_speculative_greedy(
            target_model=target_model,
            draft_model=draft_model,
            request=warm,
            draft_length=max(draft_lengths),
        )

        for rep in range(1, args.reps + 1):
            print(f"\n=== rep {rep}/{args.reps} ===", flush=True)
            rep_rows = rows_for_rep(
                rep=rep,
                target_model_short=args.target_model,
                draft_model_short=args.draft_model,
                target_model=target_model,
                draft_model=draft_model,
                requests=requests,
                draft_lengths=draft_lengths,
            )
            for row in rep_rows:
                if row["mode"] == "baseline":
                    print(
                        f"  baseline decode={float(row['decode_tps']):.2f} tok/s "
                        f"target_calls/token={float(row['target_calls_per_decode_token']):.3f}",
                        flush=True,
                    )
                else:
                    print(
                        f"  draft={row['draft_length']} "
                        f"accept={100.0 * float(row['acceptance_rate']):.1f}% "
                        f"target_calls/token={float(row['target_calls_per_decode_token']):.3f} "
                        f"decode={float(row['decode_tps']):.2f} tok/s "
                        f"match={row['matches_baseline']}",
                        flush=True,
                    )
            records.extend(rep_rows)
            clear_mlx_cache()

    summary_rows = summarize(records)
    ratio_rows = build_ratio_rows(summary_rows)
    write_csv(out_results, records)
    write_csv(out_summary, summary_rows)
    write_csv(out_ratios, ratio_rows)
    plot_summary(summary_rows, ratio_rows, out_fig)

    print("\nStage 3 speculative decoding accounting")
    print("-" * 104)
    for row in summary_rows:
        accept = row.get("acceptance_rate_mean", "")
        accept_str = "n/a" if accept == "" else f"{100.0 * float(accept):5.1f}%"
        print(
            f"{str(row['mode_label']):<28} draft={int(row['draft_length']):>2} "
            f"decode={float(row['decode_tps_mean']):8.2f} tok/s  "
            f"target_calls/token={float(row['target_calls_per_decode_token_mean']):6.3f}  "
            f"accept={accept_str}  exact={row['matches_baseline_all']}"
        )
    if ratio_rows:
        print("-" * 104)
        for row in ratio_rows:
            print(
                f"draft={int(row['draft_length']):>2} "
                f"decode_tps_ratio={float(row['decode_tps_ratio']):6.3f}x  "
                f"target_call_ratio={float(row['target_calls_per_decode_token_ratio']):6.3f}x  "
                f"wall_ratio={float(row['wall_s_ratio']):6.3f}x"
            )
    print("-" * 104)
    print(f"Raw CSV     -> {out_results}")
    print(f"Summary CSV -> {out_summary}")
    print(f"Ratios CSV  -> {out_ratios}")
    print(f"Figure      -> {out_fig}.png/.pdf/.svg/.tiff")


if __name__ == "__main__":
    main()
