"""
Component-level timing for the Stage 1 reference decode path.

This is a Phase 0.5 diagnostic script, not an optimization benchmark. It uses
synchronized MLX timings around coarse code regions so we can attribute the
Stage 1 decode bottleneck to implementation components:

    - Q/K/V projection
    - attention core
    - attention output projection
    - MLP
    - elementwise / norm / reshape work
    - LM head
    - outer Python loop, concat, and sampler

Default run:

    pdm run python benches/week1_decode_component_profile.py

Outputs:
    benches/stage1_decode_component_profile_results.csv
    benches/stage1_decode_component_profile_summary.csv
    benches/stage1_decode_component_profile.png
    benches/stage1_decode_component_profile.pdf
    benches/stage1_decode_component_profile.svg
    benches/stage1_decode_component_profile.tiff
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from random import Random
from time import perf_counter
from typing import Callable

import matplotlib.pyplot as plt
import mlx.core as mx
from mlx_lm import load


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from color_palettes import OKABE_ITO  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from style_presets import apply_publication_style  # noqa: E402
from tiny_llm_ref import models  # noqa: E402
from tiny_llm_ref.attention import scaled_dot_product_attention_grouped  # noqa: E402
from tiny_llm_ref.basics import linear, silu  # noqa: E402


OUT_RESULTS = ROOT / "benches" / "stage1_decode_component_profile_results.csv"
OUT_SUMMARY = ROOT / "benches" / "stage1_decode_component_profile_summary.csv"
OUT_FIG = ROOT / "benches" / "stage1_decode_component_profile"

RESULT_FIELDS = [
    "model",
    "prompt_len",
    "context_len",
    "rep",
    "component",
    "seconds",
    "notes",
]

SUMMARY_FIELDS = [
    "model",
    "prompt_len",
    "context_len",
    "component",
    "mean_ms",
    "stdev_ms",
    "share_pct",
    "n",
    "notes",
]

PLOT_GROUPS = [
    ("qkv_projection", "QKV projection"),
    ("attention_core", "Attention core"),
    ("attention_out_projection", "Attention out proj"),
    ("mlp", "MLP"),
    ("elementwise", "Elementwise/norm"),
    ("lm_head", "LM head"),
    ("outer_loop", "Python loop/sampler"),
]

PLOT_COMPONENTS = [name for name, _ in PLOT_GROUPS]

COMPONENT_TO_PLOT = {
    "qkv_projection": "qkv_projection",
    "attention_core": "attention_core",
    "attention_out_projection": "attention_out_projection",
    "mlp": "mlp",
    "embedding": "elementwise",
    "elementwise_norm": "elementwise",
    "elementwise_rope_reshape": "elementwise",
    "elementwise_residual": "elementwise",
    "lm_head": "lm_head",
    "outer_loop_concat": "outer_loop",
    "outer_loop_sampler": "outer_loop",
    "outer_loop_misc": "outer_loop",
}


@dataclass
class ProfileCase:
    prompt_len: int
    context_len: int
    prompt: mx.array
    token: mx.array


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile coarse components of Stage 1 reference decode."
    )
    parser.add_argument("--model", default="qwen2-0.5b")
    parser.add_argument("--prompt-lens", default="64,128,512")
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    return parser.parse_args()


def parse_prompt_lens(value: str) -> list[int]:
    prompt_lens = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not prompt_lens:
        raise ValueError("--prompt-lens must contain at least one integer")
    if any(x <= 0 for x in prompt_lens):
        raise ValueError("--prompt-lens entries must be positive")
    return prompt_lens


def eval_value(value):
    if isinstance(value, tuple):
        mx.eval(*value)
    elif isinstance(value, list):
        mx.eval(*value)
    else:
        mx.eval(value)
    return value


def timed(
    acc: dict[str, float],
    component: str,
    fn: Callable[[], object],
):
    t0 = perf_counter()
    value = fn()
    eval_value(value)
    acc[component] += perf_counter() - t0
    return value


def timed_attention(attn, x: mx.array, mask: str | mx.array | None, acc: dict[str, float]):
    B, L, _ = x.shape

    def qkv_ops():
        q = linear(x, attn.wq, bias=attn.bq).reshape(
            B, L, attn.num_heads, attn.head_dim
        )
        k = linear(x, attn.wk, bias=attn.bk).reshape(
            B, L, attn.num_kv_heads, attn.head_dim
        )
        v = linear(x, attn.wv, bias=attn.bv).reshape(
            B, L, attn.num_kv_heads, attn.head_dim
        )
        return q, k, v

    q, k, v = timed(acc, "qkv_projection", qkv_ops)

    def rope_and_layout_ops():
        q2 = attn.rope(q, offset=slice(0, L))
        k2 = attn.rope(k, offset=slice(0, L))
        q2 = q2.transpose(0, 2, 1, 3)
        k2 = k2.transpose(0, 2, 1, 3)
        v2 = v.transpose(0, 2, 1, 3)
        return q2, k2, v2

    q, k, v = timed(acc, "elementwise_rope_reshape", rope_and_layout_ops)

    def attention_ops():
        return scaled_dot_product_attention_grouped(
            q.astype(mx.float32),
            k.astype(mx.float32),
            v.astype(mx.float32),
            scale=attn.scale,
            mask=mask,
        ).astype(x.dtype)

    y = timed(acc, "attention_core", attention_ops)
    y = timed(
        acc,
        "elementwise_rope_reshape",
        lambda: y.transpose(0, 2, 1, 3).reshape(B, L, attn.hidden_size),
    )
    return timed(acc, "attention_out_projection", lambda: linear(y, attn.wo))


def timed_mlp(mlp, x: mx.array, acc: dict[str, float]) -> mx.array:
    def mlp_ops():
        gate = linear(x, mlp.w_gate)
        up = linear(x, mlp.w_up)
        return linear(silu(gate) * up, mlp.w_down)

    return timed(acc, "mlp", mlp_ops)


def timed_forward(model, inputs: mx.array, acc: dict[str, float]) -> mx.array:
    h = timed(acc, "embedding", lambda: model.embedding(inputs))
    for layer in model.layers_inner:
        attn_in = timed(acc, "elementwise_norm", lambda: layer.input_layernorm(h))
        attn_out = timed_attention(layer.self_attn, attn_in, "causal", acc)
        h = timed(acc, "elementwise_residual", lambda: h + attn_out)
        mlp_in = timed(acc, "elementwise_norm", lambda: layer.post_attention_layernorm(h))
        mlp_out = timed_mlp(layer.mlp, mlp_in, acc)
        h = timed(acc, "elementwise_residual", lambda: h + mlp_out)
    h = timed(acc, "elementwise_norm", lambda: model.norm(h))
    if model.w_lm_head is not None:
        return timed(acc, "lm_head", lambda: linear(h, model.w_lm_head))
    return timed(acc, "lm_head", lambda: model.embedding.as_linear(h))


def build_cases(
    *,
    prompt_lens: list[int],
    vocab_size: int,
    eos_token_id: int,
    seed: int,
) -> list[ProfileCase]:
    rng = Random(seed)
    low = 256 if vocab_size > 512 else 0
    high = vocab_size - 1
    cases = []
    for prompt_len in prompt_lens:
        ids: list[int] = []
        for _ in range(prompt_len):
            token_id = rng.randint(low, high)
            if token_id == eos_token_id:
                token_id = token_id - 1 if token_id > low else token_id + 1
            ids.append(token_id)
        next_id = rng.randint(low, high)
        if next_id == eos_token_id:
            next_id = next_id - 1 if next_id > low else next_id + 1
        cases.append(
            ProfileCase(
                prompt_len=prompt_len,
                context_len=prompt_len + 1,
                prompt=mx.array(ids, dtype=mx.int32),
                token=mx.array([next_id], dtype=mx.int32),
            )
        )
    return cases


def profile_decode_step(model, case: ProfileCase) -> tuple[dict[str, float], float]:
    acc: dict[str, float] = defaultdict(float)
    t_loop = perf_counter()
    context = timed(acc, "outer_loop_concat", lambda: mx.concat([case.prompt, case.token]))
    logits = timed_forward(model, context[None, :], acc)
    _ = timed(acc, "outer_loop_sampler", lambda: mx.argmax(logits[:, -1, :], axis=-1))
    loop_wall = perf_counter() - t_loop
    measured = sum(acc.values())
    acc["outer_loop_misc"] += max(0.0, loop_wall - measured)
    return dict(acc), loop_wall


def rows_for_rep(
    *,
    model: str,
    case: ProfileCase,
    rep: int,
    acc: dict[str, float],
    loop_wall: float,
) -> list[dict[str, str]]:
    rows = []
    for component, seconds in sorted(acc.items()):
        rows.append(
            {
                "model": model,
                "prompt_len": str(case.prompt_len),
                "context_len": str(case.context_len),
                "rep": str(rep),
                "component": component,
                "seconds": f"{seconds:.9f}",
                "notes": "synchronized component timing",
            }
        )
    rows.append(
        {
            "model": model,
            "prompt_len": str(case.prompt_len),
            "context_len": str(case.context_len),
            "rep": str(rep),
            "component": "outer_loop_wall",
            "seconds": f"{loop_wall:.9f}",
            "notes": "wall time for one decode iteration",
        }
    )
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    values: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for row in rows:
        component = row["component"]
        if component == "outer_loop_wall":
            continue
        plot_component = COMPONENT_TO_PLOT.get(component, component)
        key = (row["model"], row["prompt_len"], row["context_len"], plot_component)
        values[key].append(float(row["seconds"]) * 1000.0)

    totals: dict[tuple[str, str, str], float] = defaultdict(float)
    for (model, prompt_len, context_len, component), vals in values.items():
        totals[(model, prompt_len, context_len)] += statistics.mean(vals)

    summary_rows = []
    for key in sorted(values, key=lambda x: (int(x[1]), PLOT_COMPONENTS.index(x[3]))):
        model, prompt_len, context_len, component = key
        vals = values[key]
        mean_ms = statistics.mean(vals)
        stdev_ms = statistics.stdev(vals) if len(vals) > 1 else 0.0
        total = totals[(model, prompt_len, context_len)]
        share = 100.0 * mean_ms / total if total > 0 else 0.0
        summary_rows.append(
            {
                "model": model,
                "prompt_len": prompt_len,
                "context_len": context_len,
                "component": component,
                "mean_ms": f"{mean_ms:.6f}",
                "stdev_ms": f"{stdev_ms:.6f}",
                "share_pct": f"{share:.3f}",
                "n": str(len(vals)),
                "notes": "aggregated from synchronized component timings",
            }
        )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary_rows: list[dict[str, str]]) -> None:
    apply_publication_style("default")
    by_prompt: dict[int, dict[str, float]] = defaultdict(dict)
    for row in summary_rows:
        prompt_len = int(row["prompt_len"])
        by_prompt[prompt_len][row["component"]] = float(row["mean_ms"])

    prompt_lens = sorted(by_prompt)
    x = list(range(len(prompt_lens)))
    colors = [
        OKABE_ITO["sky_blue"],
        OKABE_ITO["bluish_green"],
        OKABE_ITO["orange"],
        OKABE_ITO["vermillion"],
        OKABE_ITO["reddish_purple"],
        OKABE_ITO["blue"],
        OKABE_ITO["black"],
    ]

    fig, ax = plt.subplots(figsize=(7.0, 3.8))
    bottoms = [0.0] * len(prompt_lens)
    for (component, label), color in zip(PLOT_GROUPS, colors, strict=True):
        values = [by_prompt[p].get(component, 0.0) for p in prompt_lens]
        ax.bar(x, values, bottom=bottoms, label=label, color=color, width=0.62)
        bottoms = [a + b for a, b in zip(bottoms, values, strict=True)]

    ax.set_xticks(x)
    ax.set_xticklabels([str(p) for p in prompt_lens])
    ax.set_xlabel("Prompt tokens before decode step")
    ax.set_ylabel("Synchronized decode-step time (ms)")
    ax.set_title("Stage 1 reference decode component profile")
    ax.legend(ncol=2, loc="upper left", frameon=False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.text(
        0.99,
        -0.24,
        "Qwen2 0.5B Stage 1 ref; one full-context decode step; N=3.",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#526070",
    )
    fig.tight_layout(pad=0.8)
    save_publication_figure(
        fig,
        OUT_FIG,
        formats=["png", "pdf", "svg", "tiff"],
        dpi=300,
        pad_inches=0.08,
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    prompt_lens = parse_prompt_lens(args.prompt_lens)
    model_name = models.shortcut_name_to_full_name(args.model)
    print(f"Loading {model_name} for Stage 1 component profiling")
    mlx_model, tokenizer = load(model_name)

    stream = mx.gpu if args.device == "gpu" else mx.cpu
    with mx.stream(stream):
        model = models.dispatch_model(model_name, mlx_model, week=1)
        cases = build_cases(
            prompt_lens=prompt_lens,
            vocab_size=mlx_model.args.vocab_size,
            eos_token_id=tokenizer.eos_token_id,
            seed=args.seed,
        )

        for case in cases:
            for _ in range(args.warmup):
                profile_decode_step(model, case)

        rows: list[dict[str, str]] = []
        for case in cases:
            print(f"Profiling prompt_len={case.prompt_len}, context_len={case.context_len}")
            for rep in range(1, args.reps + 1):
                acc, loop_wall = profile_decode_step(model, case)
                rows.extend(
                    rows_for_rep(
                        model=args.model,
                        case=case,
                        rep=rep,
                        acc=acc,
                        loop_wall=loop_wall,
                    )
                )

    summary_rows = summarize(rows)
    write_csv(OUT_RESULTS, rows, RESULT_FIELDS)
    write_csv(OUT_SUMMARY, summary_rows, SUMMARY_FIELDS)
    plot_summary(summary_rows)

    print("Stage 1 decode component profile")
    print("-" * 80)
    for prompt_len in prompt_lens:
        subset = [r for r in summary_rows if int(r["prompt_len"]) == prompt_len]
        total = sum(float(r["mean_ms"]) for r in subset)
        print(f"prompt={prompt_len:4d} total={total:8.2f} ms")
        for row in subset:
            print(
                f"  {row['component']:<26} "
                f"{float(row['mean_ms']):8.2f} ms "
                f"{float(row['share_pct']):6.1f}%"
            )
    print("-" * 80)
    print(f"Raw CSV     -> {OUT_RESULTS}")
    print(f"Summary CSV -> {OUT_SUMMARY}")
    print(f"Figure      -> {OUT_FIG}.png/.pdf/.svg/.tiff")


if __name__ == "__main__":
    main()
