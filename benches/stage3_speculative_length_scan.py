"""
Stage 3 speculative decoding output-length scan.

This script extends the Stage 3 accounting benchmark by varying generated
length. It asks whether longer outputs make greedy speculative decoding more
competitive after the fixed prompt/prefill cost is amortized.

Default run:
    pdm run python benches/stage3_speculative_length_scan.py

Outputs:
    benches/stage3_speculative_length_scan_results.csv
    benches/stage3_speculative_length_scan_summary.csv
    benches/stage3_speculative_length_scan_ratios.csv
    benches/stage3_speculative_length_scan.png/.pdf/.svg/.tiff
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np
from mlx_lm import load as mlx_lm_load


ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
SKILL_DIR = Path.home() / ".codex" / "skills" / "scientific-visualization"
sys.path.insert(0, str(ROOT / "benches"))
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(SKILL_DIR / "scripts"))
sys.path.insert(0, str(SKILL_DIR / "assets"))

from color_palettes import OKABE_ITO, OKABE_ITO_LIST  # noqa: E402
from figure_export import save_publication_figure  # noqa: E402
from stage3_speculative_accounting import (  # noqa: E402
    BenchRequest,
    build_ratio_rows,
    build_requests,
    clear_mlx_cache,
    parse_int_list,
    rows_for_rep,
    summarize,
    write_csv,
)
from style_presets import get_base_style  # noqa: E402
from tiny_llm import models as self_models  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage 3 output-length scan.")
    parser.add_argument("--target-model", default="qwen2-1.5b")
    parser.add_argument("--draft-model", default="qwen2-0.5b")
    parser.add_argument("--draft-lengths", default="2,4,8")
    parser.add_argument("--output-lengths", default="64,128,256")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--num-seqs", type=int, default=2)
    parser.add_argument("--min-prompt-len", type=int, default=64)
    parser.add_argument("--max-prompt-len", type=int, default=96)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument(
        "--output-stem",
        default="stage3_speculative_length_scan",
        help="Output file stem under benches/ without suffixes.",
    )
    return parser.parse_args()


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


def add_output_len(rows: list[dict[str, object]], output_len: int) -> list[dict[str, object]]:
    out = []
    for row in rows:
        new_row = dict(row)
        new_row["output_len"] = output_len
        out.append(new_row)
    return out


def write_csv_ordered(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    preferred = [
        "output_len",
        "mode",
        "mode_label",
        "draft_length",
        "target_model",
        "draft_model",
        "model_pair",
        "rep",
        "n",
        "matches_baseline",
        "matches_baseline_all",
        "acceptance_rate",
        "acceptance_rate_mean",
        "target_calls_per_decode_token",
        "target_calls_per_decode_token_mean",
        "draft_calls_per_decode_token",
        "draft_calls_per_decode_token_mean",
        "decode_tps",
        "decode_tps_mean",
        "wall_tokens_per_s",
        "wall_tokens_per_s_mean",
        "draft_time_share",
        "draft_time_share_mean",
        "target_verify_time_share",
        "target_verify_time_share_mean",
    ]
    fields = [field for field in preferred if any(field in row for row in rows)]
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    write_csv(path, rows, fields)


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def plot_summary(summary_rows: list[dict[str, object]], ratio_rows: list[dict[str, object]], out_fig: Path) -> None:
    apply_style()
    spec_rows = [row for row in summary_rows if row["mode"] == "speculative"]
    draft_lengths = sorted({int(row["draft_length"]) for row in spec_rows})
    output_lengths = sorted({int(row["output_len"]) for row in spec_rows})
    ratios_by_key = {
        (int(row["output_len"]), int(row["draft_length"])): row for row in ratio_rows
    }
    summary_by_key = {
        (int(row["output_len"]), int(row["draft_length"])): row for row in spec_rows
    }
    colors = {
        draft_length: OKABE_ITO_LIST[idx % len(OKABE_ITO_LIST)]
        for idx, draft_length in enumerate(draft_lengths)
    }

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.2), sharex=True)
    panels = [
        ("acceptance_rate_mean", "Draft acceptance", "accepted draft tokens (%)", 100.0),
        (
            "target_calls_per_decode_token_mean",
            "Target calls",
            "calls per decode token",
            1.0,
        ),
        ("decode_tps_ratio", "Decode throughput", "ratio vs baseline", 1.0),
        ("draft_time_share_mean", "Draft time share", "decode time (%)", 100.0),
    ]

    for ax, (metric, title, ylabel, scale) in zip(axes.ravel(), panels, strict=True):
        for draft_length in draft_lengths:
            ys = []
            for output_len in output_lengths:
                if metric == "decode_tps_ratio":
                    value = ratios_by_key[(output_len, draft_length)][metric]
                else:
                    value = summary_by_key[(output_len, draft_length)][metric]
                ys.append(float(value) * scale)
            ax.plot(
                output_lengths,
                ys,
                marker="o",
                linewidth=1.4,
                markersize=4,
                color=colors[draft_length],
                label=f"draft {draft_length}",
            )
        if metric in {"target_calls_per_decode_token_mean", "decode_tps_ratio"}:
            ax.axhline(
                1.0,
                color=OKABE_ITO["vermillion"],
                linewidth=1.0,
                linestyle="--",
                label="baseline" if metric == "decode_tps_ratio" else None,
            )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#d9dde3", linewidth=0.6, alpha=0.75)
        ax.set_xticks(output_lengths)
        ax.set_xticklabels([str(x) for x in output_lengths])

    for ax in axes[1, :]:
        ax.set_xlabel("generated tokens")
    axes[0, 0].legend(frameon=False, ncol=1)
    exact = all(int(row["matches_baseline_all"]) == 1 for row in spec_rows)
    fig.suptitle("Stage 3 speculative decoding output-length scan")
    fig.text(
        0.99,
        0.015,
        f"Qwen2 1.5B target / 0.5B draft; exact greedy match: {'yes' if exact else 'no'}",
        ha="right",
        va="bottom",
        fontsize=7,
    )
    fig.tight_layout(pad=0.8, rect=[0.0, 0.06, 1.0, 0.94])
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
    output_lengths = parse_int_list(args.output_lengths)
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

    all_records: list[dict[str, object]] = []
    all_summary: list[dict[str, object]] = []
    all_ratios: list[dict[str, object]] = []

    with mx.stream(mx.gpu if args.device == "gpu" else mx.cpu):
        for output_len in output_lengths:
            print(f"\n=== output_len={output_len} ===")
            requests = build_requests(
                num_seqs=args.num_seqs,
                min_prompt_len=args.min_prompt_len,
                max_prompt_len=args.max_prompt_len,
                max_new_tokens=output_len,
                vocab_size=vocab_size,
                eos_token_id=target_tokenizer.eos_token_id,
                seed=args.seed,
            )
            warm = BenchRequest(requests[0].prompt_token_ids[:16], min(8, output_len))
            from stage3_speculative_accounting import run_speculative_greedy, run_target_greedy

            run_target_greedy(target_model, warm)
            run_speculative_greedy(
                target_model=target_model,
                draft_model=draft_model,
                request=warm,
                draft_length=max(draft_lengths),
            )

            length_records: list[dict[str, object]] = []
            for rep in range(1, args.reps + 1):
                print(f"  rep {rep}/{args.reps}", flush=True)
                rep_rows = rows_for_rep(
                    rep=rep,
                    target_model_short=args.target_model,
                    draft_model_short=args.draft_model,
                    target_model=target_model,
                    draft_model=draft_model,
                    requests=requests,
                    draft_lengths=draft_lengths,
                )
                length_records.extend(rep_rows)
                clear_mlx_cache()

            length_summary = add_output_len(summarize(length_records), output_len)
            length_ratios = add_output_len(build_ratio_rows(length_summary), output_len)
            all_records.extend(add_output_len(length_records, output_len))
            all_summary.extend(length_summary)
            all_ratios.extend(length_ratios)

            for row in length_summary:
                if row["mode"] != "speculative":
                    continue
                print(
                    f"    draft={row['draft_length']} "
                    f"accept={100.0 * float(row['acceptance_rate_mean']):.1f}% "
                    f"target_calls/token={float(row['target_calls_per_decode_token_mean']):.3f} "
                    f"decode={float(row['decode_tps_mean']):.2f} tok/s "
                    f"exact={row['matches_baseline_all']}",
                    flush=True,
                )

    write_csv_ordered(out_results, all_records)
    write_csv_ordered(out_summary, all_summary)
    write_csv_ordered(out_ratios, all_ratios)
    plot_summary(all_summary, all_ratios, out_fig)

    print("\nStage 3 output-length scan")
    print("-" * 104)
    for row in all_ratios:
        print(
            f"out={int(row['output_len']):>3} draft={int(row['draft_length']):>2} "
            f"accept={100.0 * float(row['acceptance_rate']):5.1f}% "
            f"target_call_ratio={float(row['target_calls_per_decode_token_ratio']):6.3f}x "
            f"decode_ratio={float(row['decode_tps_ratio']):6.3f}x "
            f"exact={row['matches_baseline_all']}"
        )
    print("-" * 104)
    print(f"Raw CSV     -> {out_results}")
    print(f"Summary CSV -> {out_summary}")
    print(f"Ratios CSV  -> {out_ratios}")
    print(f"Figure      -> {out_fig}.png/.pdf/.svg/.tiff")


if __name__ == "__main__":
    main()
