from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


REPO = Path(__file__).resolve().parents[1]
PROJECT = REPO / "projects" / "siliconllm_2026_07_01_formal_ppt169_20260701"
OUT = PROJECT / "images"

SCI = Path("/Users/laiqixiao/.codex/skills/scientific-visualization")
sys.path.insert(0, str(SCI / "scripts"))
sys.path.insert(0, str(SCI / "assets"))

from figure_export import save_publication_figure  # noqa: E402


COLORS = {
    "bg": "#F8FAFC",
    "text": "#0F172A",
    "muted": "#64748B",
    "grid": "#CBD5E1",
    "stage1": "#1E3A5F",
    "llama": "#0F766E",
    "mlx": "#D97706",
    "accent": "#2563EB",
    "soft_blue": "#DBEAFE",
    "soft_green": "#D1FAE5",
    "soft_orange": "#FEF3C7",
}


def setup_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 8.5,
            "axes.labelsize": 9,
            "axes.titlesize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#334155",
            "axes.linewidth": 0.8,
            "axes.grid": True,
            "grid.color": COLORS["grid"],
            "grid.linewidth": 0.5,
            "grid.alpha": 0.45,
            "savefig.facecolor": "white",
        }
    )


def export(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    save_publication_figure(
        fig,
        OUT / stem,
        formats=["png", "pdf", "svg", "tiff"],
        dpi=420,
        bbox_inches="tight",
        pad_inches=0.06,
        facecolor="white",
    )
    plt.close(fig)


def despine(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_phase0_decode() -> pd.DataFrame:
    df = pd.read_csv(REPO / "benches" / "phase0_unified_summary.csv")
    return df[
        (df["experiment_id"] == "model_size_engine_comparison")
        & (df["metric"] == "decode_tps")
        & (df["statistic"] == "mean")
    ].copy()


def fig_evidence_chain() -> None:
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    nodes = [
        ("Phase 0", "Measure the gap", "decode is the largest gap", COLORS["soft_blue"]),
        ("Phase 0.5", "Explain the gap", "full-context cost is measurable", COLORS["soft_orange"]),
        ("Phase 1", "Build a trusted baseline", "self path matches reference", COLORS["soft_green"]),
        ("Next", "Optimize one step at a time", "KV cache first", "#E0E7FF"),
    ]
    xs = [0.12, 0.38, 0.64, 0.88]
    for i, (title, main, sub, fill) in enumerate(nodes):
        x = xs[i]
        box = FancyBboxPatch(
            (x - 0.105, 0.38),
            0.21,
            0.30,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.1,
            edgecolor="#94A3B8",
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.text(x, 0.62, title, ha="center", va="center", weight="bold", color=COLORS["text"], fontsize=10)
        ax.text(x, 0.535, main, ha="center", va="center", color=COLORS["text"], fontsize=8.6)
        ax.text(x, 0.455, sub, ha="center", va="center", color=COLORS["muted"], fontsize=7.4)
        if i < len(nodes) - 1:
            arrow = FancyArrowPatch(
                (x + 0.11, 0.53),
                (xs[i + 1] - 0.11, 0.53),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#475569",
            )
            ax.add_patch(arrow)

    ax.text(0.5, 0.83, "From gap measurement to a credible optimization baseline", ha="center", fontsize=14, weight="bold", color=COLORS["text"])
    ax.text(0.5, 0.22, "Current formal update: the project has moved from observation to diagnosis and baseline validation.", ha="center", fontsize=9.3, color=COLORS["muted"])
    export(fig, "formal_fig01_evidence_chain")


def fig_three_engine_decode() -> None:
    phase = load_phase0_decode()
    llama = pd.read_csv(REPO / "benches" / "llama_cpp" / "llama_cpp_b9354_summary.csv")
    models = ["qwen2-0.5b", "qwen2-1.5b", "qwen2-7b"]
    labels = ["0.5B", "1.5B", "7B"]

    values = {
        "Stage 1 ref": [],
        "llama.cpp Metal": [],
        "mlx-lm": [],
    }
    for model in models:
        s1 = phase[(phase["engine"] == "silicon_w1ref") & (phase["model"] == model)]["value"]
        mlx = phase[(phase["engine"] == "mlx_lm") & (phase["model"] == model)]["value"]
        llm = llama[llama["model"] == model]["decode_tps_mean"]
        values["Stage 1 ref"].append(float(s1.iloc[0]) if not s1.empty else np.nan)
        values["llama.cpp Metal"].append(float(llm.iloc[0]) if not llm.empty else np.nan)
        values["mlx-lm"].append(float(mlx.iloc[0]) if not mlx.empty else np.nan)

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    x = np.arange(len(models))
    width = 0.24
    color_map = {"Stage 1 ref": COLORS["stage1"], "llama.cpp Metal": COLORS["llama"], "mlx-lm": COLORS["mlx"]}
    offsets = [-width, 0, width]
    for (name, vals), off in zip(values.items(), offsets):
        bars = ax.bar(x + off, vals, width, label=name, color=color_map[name], edgecolor="white", linewidth=0.8)
        for bar, val in zip(bars, vals):
            if np.isfinite(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val * 1.08, f"{val:.1f}", ha="center", va="bottom", fontsize=7.5)
            else:
                ax.text(bar.get_x() + bar.get_width() / 2, 1.1, "not run", ha="center", va="bottom", fontsize=7.2, color=COLORS["muted"], rotation=90)

    ax.set_yscale("log")
    ax.set_ylabel("Decode throughput (tok/s, log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("Decode gap remains the main production-engine difference")
    ax.legend(frameon=False, ncols=3, loc="upper center", bbox_to_anchor=(0.5, 1.16))
    ax.set_ylim(1, 500)
    ax.text(2 - width, 1.6, "Stage 1 7B omitted\n(memory boundary)", ha="center", va="bottom", fontsize=7, color=COLORS["muted"])
    despine(ax)
    export(fig, "formal_fig02_three_engine_decode")


def fig_prompt_length_gap() -> None:
    w05 = pd.read_csv(REPO / "benches" / "workload_length_scan_summary.csv")
    w15 = pd.read_csv(REPO / "benches" / "workload_length_scan_1p5b_summary.csv")
    df = pd.concat([w05, w15], ignore_index=True)

    rows = []
    for (model, prompt), group in df.groupby(["model", "requested_prompt_len"]):
        s1 = group[group["engine"] == "silicon_w1ref"]["decode_tps_mean"].iloc[0]
        mlx = group[group["engine"] == "mlx_lm"]["decode_tps_mean"].iloc[0]
        rows.append({"model": model, "prompt": prompt, "gap": mlx / s1, "s1": s1, "mlx": mlx})
    g = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=True)
    for ax, model, label in zip(axes, ["qwen2-0.5b", "qwen2-1.5b"], ["Qwen2 0.5B", "Qwen2 1.5B"]):
        sub = g[g["model"] == model].sort_values("prompt")
        ax.plot(sub["prompt"], sub["gap"], marker="o", color=COLORS["accent"], linewidth=2.2)
        ax.fill_between(sub["prompt"], sub["gap"], color=COLORS["accent"], alpha=0.08)
        for _, r in sub.iterrows():
            ax.text(r["prompt"], r["gap"] + 2.3, f"{r['gap']:.1f}x", ha="center", fontsize=7.7)
        ax.set_xscale("log", base=2)
        ax.set_xticks([64, 128, 512])
        ax.set_xticklabels(["64", "128", "512"])
        ax.set_xlabel("Prompt length (tokens)")
        ax.set_title(label)
        despine(ax)
    axes[0].set_ylabel("mlx-lm / Stage 1 decode ratio")
    fig.suptitle("Longer prompts amplify the Stage 1 decode gap", y=1.04, fontsize=11, weight="bold")
    export(fig, "formal_fig03_prompt_length_gap")


def fig_component_profile() -> None:
    df = pd.read_csv(REPO / "benches" / "stage1_decode_component_profile_summary.csv")
    order = ["qkv_projection", "attention_core", "attention_out_projection", "mlp", "elementwise", "lm_head", "outer_loop"]
    labels = {
        "qkv_projection": "QKV",
        "attention_core": "Attention",
        "attention_out_projection": "Attn out",
        "mlp": "MLP",
        "elementwise": "Elem.",
        "lm_head": "LM head",
        "outer_loop": "Python",
    }
    colors = ["#93C5FD", "#2563EB", "#60A5FA", "#0F766E", "#A7F3D0", "#D97706", "#94A3B8"]
    piv = df.pivot(index="prompt_len", columns="component", values="mean_ms").loc[[64, 128, 512], order]
    shares = df.pivot(index="prompt_len", columns="component", values="share_pct").loc[[64, 128, 512], order]

    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9), gridspec_kw={"width_ratios": [1.45, 1.0]})
    ax = axes[0]
    bottom = np.zeros(len(piv))
    x = np.arange(len(piv))
    for comp, color in zip(order, colors):
        vals = piv[comp].values
        ax.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.6, label=labels[comp])
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in piv.index])
    ax.set_xlabel("Prompt length (tokens)")
    ax.set_ylabel("Measured decode step time (ms)")
    ax.set_title("Full-context decode pays whole-layer cost")
    ax.legend(frameon=False, ncols=2, bbox_to_anchor=(1.02, 1.02), loc="upper left")
    despine(ax)

    ax2 = axes[1]
    s512 = shares.loc[512].sort_values(ascending=True)
    ax2.barh([labels[c] for c in s512.index], s512.values, color=[colors[order.index(c)] for c in s512.index])
    for y, v in enumerate(s512.values):
        ax2.text(v + 0.8, y, f"{v:.1f}%", va="center", fontsize=7.8)
    ax2.set_xlabel("Share at prompt 512 (%)")
    ax2.set_xlim(0, 58)
    ax2.set_title("Prompt 512 split")
    despine(ax2)
    export(fig, "formal_fig04_stage1_decode_profile")


def fig_self_reference_parity() -> None:
    s05 = pd.read_csv(REPO / "benches" / "stage1_self_baseline_compare_summary.csv")
    s15 = pd.read_csv(REPO / "benches" / "stage1_self_baseline_compare_1p5b_smoke_summary.csv")
    s15["model"] = "qwen2-1.5b"
    df = pd.concat([s05, s15], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.6), sharey=False)
    for ax, metric, title, unit in [
        (axes[0], "decode_tps_mean", "Decode throughput", "tok/s"),
        (axes[1], "wall_s_mean", "Wall time", "s"),
    ]:
        width = 0.34
        models = ["qwen2-0.5b", "qwen2-1.5b"]
        x = np.arange(len(models))
        self_vals = [df[(df["model"] == m) & (df["engine_label"] == "Self Stage 1")][metric].iloc[0] for m in models]
        ref_vals = [df[(df["model"] == m) & (df["engine_label"] == "Reference Stage 1")][metric].iloc[0] for m in models]
        ax.bar(x - width / 2, self_vals, width, label="Self", color=COLORS["llama"], edgecolor="white")
        ax.bar(x + width / 2, ref_vals, width, label="Reference", color=COLORS["stage1"], edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(["0.5B\nN=3", "1.5B\nN=2 smoke"])
        ax.set_ylabel(unit)
        ax.set_title(title)
        for i, (sv, rv) in enumerate(zip(self_vals, ref_vals)):
            ratio = sv / rv
            ypos = max(sv, rv) * 1.04
            ax.text(i, ypos, f"{ratio:.3f}x", ha="center", fontsize=7.8, color=COLORS["text"])
        despine(ax)
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Self Stage 1 matches the reference baseline", y=1.04, fontsize=11, weight="bold")
    export(fig, "formal_fig05_self_reference_parity")


def fig_reproducibility_summary() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.axis("off")
    rows = [
        ("Phase 0 evidence rows", "151", "engine, workload, metric, caveat"),
        ("Phase 1 baseline rows", "24", "self/reference parity checks"),
        ("Research-level rows", "175", "single auditable long table"),
    ]
    x0, y0 = 0.08, 0.62
    widths = [0.26, 0.26, 0.26]
    for i, (label, value, sub) in enumerate(rows):
        x = x0 + i * 0.30
        box = FancyBboxPatch(
            (x, y0 - 0.25),
            widths[i],
            0.38,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.0,
            edgecolor="#CBD5E1",
            facecolor=["#EFF6FF", "#ECFDF5", "#FFF7ED"][i],
        )
        ax.add_patch(box)
        ax.text(x + widths[i] / 2, y0 + 0.02, value, ha="center", va="center", fontsize=24, weight="bold", color=[COLORS["stage1"], COLORS["llama"], COLORS["mlx"]][i])
        ax.text(x + widths[i] / 2, y0 - 0.095, label, ha="center", va="center", fontsize=8.8, weight="bold", color=COLORS["text"])
        ax.text(x + widths[i] / 2, y0 - 0.18, sub, ha="center", va="center", fontsize=7.4, color=COLORS["muted"])
    ax.text(0.5, 0.88, "The formal update is backed by a reproducible result package", ha="center", fontsize=13, weight="bold", color=COLORS["text"])
    ax.text(0.5, 0.18, "The table keeps Phase 0 diagnosis separate from Phase 1 baseline validation.", ha="center", fontsize=9, color=COLORS["muted"])
    export(fig, "formal_fig06_reproducibility_summary")


def main() -> None:
    setup_style()
    fig_evidence_chain()
    fig_three_engine_decode()
    fig_prompt_length_gap()
    fig_component_profile()
    fig_self_reference_parity()
    fig_reproducibility_summary()
    print(f"Saved formal deck figures to {OUT}")


if __name__ == "__main__":
    main()
