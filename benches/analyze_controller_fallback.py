"""Diagnose a request-level controller state that cannot resume speculation."""

import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

from tiny_llm.adaptive_decode import CostPolicy


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1] / "benches/agent_study_20261021_holdout"
    costs = json.loads((root / "calibration_history.json").read_text())
    directory = root / "replay"
    metadata = json.loads((directory / "metadata.json").read_text())
    rows = [
        json.loads(line)
        for line in (directory / "replay.jsonl").read_text().splitlines()
    ]
    by_request = defaultdict(dict)
    timings = defaultdict(list)
    for row in rows:
        key = (row["task_id"], row["attempt"], row["rep"])
        by_request[key][row["mode"]] = row
        timings[(row["task_id"], row["attempt"], row["mode"])].append(row["wall_s"])
    candidates = []
    for (task, attempt, rep), methods in by_request.items():
        if rep != 0:
            continue
        adaptive, fixed = methods["warm_adaptive"], methods["warm_fixed"]
        if not adaptive["decisions"] or not adaptive["exact"] or not fixed["exact"]:
            continue
        p = costs["initial_acceptance"]
        for event in adaptive["decisions"]:
            p = 0.7 * p + 0.3 * event["accepted"] / event["k"]
        lower_cost_policy = CostPolicy(
            costs["baseline_s"],
            {int(k): value for k, value in costs["verification_s"].items()},
            acceptance=p,
            lookup_s=0,
        )
        last = adaptive["decisions"][-1]
        ordinary_from = last["output_offset"] + last["accepted"] + 1
        later_accepts = sum(
            event["accepted"]
            for event in fixed["decisions"]
            if event["output_offset"] >= ordinary_from
        )
        if lower_cost_policy.choose(16) != 0 or later_accepts < 16:
            continue
        means = {
            mode: statistics.mean(timings[(task, attempt, mode)])
            for mode in ("warm_fixed", "warm_adaptive")
        }
        candidates.append(
            {
                "task_id": task,
                "attempt": attempt,
                "ordinary_from_output_offset": ordinary_from,
                "output_tokens": len(adaptive["tokens"]),
                "later_fixed_accepted_tokens": later_accepts,
                "final_acceptance_estimate": p,
                "choice_even_with_free_lookup": lower_cost_policy.choose(16),
                "controller_over_fixed_time_ratio": means["warm_adaptive"]
                / means["warm_fixed"],
                "fixed_mean_s": means["warm_fixed"],
                "controller_mean_s": means["warm_adaptive"],
            }
        )
    if not candidates:
        raise RuntimeError(
            "No verified absorbing fallback case with later useful matches"
        )
    candidates.sort(
        key=lambda row: row["controller_over_fixed_time_ratio"], reverse=True
    )
    chosen = candidates[0]
    result = {
        "repetitions": metadata["reps"],
        "cases": candidates,
        "selected_case": chosen,
        "selection": "Largest observed controller/fixed time ratio among absorbing cases; explanatory post-hoc selection",
        "mechanism": "Acceptance updates only after speculation; when every action is rejected even at zero lookup cost, ordinary steps cannot update that estimate",
    }
    (directory / "fallback_diagnosis.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2), flush=True)
    if args.no_plot:
        return
    methods = by_request[(chosen["task_id"], chosen["attempt"], 0)]
    feedback_path = (
        root.parent / "agent_feedback_study_20261021/known_case_regression.json"
    )
    feedback = json.loads(feedback_path.read_text()) if feedback_path.exists() else None
    if feedback and (
        not feedback["exact"]
        or feedback["task_id"] != chosen["task_id"]
        or feedback["attempt"] != chosen["attempt"]
    ):
        feedback = None
    if feedback:
        methods["warm_feedback"] = feedback
    skill = Path.home() / ".codex/skills/scientific-visualization"
    sys.path[:0] = [str(skill / "scripts"), str(skill / "assets")]
    from style_presets import apply_publication_style
    from figure_export import save_publication_figure
    from color_palettes import OKABE_ITO
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    apply_publication_style("default")
    fig, axes = plt.subplots(2, 1, figsize=(8, 5.8), sharex=True, layout="constrained")
    for mode, name, marker, color in (
        ("warm_fixed", "Fixed history", "o", OKABE_ITO["bluish_green"]),
        ("warm_adaptive", "Cost control", "^", OKABE_ITO["blue"]),
        ("warm_feedback", "Control + fallback feedback", "x", OKABE_ITO["vermillion"]),
    ):
        if mode not in methods:
            continue
        events = methods[mode]["decisions"]
        axes[0].scatter(
            [event["output_offset"] for event in events],
            [event["accepted"] for event in events],
            marker=marker,
            s=24,
            label=name,
            color=color,
            alpha=0.8,
        )
    axes[0].set_ylabel("Accepted candidate tokens")
    axes[0].set_title(
        f"A  {chosen['task_id']}, repair {chosen['attempt']}: later matches remain useful"
    )
    axes[0].legend(fontsize=9, loc="center left", bbox_to_anchor=(0.02, 0.62))
    x, y = [0], [costs["initial_acceptance"]]
    p = y[0]
    for event in methods["warm_adaptive"]["decisions"]:
        p = 0.7 * p + 0.3 * event["accepted"] / event["k"]
        x.append(event["output_offset"] + event["accepted"] + 1)
        y.append(p)
    x.append(chosen["output_tokens"])
    y.append(p)
    axes[1].step(x, y, where="post", color=OKABE_ITO["blue"], label="Original control")
    if feedback:
        events = [
            (event["output_offset"] + event["accepted"] + 1, "speculative", event)
            for event in feedback["decisions"]
        ] + [
            (event["output_offset"] + 1, "ordinary", event)
            for event in feedback["feedback_events"]
        ]
        x, y = [0], [costs["initial_acceptance"]]
        p = y[0]
        for position, kind, event in sorted(events, key=lambda item: item[0]):
            p = (
                event["p_after"]
                if kind == "ordinary"
                else 0.7 * p + 0.3 * event["accepted"] / event["k"]
            )
            x.append(position)
            y.append(p)
        x.append(len(feedback["tokens"]))
        y.append(p)
        axes[1].step(
            x, y, where="post", color=OKABE_ITO["vermillion"], label="With feedback"
        )
        axes[1].legend(fontsize=9, loc="lower right")
    for ax in axes:
        ax.axvspan(
            chosen["ordinary_from_output_offset"],
            chosen["output_tokens"],
            color="#D9D9D9",
            alpha=0.45,
            zorder=0,
        )
        ax.set_xlim(0, chosen["output_tokens"])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_xlabel("Generated output offset")
    axes[1].set_ylabel("Controller acceptance estimate")
    axes[1].set_title("B  Fallback feedback allows the estimate to recover")
    fig.suptitle(
        "Known development case; not an unseen performance result", fontsize=10
    )
    save_publication_figure(
        fig,
        directory / "figures/absorbing_fallback",
        formats=["png", "pdf", "svg"],
        dpi=600,
    )
    plt.close(fig)


if __name__ == "__main__":
    main()
