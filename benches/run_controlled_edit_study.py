"""Controlled editing sensitivity study, separate from the public task cohorts."""

import json
import random
from pathlib import Path
from time import perf_counter

from analyze_agent_study import METHOD_COLORS, summarize
from run_agent_study import (
    ROOT,
    encode,
    fingerprint,
    load_engines,
    policy_from,
    write_json,
)
from tiny_llm.adaptive_decode import decode
from tiny_llm.agent_pilot import extract_code, repair_feedback, run_tests, task_messages


def edit_episode(engine, tokenizer, task, record):
    prompt_task = {
        "prompt": "Update the existing Python file for this change.\n\n"
        + task["change"]
        + "\n\nExisting file:\n```python\n"
        + task["initial_code"]
        + "\n```",
        "test_list": [task["visible"]],
    }
    messages = task_messages(prompt_task)
    history = [tokenizer.encode(task["initial_code"], add_special_tokens=False)]
    turns = []
    start = perf_counter()
    for attempt in range(3):
        prompt = encode(tokenizer, messages)
        result = decode(engine, prompt, 384, {tokenizer.eos_token_id})
        response = tokenizer.decode(
            result.tokens[:-1] if result.stop_reason == "eos" else result.tokens
        )
        code = extract_code(response)
        visible = run_tests(code, [task["visible"]], [])
        if visible["status"] == "tool_error":
            raise RuntimeError(visible)
        row = {
            "task_id": task["id"],
            "dataset": "controlled",
            "attempt": attempt,
            "phase": "generation" if attempt == 0 else "repair",
            "mode": "baseline",
            "prompt_tokens": prompt,
            "history": [list(tokens) for tokens in history],
            "response": response,
            "code": code,
            "result": result.to_dict(),
            "visible": visible,
        }
        record(row)
        turns.append(row)
        if visible["passed"]:
            break
        history.append(tokenizer.encode(code, add_special_tokens=False))
        messages.extend(
            [
                {"role": "assistant", "content": response},
                {"role": "user", "content": repair_feedback(visible)},
            ]
        )
    agent_s = perf_counter() - start
    final = run_tests(code, task["old_tests"] + task["heldout"], [])
    if final["status"] == "tool_error":
        raise RuntimeError(final)
    return {
        "task_id": task["id"],
        "dataset": "controlled",
        "mode": "baseline",
        "calls": len(turns),
        "passed": visible["passed"] and final["passed"],
        "agent_s": agent_s,
        "model_s": sum(row["result"]["wall_s"] for row in turns),
        "changed_text": code != task["initial_code"],
        "heldout": final,
    }


def plot_summary(output, summaries):
    import sys

    skill = Path.home() / ".codex/skills/scientific-visualization"
    sys.path[:0] = [str(skill / "scripts"), str(skill / "assets")]
    from style_presets import apply_publication_style
    from figure_export import save_publication_figure
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    points = {
        row["mode"]: row
        for row in summaries
        if row["dataset"] == "all" and row["subset"] == "all"
    }
    methods = (
        "baseline",
        "warm_fixed",
        "warm_adaptive",
        "warm_feedback",
        "warm_acceptance",
    )
    names = ("Ordinary", "History fixed", "Cost control", "Feedback", "Acceptance rule")
    apply_publication_style("default")
    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    for i, mode in enumerate(methods):
        row = points[mode]
        ax.bar(
            i,
            row["speedup"],
            color=METHOD_COLORS[mode],
            hatch="" if row["sequence_preservation"] else "///",
        )
        ax.errorbar(
            i,
            row["speedup"],
            yerr=[
                [max(0, row["speedup"] - row["ci95_low"])],
                [max(0, row["ci95_high"] - row["speedup"])],
            ],
            fmt="none",
            color="black",
            capsize=3,
        )
    ax.axhline(1, color="black", linestyle="--", linewidth=0.8)
    ax.set_xticks(range(len(methods)), names)
    ax.set_ylabel("Model wall-time ratio to ordinary decoding (x)")
    ax.set_title(
        "Controlled code edits: ten authored tasks\nThree repetitions, 95% task-bootstrap intervals"
    )
    save_publication_figure(
        fig, output / "controlled_edits", formats=["png", "pdf", "svg"], dpi=600
    )
    plt.close(fig)


def main():
    output = ROOT / "benches/controlled_edit_study_20261021"
    output.mkdir(exist_ok=False)
    task_file = ROOT / "benches/controlled_code_edits.json"
    tasks = json.loads(task_file.read_text())
    conditions = []
    for task in tasks:
        old = run_tests(task["initial_code"], task["old_tests"], [])
        new = run_tests(task["initial_code"], [task["visible"]], [])
        conditions.append(
            {
                "task_id": task["id"],
                "old_contract_pass": old["passed"],
                "new_contract_already_passed": new["passed"],
            }
        )
    write_json(output / "preconditions.json", conditions)
    if not all(
        row["old_contract_pass"] and not row["new_contract_already_passed"]
        for row in conditions
    ):
        raise RuntimeError("Every task must require a real behavioral edit")
    costs_file = ROOT / "benches/agent_study_20261021_holdout/calibration_history.json"
    costs = json.loads(costs_file.read_text())
    write_json(
        output / "protocol.json",
        {
            "cohort": "Ten manually authored controlled edits, not a public benchmark",
            "task_sha256": fingerprint(task_file),
            "calibration_sha256": fingerprint(costs_file),
            "reps": 3,
            "max_tokens": 384,
            "max_repairs": 2,
            "history": "Provided source file and earlier generated code from the same task only",
            "test_contract": "Initial source passes old tests and fails the new visible test; final source must pass both contracts",
            "interpretation": "Sensitivity check for meaningful edits, reported separately from MBPP and HumanEval",
        },
    )
    engine, _, _, tokenizer, revision = load_engines()
    development = next(
        task
        for task in json.loads(
            (ROOT / "benches/agent_study_20261021_holdout/tasks.json").read_text()
        )
        if task["split"] == "calibration"
    )
    decode(
        engine,
        encode(tokenizer, task_messages(development)),
        8,
        {tokenizer.eos_token_id},
    )
    baseline_dir = output / "baseline"
    baseline_dir.mkdir()
    traces, outcomes = [], []
    with (baseline_dir / "turns.jsonl").open("w") as stream:

        def record(row):
            traces.append(row)
            stream.write(json.dumps(row) + "\n")
            stream.flush()

        for i, task in enumerate(tasks):
            outcome = edit_episode(engine, tokenizer, task, record)
            outcomes.append(outcome)
            write_json(baseline_dir / "outcomes.json", outcomes)
            print(
                f"controlled baseline {i + 1}/{len(tasks)} pass={outcome['passed']}",
                flush=True,
            )
    modes = [
        "baseline",
        "warm_fixed",
        "warm_adaptive",
        "warm_feedback",
        "warm_acceptance",
    ]
    for mode in modes:
        trace = traces[0]
        decode(
            engine,
            trace["prompt_tokens"],
            24,
            {tokenizer.eos_token_id},
            mode=mode,
            history=trace["history"],
            draft_length=16,
            policy=policy_from(costs),
        )
    directory = output / "replay"
    directory.mkdir()
    rows = []
    with (directory / "replay.jsonl").open("w") as stream:
        for rep in range(3):
            order = list(range(len(traces)))
            random.Random(100 + rep).shuffle(order)
            for index in order:
                trace = traces[index]
                mode_order = list(modes)
                random.Random(1000 + rep * 10000 + index).shuffle(mode_order)
                for mode in mode_order:
                    result = decode(
                        engine,
                        trace["prompt_tokens"],
                        384,
                        {tokenizer.eos_token_id},
                        mode=mode,
                        history=trace["history"],
                        draft_length=16,
                        policy=policy_from(costs),
                    )
                    row = {
                        "task_id": trace["task_id"],
                        "dataset": "controlled",
                        "attempt": trace["attempt"],
                        "phase": trace["phase"],
                        "mode": mode,
                        "rep": rep,
                        "exact": result.tokens == trace["result"]["tokens"],
                        **result.to_dict(),
                    }
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
            print(f"controlled replay repetition {rep + 1}/3", flush=True)
    summaries = summarize(rows, outcomes)
    write_json(directory / "summary.json", summaries)
    write_json(
        directory / "controller_comparisons.json",
        [
            row
            for row in summarize(rows, outcomes, reference_mode="warm_fixed")
            if row["mode"] in {"warm_adaptive", "warm_acceptance", "warm_feedback"}
        ],
    )
    write_json(
        directory / "metadata.json",
        {
            "model_revision": revision,
            "modes": modes,
            "reps": 3,
            "max_tokens": 384,
            "script_sha256": fingerprint(__file__),
            "decoder_sha256": fingerprint(ROOT / "src/tiny_llm/adaptive_decode.py"),
        },
    )
    plot_summary(output, summaries)
    print(
        {
            "passed": sum(row["passed"] for row in outcomes),
            "tasks": len(tasks),
            "runs": len(rows),
            "mismatches": sum(not row["exact"] for row in rows),
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
