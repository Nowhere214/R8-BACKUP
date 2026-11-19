"""Compare feedback control with fixed lengths, selected on validation only."""

import json
import random

from analyze_agent_study import summarize
from run_agent_study import ROOT, fingerprint, load_engines, policy_from, write_json
from tiny_llm.adaptive_decode import decode


def read_turns(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def measure(engine, tokenizer, turns, costs, modes, output, seed):
    rows = []
    with output.open("w") as stream:
        for rep in range(3):
            order = list(range(len(turns)))
            random.Random(seed + rep).shuffle(order)
            for position, index in enumerate(order):
                trace = turns[index]
                shuffled = list(modes)
                random.Random(seed + rep * 10000 + index).shuffle(shuffled)
                for name in shuffled:
                    mode, length = modes[name]
                    result = decode(
                        engine,
                        trace["prompt_tokens"],
                        384,
                        {tokenizer.eos_token_id},
                        mode=mode,
                        history=trace["history"],
                        draft_length=length,
                        policy=policy_from(costs),
                    )
                    row = {
                        "task_id": trace["task_id"],
                        "dataset": trace["dataset"],
                        "attempt": trace["attempt"],
                        "phase": trace["phase"],
                        "mode": name,
                        "rep": rep,
                        "exact": result.tokens == trace["result"]["tokens"],
                        **result.to_dict(),
                    }
                    rows.append(row)
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                if position % 10 == 0:
                    print(
                        f"{output.parent.name}: rep={rep + 1}/3 request={position + 1}/{len(turns)}",
                        flush=True,
                    )
    return rows


def main():
    source = ROOT / "benches/agent_feedback_study_20261021"
    status = json.loads((source / "suite_status.json").read_text())
    if status["stage"] != "all" or status["state"] != "complete":
        raise RuntimeError("Finish the serialized feedback suite first")
    output = source / "fixed_length_ablation"
    output.mkdir(exist_ok=False)
    calibration = (
        ROOT / "benches/agent_study_20261021_main/policy_calibration/turns.jsonl"
    )
    evaluation = source / "baseline/turns.jsonl"
    cost_file = source / "calibration_history.json"
    costs = json.loads(cost_file.read_text())
    fixed = {f"fixed_k{k}": ("warm_fixed", k) for k in (2, 4, 8, 16)}
    modes = {
        "baseline": ("baseline", 16),
        **fixed,
        "warm_feedback": ("warm_feedback", 16),
    }
    write_json(
        output / "protocol.json",
        {
            "purpose": "Independent controller comparison against all fixed lengths and a validation-selected fixed length",
            "registration": "Written after the earlier feedback study was started; no controller change or threshold tuning is made for this ablation",
            "reps": 3,
            "fixed_lengths": [2, 4, 8, 16],
            "selection": "Lowest total validation model wall time among fully exact fixed lengths; not chosen by evaluation performance",
            "validation_sha256": fingerprint(calibration),
            "evaluation_sha256": fingerprint(evaluation),
            "cost_sha256": fingerprint(cost_file),
            "decoder_sha256": fingerprint(ROOT / "src/tiny_llm/adaptive_decode.py"),
            "script_sha256": fingerprint(__file__),
            "scope": "Fixed-input replay on the same third cohort, not another independent task sample or actual closed-loop timing",
        },
    )
    engine, _, _, tokenizer, revision = load_engines()
    validation = read_turns(calibration)
    warm_trace = next(row for row in validation if row["history"])
    for mode, length in modes.values():
        decode(
            engine,
            warm_trace["prompt_tokens"],
            32,
            {tokenizer.eos_token_id},
            mode=mode,
            history=warm_trace["history"],
            draft_length=length,
            policy=policy_from(costs),
        )
    validation_dir = output / "validation"
    validation_dir.mkdir()
    validation_rows = measure(
        engine,
        tokenizer,
        validation,
        costs,
        {"baseline": modes["baseline"], **fixed},
        validation_dir / "replay.jsonl",
        6100,
    )
    validation_summaries = summarize(validation_rows)
    write_json(validation_dir / "summary.json", validation_summaries)
    eligible = [
        row
        for row in validation_summaries
        if row["dataset"] == "all"
        and row["subset"] == "all"
        and row["mode"] in fixed
        and row["sequence_preservation"]
    ]
    if not eligible:
        raise RuntimeError("No fixed length passed validation exactness")
    selected = min(eligible, key=lambda row: row["method_model_s"])["mode"]
    write_json(
        output / "selected_fixed.json", {"mode": selected, "split": "validation"}
    )
    outcomes = json.loads((source / "baseline/outcomes.json").read_text())
    evaluation_dir = output / "evaluation"
    evaluation_dir.mkdir()
    rows = measure(
        engine,
        tokenizer,
        read_turns(evaluation),
        costs,
        modes,
        evaluation_dir / "replay.jsonl",
        7100,
    )
    write_json(evaluation_dir / "summary.json", summarize(rows, outcomes))
    write_json(
        evaluation_dir / "controller_comparisons.json",
        [
            row
            for reference in fixed
            for row in summarize(rows, outcomes, reference_mode=reference)
            if row["mode"] == "warm_feedback"
        ],
    )
    write_json(
        output / "metadata.json",
        {
            "state": "complete",
            "model_revision": revision,
            "reps": 3,
            "validation_selected_fixed": selected,
            "modes": list(modes),
            "evaluation_requests": len(read_turns(evaluation)),
            "evaluation_runs": len(rows),
            "mismatches": sum(not row["exact"] for row in rows),
        },
    )
    print(json.loads((output / "metadata.json").read_text()), flush=True)


if __name__ == "__main__":
    main()
