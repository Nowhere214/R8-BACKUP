"""Secondary native-engine study; its own target defines greedy equivalence."""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path
from time import perf_counter

from run_agent_study import (
    ROOT,
    MODEL_ID,
    encode,
    episode,
    fingerprint,
    policy_from,
    write_json,
)
from tiny_llm.adaptive_decode import decode
from tiny_llm.agent_pilot import task_messages
from tiny_llm.native_history import NativeHistoryEngine


def load_native():
    from huggingface_hub import snapshot_download
    from mlx_lm import load

    path = snapshot_download(MODEL_ID, local_files_only=True)
    model, tokenizer = load(path)
    return NativeHistoryEngine(model), tokenizer, Path(path).name


def selected_tasks(directory, per_dataset):
    tasks = json.loads((directory / "tasks.json").read_text())
    calibration = [task for task in tasks if task["split"] == "calibration"]
    evaluation = []
    for dataset in ("mbpp", "humaneval"):
        evaluation.extend(
            [
                task
                for task in tasks
                if task["split"] == "evaluation" and task["dataset"] == dataset
            ][:per_dataset]
        )
    return calibration, evaluation


def calibrate(args, engine, tokenizer, calibration):
    costs = {k: [] for k in (0, 2, 4, 8, 16)}
    accepted = proposed = 0
    comparisons = []
    traces = []
    eos = {tokenizer.eos_token_id}
    decode(engine, encode(tokenizer, task_messages(calibration[0])), 16, eos)
    for i, task in enumerate(calibration):
        captured = []
        episode(
            engine,
            tokenizer,
            task,
            args.max_tokens,
            "baseline",
            None,
            None,
            captured.append,
        )
        for trace in captured:
            traces.append(trace)
            result = decode(
                engine,
                trace["prompt_tokens"],
                args.max_tokens,
                eos,
                mode="structure",
                history=trace["history"],
                draft_length=16,
            )
            comparisons.append(
                {
                    "task_id": task["id"],
                    "attempt": trace["attempt"],
                    "exact": result.tokens == trace["result"]["tokens"],
                    "accepted": result.accepted,
                    "proposed": result.proposed,
                }
            )
            accepted += result.accepted
            proposed += result.proposed
        prompt = captured[0]["prompt_tokens"]
        tokens = captured[0]["result"]["tokens"]
        for k in costs:
            cache = engine.cache()
            token = engine.predict(prompt, 0, cache, last_only=True)[-1]
            trial = [token, *tokens[1 : k + 1]]
            trial += [token] * (k + 1 - len(trial))
            engine.predict(trial, len(prompt), cache)
            engine.rewind(cache, k + 1)
            for _ in range(5):
                start = perf_counter()
                engine.predict(trial, len(prompt), cache)
                costs[k].append(perf_counter() - start)
                engine.rewind(cache, k + 1)
        print(f"native calibration {i + 1}/{len(calibration)} {task['id']}", flush=True)
    write_json(args.out / "calibration_checks.json", comparisons)
    with (args.out / "calibration_turns.jsonl").open("w") as stream:
        for trace in traces:
            stream.write(json.dumps(trace) + "\n")
    repairs = [row for row in comparisons if row["attempt"] > 0]
    accepted = sum(row["accepted"] for row in repairs)
    proposed = sum(row["proposed"] for row in repairs)
    if not proposed:
        raise RuntimeError("Native history has no validation proposals")
    config = {
        "baseline_s": statistics.median(costs[0]),
        "verification_s": {
            k: statistics.median(values) for k, values in costs.items() if k
        },
        "cost_samples_s": costs,
        "initial_acceptance": min(0.99, max(0.05, (accepted + 1) / (proposed + 2))),
        "accepted": accepted,
        "proposed": proposed,
        "recovery_cost_included": True,
    }
    write_json(args.out / "calibration.json", config)
    checks = []
    for trace in traces:
        result = decode(
            engine,
            trace["prompt_tokens"],
            args.max_tokens,
            eos,
            mode="warm_adaptive",
            history=trace["history"],
            draft_length=16,
            policy=policy_from(config),
        )
        checks.append(
            {
                "task_id": trace["task_id"],
                "attempt": trace["attempt"],
                "exact": result.tokens == trace["result"]["tokens"],
            }
        )
    write_json(args.out / "controller_calibration_checks.json", checks)
    if not all(row["exact"] for row in checks):
        raise RuntimeError("Native controller failed independent calibration exactness")
    return config


def baseline(args, engine, tokenizer, evaluation):
    output = args.out / "baseline"
    output.mkdir()
    outcomes = []
    with (output / "turns.jsonl").open("w") as stream:

        def record(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()

        for i, task in enumerate(evaluation):
            outcomes.append(
                episode(
                    engine,
                    tokenizer,
                    task,
                    args.max_tokens,
                    "baseline",
                    None,
                    None,
                    record,
                )
            )
            write_json(output / "outcomes.json", outcomes)
            print(f"native baseline {i + 1}/{len(evaluation)} {task['id']}", flush=True)


def replay(args, engine, tokenizer, costs):
    traces = [
        json.loads(line)
        for line in (args.out / "baseline/turns.jsonl").read_text().splitlines()
    ]
    output = args.out / "replay"
    output.mkdir()
    modes = ["baseline", "warm_fixed", "warm_adaptive", "warm_acceptance"]
    for mode in modes:
        first = traces[0]
        decode(
            engine,
            first["prompt_tokens"],
            24,
            {tokenizer.eos_token_id},
            mode=mode,
            history=first["history"],
            draft_length=16,
            policy=policy_from(costs),
        )
    mismatches = []
    with (output / "replay.jsonl").open("w") as stream:
        for rep in range(args.reps):
            order = list(range(len(traces)))
            random.Random(100 + rep).shuffle(order)
            for number, index in enumerate(order):
                trace = traces[index]
                mode_order = list(modes)
                random.Random(1000 + rep * 10000 + index).shuffle(mode_order)
                for mode in mode_order:
                    result = decode(
                        engine,
                        trace["prompt_tokens"],
                        args.max_tokens,
                        {tokenizer.eos_token_id},
                        mode=mode,
                        history=trace["history"],
                        draft_length=16,
                        policy=policy_from(costs),
                    )
                    row = {
                        "task_id": trace["task_id"],
                        "dataset": trace["dataset"],
                        "attempt": trace["attempt"],
                        "phase": trace["phase"],
                        "mode": mode,
                        "rep": rep,
                        "exact": result.tokens == trace["result"]["tokens"],
                        **result.to_dict(),
                    }
                    if not row["exact"]:
                        mismatches.append(
                            (trace["task_id"], trace["attempt"], mode, rep)
                        )
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                if number % 10 == 0:
                    print(
                        f"native replay rep={rep + 1}/{args.reps} request={number + 1}/{len(traces)}",
                        flush=True,
                    )
    write_json(
        output / "metadata.json",
        {
            "modes": modes,
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "draft_length": 16,
            "calibration_sha256": fingerprint(args.out / "calibration.json"),
            "reference": "Native mlx-lm greedy target, not SiliconLLM greedy target",
            "mismatches": mismatches,
        },
    )
    print("native mismatches", mismatches, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source", type=Path, default=ROOT / "benches/agent_study_20261021_main"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "benches/agent_study_20261021_main/native_extension",
    )
    parser.add_argument("--per-dataset", type=int, default=10)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=384)
    args = parser.parse_args()
    args.out.mkdir(exist_ok=False)
    calibration, evaluation = selected_tasks(args.source, args.per_dataset)
    write_json(
        args.out / "protocol.json",
        {
            "engine": "native mlx-lm",
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "selected_before_native_inference": [task["id"] for task in evaluation],
            "selection": "First ten tasks per dataset in the frozen primary manifest; no native outcome filtering",
        },
    )
    engine, tokenizer, revision = load_native()
    costs = calibrate(args, engine, tokenizer, calibration)
    baseline(args, engine, tokenizer, evaluation)
    replay(args, engine, tokenizer, costs)
    write_json(
        args.out / "metadata.json",
        {
            "model_revision": revision,
            "script_sha256": fingerprint(__file__),
            "adapter_sha256": fingerprint(ROOT / "src/tiny_llm/native_history.py"),
            "decoder_sha256": fingerprint(ROOT / "src/tiny_llm/adaptive_decode.py"),
        },
    )


if __name__ == "__main__":
    main()
