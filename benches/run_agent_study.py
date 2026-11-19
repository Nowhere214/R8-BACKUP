"""Reproducible agent study: preparation, calibration, collection and paired replay.

Only calibration tasks inform frozen hardware costs. No held-out test feedback
or future generated content is used by historical proposals.
"""

from __future__ import annotations

import argparse
import ast
import doctest
import gzip
import hashlib
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from tiny_llm.adaptive_decode import CostPolicy, ModelEngine, decode  # noqa: E402
from tiny_llm.agent_pilot import extract_code, repair_feedback, run_tests, task_messages  # noqa: E402

MODEL_ID = "Qwen/Qwen2-1.5B-Instruct-MLX"


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare(args):
    mbpp = json.loads(args.mbpp.read_text())
    pilot_ids = {113, 61, 274, 257, 245}
    excluded = set()
    if getattr(args, "exclude_manifest", None):
        excluded = {
            task["id"]
            for task in json.loads(args.exclude_manifest.read_text())
            if task["split"] == "evaluation"
        }
    development = random.Random(42).sample(
        sorted(
            [t for t in mbpp if 511 <= t["task_id"] <= 600], key=lambda t: t["task_id"]
        ),
        10,
    )
    evaluation = random.Random(getattr(args, "seed", 20261021)).sample(
        sorted(
            [
                t
                for t in mbpp
                if 11 <= t["task_id"] <= 510
                and t["task_id"] not in pilot_ids
                and f"mbpp/{t['task_id']}" not in excluded
            ],
            key=lambda t: t["task_id"],
        ),
        args.mbpp_count,
    )

    def mbpp_task(row, split):
        return {
            "id": f"mbpp/{row['task_id']}",
            "dataset": "mbpp",
            "split": split,
            "prompt": row["prompt"],
            "test_list": row["test_list"][:1],
            "test_imports": row["test_imports"],
            "heldout_tests": row["test_list"][1:],
        }

    tasks = [mbpp_task(row, "calibration") for row in development] + [
        mbpp_task(row, "evaluation") for row in evaluation
    ]
    with gzip.open(args.humaneval, "rt") as source:
        human = [json.loads(line) for line in source]
    human = sorted(human, key=lambda row: row["task_id"])
    for row in human:
        entry = row["entry_point"]
        visible = f"assert callable({entry})"
        function = next(
            node
            for node in ast.parse(row["prompt"]).body
            if isinstance(node, ast.FunctionDef) and node.name == entry
        )
        description = ast.get_docstring(function) or ""
        try:
            examples = doctest.DocTestParser().get_examples(description)
        except ValueError:
            examples = []
        for example in examples:
            try:
                expression = ast.parse(example.source.strip(), mode="eval")
                expected = ast.literal_eval(example.want.strip())
            except (ValueError, SyntaxError):
                continue
            if isinstance(expression.body, ast.Call) and entry in example.source:
                visible = f"assert {ast.unparse(expression.body)} == {repr(expected)}"
                break
        if visible.startswith("assert callable"):
            for match in re.finditer(
                r"(?m)^\s*(.+?)\s*(?:==>|=>|==|->)\s*(.+?)\s*$", description
            ):
                try:
                    expression = ast.parse(match[1], mode="eval")
                    expected = ast.literal_eval(match[2].strip())
                except (SyntaxError, ValueError):
                    continue
                if isinstance(expression.body, ast.Call) and entry in match[1]:
                    visible = (
                        f"assert {ast.unparse(expression.body)} == {repr(expected)}"
                    )
                    break
        tasks.append(
            {
                "id": row["task_id"].replace("HumanEval", "humaneval"),
                "dataset": "humaneval",
                "split": "evaluation",
                "prompt": "Implement the complete Python function below, including its signature.\n\n"
                + row["prompt"],
                "test_list": [visible],
                "test_imports": ["from typing import *"],
                "heldout_tests": [row["test"] + f"\ncheck({entry})"],
            }
        )
    eligible_human = [
        t
        for t in tasks
        if t["dataset"] == "humaneval"
        and not t["test_list"][0].startswith("assert callable")
    ]
    available_human = [task for task in eligible_human if task["id"] not in excluded]
    tasks = [t for t in tasks if t["dataset"] == "mbpp"] + random.Random(
        getattr(args, "human_seed", 20261022)
    ).sample(available_human, args.humaneval_count)
    args.out.mkdir(parents=True, exist_ok=False)
    write_json(args.out / "tasks.json", tasks)
    write_json(
        args.out / "protocol.json",
        {
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "model": MODEL_ID,
            "mbpp_sha256": fingerprint(args.mbpp),
            "humaneval_sha256": fingerprint(args.humaneval),
            "mbpp_source": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
            "humaneval_source": "https://raw.githubusercontent.com/openai/human-eval/master/data/HumanEval.jsonl.gz",
            "pilot_ids_excluded_from_evaluation": sorted(pilot_ids),
            "max_tokens": args.max_tokens,
            "max_repairs": 2,
            "history": "Only current prompt or earlier code from the same task; cold cache between tasks",
            "controller": "EMA accepted-prefix ratio, frozen hardware costs, 5% margin; no evaluation-label tuning",
            "evaluation_counts": dict(
                Counter(t["dataset"] for t in tasks if t["split"] == "evaluation")
            ),
            "humaneval_eligible": len(eligible_human),
            "humaneval_available_after_exclusion": len(available_human),
            "excluded_evaluation_ids": sorted(excluded),
            "selection_seeds": {
                "mbpp": getattr(args, "seed", 20261021),
                "humaneval": getattr(args, "human_seed", 20261022),
            },
            "humaneval_inclusion": "Documented example convertible to an executable equality test; selected before inference",
            "calibration_task_ids": [
                t["id"] for t in tasks if t["split"] == "calibration"
            ],
        },
    )
    print(f"Prepared {len(tasks)} tasks", flush=True)


def load_engines(need_draft=False):
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from tiny_llm.models import dispatch_model

    path = snapshot_download(MODEL_ID, local_files_only=True)
    base, tokenizer = load(path)
    engine = ModelEngine(dispatch_model(MODEL_ID, base, week=2))
    draft = None
    if need_draft:
        draft_id = "Qwen/Qwen2-0.5B-Instruct-MLX"
        draft_base, _ = load(snapshot_download(draft_id, local_files_only=True))
        draft = ModelEngine(dispatch_model(draft_id, draft_base, week=2))
    return engine, ModelEngine(base, native=True), draft, tokenizer, Path(path).name


def encode(tokenizer, messages):
    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return tokenizer.encode(rendered, add_special_tokens=False)


def policy_from(config):
    return CostPolicy(
        config["baseline_s"],
        {int(k): v for k, v in config["verification_s"].items()},
        acceptance=config.get("initial_acceptance", 0.8),
    )


def calibrate_policy(args, tasks):
    engine, _, _, tokenizer, revision = load_engines()
    costs = json.loads((args.out / "calibration.json").read_text())
    directory = args.out / "policy_calibration"
    directory.mkdir(exist_ok=False)
    comparison = []
    calibration = [task for task in tasks if task["split"] == "calibration"]
    with (directory / "turns.jsonl").open("w") as stream:
        for i, task in enumerate(calibration):
            captured = []

            def record(row):
                captured.append(row)
                stream.write(json.dumps(row) + "\n")
                stream.flush()

            episode(
                engine, tokenizer, task, args.max_tokens, "baseline", None, None, record
            )
            for trace in captured:
                result = decode(
                    engine,
                    trace["prompt_tokens"],
                    args.max_tokens,
                    {tokenizer.eos_token_id},
                    mode="structure",
                    history=trace["history"],
                    draft_length=16,
                )
                row = {
                    "task_id": task["id"],
                    "attempt": trace["attempt"],
                    "exact": result.tokens == trace["result"]["tokens"],
                    **result.to_dict(),
                }
                comparison.append(row)
            print(f"policy calibration {i + 1}/10 {task['id']}", flush=True)
    accepted = sum(row["accepted"] for row in comparison)
    proposed = sum(row["proposed"] for row in comparison)
    if not proposed or not all(row["exact"] for row in comparison):
        write_json(directory / "history_results.json", comparison)
        raise RuntimeError(
            "Guarded policy calibration failed its coverage or exactness gate"
        )
    prior = min(0.99, max(0.05, (accepted + 1) / (proposed + 2)))
    costs.update(
        initial_acceptance=prior,
        prior_source="Independent validation trajectories, Laplace-smoothed accepted/proposed ratio",
        recovery_cost_included=True,
        guarded_revision=True,
        model_revision=revision,
        calibration_proposed=proposed,
        calibration_accepted=accepted,
    )
    write_json(directory / "history_results.json", comparison)
    write_json(args.out / "calibration_guarded.json", costs)
    print(
        {
            "prior": prior,
            "accepted": accepted,
            "proposed": proposed,
            "exact": len(comparison),
        },
        flush=True,
    )


def episode(engine, tokenizer, task, max_tokens, mode, costs, draft_engine, record):
    eos = {tokenizer.eos_token_id}
    messages = task_messages(task)
    history = []
    turns = []
    start = perf_counter()
    for attempt in range(3):
        prompt = encode(tokenizer, messages)
        # Reset controller state per request, with the same calibrated prior.
        result = decode(
            engine,
            prompt,
            max_tokens,
            eos,
            mode=mode,
            history=history,
            draft_engine=draft_engine,
            draft_length=2 if mode == "small" else 16,
            policy=policy_from(costs) if costs else None,
        )
        response = tokenizer.decode(
            result.tokens[:-1] if result.stop_reason == "eos" else result.tokens
        )
        code = extract_code(response)
        visible = run_tests(code, task["test_list"], task["test_imports"])
        row = {
            "task_id": task["id"],
            "dataset": task["dataset"],
            "attempt": attempt,
            "phase": "generation" if attempt == 0 else "repair",
            "mode": mode,
            "prompt_tokens": prompt,
            "history": [list(h) for h in history],
            "result": result.to_dict(),
            "response": response,
            "code": code,
            "visible": visible,
        }
        record(row)
        turns.append(row)
        history.append(tokenizer.encode(code, add_special_tokens=False))
        if visible["passed"]:
            break
        if visible["status"] == "tool_error":
            raise RuntimeError(f"Test infrastructure failure {task['id']}: {visible}")
        messages.extend(
            [
                {"role": "assistant", "content": response},
                {"role": "user", "content": repair_feedback(visible)},
            ]
        )
    agent_s = perf_counter() - start
    heldout = run_tests(code, task["heldout_tests"], task["test_imports"])
    if heldout["status"] == "tool_error":
        raise RuntimeError(f"Held-out infrastructure failure {task['id']}: {heldout}")
    return {
        "task_id": task["id"],
        "dataset": task["dataset"],
        "mode": mode,
        "calls": len(turns),
        "initial_visible_pass": turns[0]["visible"]["passed"],
        "passed": visible["passed"] and heldout["passed"],
        "heldout": heldout,
        "agent_s": agent_s,
        "model_s": sum(t["result"]["wall_s"] for t in turns),
        "duplicate_repairs": sum(
            t["code"] == turns[i - 1]["code"] for i, t in enumerate(turns) if i
        ),
        "length_stops": sum(t["result"]["stop_reason"] == "length" for t in turns),
    }


def calibrate(args, tasks):
    engine, native, _, tokenizer, revision = load_engines()
    rows = []
    times = {k: [] for k in (0, 2, 4, 8, 16)}
    calibration = [t for t in tasks if t["split"] == "calibration"]
    eos = {tokenizer.eos_token_id}
    for i, task in enumerate(calibration):
        prompt = encode(tokenizer, task_messages(task))
        if i == 0:
            decode(engine, prompt, 8, eos)
        a = decode(engine, prompt, 64, eos)
        b = decode(native, prompt, 64, eos)
        rows.append(
            {
                "task_id": task["id"],
                "exact_native": a.tokens == b.tokens,
                "silicon_tokens": a.tokens,
                "native_tokens": b.tokens,
            }
        )
        for k in times:
            cache = engine.cache()
            token = engine.predict(prompt, 0, cache, last_only=True)[-1]
            offset = len(prompt)
            trial = [token] + a.tokens[1 : k + 1]
            trial += [token] * (k + 1 - len(trial))
            engine.predict(trial, offset, cache)
            engine.rewind(cache, k + 1)
            for _ in range(5):
                start = perf_counter()
                engine.predict(trial, offset, cache)
                times[k].append(perf_counter() - start)
                engine.rewind(cache, k + 1)
        print(
            f"calibration {i + 1}/{len(calibration)} {task['id']} exact_native={rows[-1]['exact_native']}",
            flush=True,
        )
    import statistics

    config = {
        "baseline_s": statistics.median(times[0]),
        "verification_s": {k: statistics.median(times[k]) for k in times if k},
        "cost_samples_s": times,
        "model_revision": revision,
        "calibration_ids": [t["id"] for t in calibration],
        "native_audit": rows,
        "initial_acceptance": 0.8,
        "ema_weight": 0.3,
        "margin": 0.05,
    }
    write_json(args.out / "calibration.json", config)
    print(
        {
            "baseline_s": config["baseline_s"],
            "verification_s": config["verification_s"],
        },
        flush=True,
    )


def collect(args, tasks):
    engine, _, _, tokenizer, revision = load_engines()
    output = args.out / "baseline"
    output.mkdir(exist_ok=False)
    evaluations = [t for t in tasks if t["split"] == "evaluation"]
    if args.limit:
        evaluations = evaluations[: args.limit]
    outcomes = []
    first = encode(tokenizer, task_messages(evaluations[0]))
    decode(engine, first, 8, {tokenizer.eos_token_id})
    with (output / "turns.jsonl").open("w") as stream:

        def record(row):
            stream.write(json.dumps(row) + "\n")
            stream.flush()

        for i, task in enumerate(evaluations):
            outcome = episode(
                engine, tokenizer, task, args.max_tokens, "baseline", None, None, record
            )
            outcomes.append(outcome)
            write_json(output / "outcomes.json", outcomes)
            print(
                f"baseline {i + 1}/{len(evaluations)} {task['id']} pass={outcome['passed']} calls={outcome['calls']} agent_s={outcome['agent_s']:.2f}",
                flush=True,
            )
    write_json(
        output / "metadata.json",
        {
            "model_revision": revision,
            "max_tokens": args.max_tokens,
            "script_sha256": fingerprint(__file__),
            "decoder_sha256": fingerprint(ROOT / "src/tiny_llm/adaptive_decode.py"),
        },
    )


def replay(args, tasks):
    costs = json.loads((args.out / args.cost_file).read_text())
    traces = [
        json.loads(line)
        for line in (args.out / "baseline/turns.jsonl").read_text().splitlines()
    ]
    if args.limit:
        ids = list(dict.fromkeys(t["task_id"] for t in traces))[: args.limit]
        traces = [t for t in traces if t["task_id"] in ids]
    if args.per_dataset:
        ids = set()
        for dataset in ("mbpp", "humaneval"):
            ids.update(
                list(
                    dict.fromkeys(
                        t["task_id"] for t in traces if t["dataset"] == dataset
                    )
                )[: args.per_dataset]
            )
        traces = [t for t in traces if t["task_id"] in ids]
    modes = args.modes.split(",")
    engine, native, draft, tokenizer, revision = load_engines("small" in modes)
    eos = {tokenizer.eos_token_id}
    for mode in modes:
        chosen = native if mode == "native" else engine
        warm_task = next(t for t in tasks if t["split"] == "calibration")
        warm_history = [costs["native_audit"][0]["silicon_tokens"]]
        decode(
            chosen,
            encode(tokenizer, task_messages(warm_task)),
            32,
            eos,
            mode="baseline" if mode == "native" else mode,
            history=warm_history,
            draft_engine=draft,
            draft_length=2 if mode == "small" else args.draft_length,
            policy=policy_from(costs),
        )
    output = args.out / args.label
    output.mkdir(exist_ok=False)
    records = []
    with (output / "replay.jsonl").open("w") as stream:
        for rep in range(args.reps):
            order = list(range(len(traces)))
            random.Random(100 + rep).shuffle(order)
            for number, index in enumerate(order):
                trace = traces[index]
                shuffled_modes = list(modes)
                random.Random(1000 + rep * 10000 + index).shuffle(shuffled_modes)
                for mode in shuffled_modes:
                    chosen = native if mode == "native" else engine
                    active_mode = "baseline" if mode == "native" else mode
                    result = decode(
                        chosen,
                        trace["prompt_tokens"],
                        args.max_tokens,
                        eos,
                        mode=active_mode,
                        history=trace["history"],
                        draft_engine=draft,
                        draft_length=2 if mode == "small" else args.draft_length,
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
                    stream.write(json.dumps(row) + "\n")
                    stream.flush()
                    records.append(row)
                if number % 10 == 0:
                    print(
                        f"replay rep={rep + 1}/{args.reps} request={number + 1}/{len(traces)} {trace['task_id']}",
                        flush=True,
                    )
    write_json(
        output / "metadata.json",
        {
            "model_revision": revision,
            "modes": modes,
            "reps": args.reps,
            "max_tokens": args.max_tokens,
            "draft_length": args.draft_length,
            "calibration_sha256": fingerprint(args.out / args.cost_file),
            "script_sha256": fingerprint(__file__),
            "decoder_sha256": fingerprint(ROOT / "src/tiny_llm/adaptive_decode.py"),
        },
    )
    print(
        "mismatches",
        dict(Counter(r["mode"] for r in records if not r["exact"])),
        flush=True,
    )


def closed_loop(args, tasks):
    costs = json.loads((args.out / args.cost_file).read_text())
    output = args.out / args.label
    output.mkdir(exist_ok=False)
    modes = args.modes.split(",")
    engine, _, draft, tokenizer, revision = load_engines("small" in modes)
    evaluations = [t for t in tasks if t["split"] == "evaluation"]
    if args.limit:
        evaluations = evaluations[: args.limit]
    if args.per_dataset:
        evaluations = [
            task
            for dataset in ("mbpp", "humaneval")
            for task in [t for t in evaluations if t["dataset"] == dataset][
                : args.per_dataset
            ]
        ]
    baseline_traces = [
        json.loads(line)
        for line in (args.out / "baseline/turns.jsonl").read_text().splitlines()
    ]
    expected = {
        (t["task_id"], t["attempt"]): t["result"]["tokens"] for t in baseline_traces
    }
    outcomes = []
    warm_task = next(task for task in tasks if task["split"] == "calibration")
    warm_history = [costs["native_audit"][0]["silicon_tokens"]]
    for mode in modes:
        decode(
            engine,
            encode(tokenizer, task_messages(warm_task)),
            32,
            {tokenizer.eos_token_id},
            mode=mode,
            history=warm_history,
            draft_engine=draft,
            draft_length=2 if mode == "small" else 16,
            policy=policy_from(costs),
        )
    with (output / "turns.jsonl").open("w") as stream:
        for rep in range(args.reps):
            for i, task in enumerate(evaluations):
                order = list(modes)
                random.Random(rep * 1000 + i).shuffle(order)
                for mode in order:
                    exacts = []

                    def record(row):
                        row["rep"] = rep
                        row["exact"] = (
                            expected.get((row["task_id"], row["attempt"]))
                            == row["result"]["tokens"]
                        )
                        exacts.append(row["exact"])
                        stream.write(json.dumps(row) + "\n")
                        stream.flush()

                    outcome = episode(
                        engine,
                        tokenizer,
                        task,
                        args.max_tokens,
                        mode,
                        costs,
                        draft,
                        record,
                    )
                    outcome.update(
                        rep=rep,
                        exact=all(exacts),
                        expected_calls=sum(
                            t["task_id"] == task["id"] for t in baseline_traces
                        ),
                    )
                    outcome["exact"] &= outcome["calls"] == outcome["expected_calls"]
                    outcomes.append(outcome)
                    write_json(output / "outcomes.json", outcomes)
                print(
                    f"closed rep={rep + 1}/{args.reps} task={i + 1}/{len(evaluations)} {task['id']}",
                    flush=True,
                )
    write_json(
        output / "metadata.json",
        {
            "model_revision": revision,
            "modes": modes,
            "reps": args.reps,
            "task_ids": [task["id"] for task in evaluations],
            "max_tokens": args.max_tokens,
            "calibration_sha256": fingerprint(args.out / args.cost_file),
            "script_sha256": fingerprint(__file__),
            "decoder_sha256": fingerprint(ROOT / "src/tiny_llm/adaptive_decode.py"),
            "agent_time_excludes": [
                "model loading",
                "warmup",
                "final held-out grading",
            ],
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "calibrate",
            "calibrate-policy",
            "collect",
            "replay",
            "closed",
        ],
    )
    parser.add_argument(
        "--out", type=Path, default=ROOT / "benches/agent_study_20261021"
    )
    parser.add_argument(
        "--mbpp", type=Path, default=Path("/private/tmp/siliconllm-sanitized-mbpp.json")
    )
    parser.add_argument(
        "--humaneval",
        type=Path,
        default=Path("/private/tmp/siliconllm-humaneval.jsonl.gz"),
    )
    parser.add_argument("--mbpp-count", type=int, default=60)
    parser.add_argument("--humaneval-count", type=int, default=30)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-dataset", type=int, default=0)
    parser.add_argument(
        "--modes", default="baseline,small,global,structure,adaptive,acceptance,native"
    )
    parser.add_argument("--draft-length", type=int, default=16)
    parser.add_argument("--label", default="replay")
    parser.add_argument("--cost-file", default="calibration.json")
    parser.add_argument("--exclude-manifest", type=Path)
    parser.add_argument("--seed", type=int, default=20261021)
    parser.add_argument("--human-seed", type=int, default=20261022)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare(args)
        return
    tasks = json.loads((args.out / "tasks.json").read_text())
    {
        "calibrate": calibrate,
        "calibrate-policy": calibrate_policy,
        "collect": collect,
        "replay": replay,
        "closed": closed_loop,
    }[args.command](args, tasks)


if __name__ == "__main__":
    main()
