"""Small local API integration pilot, not a benchmark or agent-success study."""

import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
from importlib import import_module, metadata
import json
from pathlib import Path
import random
import resource
from time import perf_counter

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generation_stream, stream_generate

from tiny_llm.mlx_lm_history_api import HistoryAPIStats, history_stream_generate


TASKS = [
    {
        "id": "cold_generation",
        "kind": "cold_control",
        "source": "",
        "request": "Write a Python function count_positive(values) that counts values greater than zero.",
    },
    {
        "id": "unchanged_source",
        "kind": "copy_control",
        "source": "def sum_positive(values):\n    total = 0\n    for value in values:\n        if value > 0:\n            total += value\n    return total\n",
        "request": "Return this code unchanged. Do not add comments or explanations.",
    },
    {
        "id": "include_zero",
        "kind": "behavioral_edit",
        "source": "def count_positive(values):\n    count = 0\n    for value in values:\n        if value > 0:\n            count += 1\n    return count\n",
        "request": "Change count_positive so that zero is also counted. Keep the function name and the rest of the structure.",
    },
    {
        "id": "case_insensitive",
        "kind": "behavioral_edit",
        "source": "def unique_words(words):\n    seen = set()\n    result = []\n    for word in words:\n        if word not in seen:\n            seen.add(word)\n            result.append(word)\n    return result\n",
        "request": "Make duplicate detection case-insensitive, while keeping the original spelling of each first word and preserving order. Keep the function name.",
    },
    {
        "id": "inclusive_interval",
        "kind": "behavioral_edit",
        "source": "def filter_range(values, lower, upper):\n    result = []\n    for value in values:\n        if lower < value < upper:\n            result.append(value)\n    return result\n",
        "request": "Change filter_range so that both boundaries are included. Keep the function name and the rest of the structure.",
    },
]


def save(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def synchronize():
    mx.synchronize(generation_stream)
    mx.synchronize()


def run(model, tokenizer, task, mode, max_tokens):
    stats = HistoryAPIStats()
    synchronize()
    mx.clear_cache()
    mx.reset_peak_memory()
    started = perf_counter()
    if mode == "ordinary":
        responses = stream_generate(
            model, tokenizer, task["prompt_tokens"], max_tokens=max_tokens
        )
    else:
        responses = history_stream_generate(
            model,
            tokenizer,
            task["prompt_tokens"],
            history=task["history_tokens"],
            stats=stats,
            draft_length=16,
            max_tokens=max_tokens,
        )
    tokens, text = [], []
    finish, terminal = None, None
    try:
        for response in responses:
            text.append(response.text)
            if response.finish_reason is None:
                tokens.append(response.token)
            else:
                finish, terminal = response.finish_reason, response.token
                if finish == "stop":
                    tokens.append(terminal)
    finally:
        responses.close()
    synchronize()
    elapsed = perf_counter() - started
    return {
        "task": task["id"],
        "kind": task["kind"],
        "mode": mode,
        "wall_s": elapsed,
        "tokens": tokens,
        "text": "".join(text),
        "finish_reason": finish,
        "terminal_token": terminal,
        "peak_mlx_gib": mx.get_peak_memory() / 2**30,
        "stats": asdict(stats),
    }


def equivalent(a, b):
    return all(
        a[key] == b[key]
        for key in ("tokens", "text", "finish_reason", "terminal_token")
    )


def first_difference(a, b):
    for i, (x, y) in enumerate(zip(a["tokens"], b["tokens"])):
        if x != y:
            return {"position": i, "ordinary_token": x, "fixed_token": y}
    return {"ordinary_length": len(a["tokens"]), "fixed_length": len(b["tokens"])}


def main():
    experiment_started = perf_counter()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=192)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    module = import_module("mlx_lm.generate")
    model_path = args.model.resolve()
    manifest = {
        "study": "local full-streaming-wrapper integration pilot",
        "date": "2026-11-17",
        "host_timestamp": datetime.now().isoformat(),
        "model": str(model_path),
        "model_revision": model_path.name,
        "model_config": json.loads((model_path / "config.json").read_text()),
        "model_weight_sha256": sha(model_path / "model.safetensors"),
        "mlx_version": metadata.version("mlx"),
        "mlx_lm_version": metadata.version("mlx-lm"),
        "installed_generate_sha256": sha(Path(module.__file__)),
        "adapter_sha256": sha(
            Path(import_module("tiny_llm.mlx_lm_history_api").__file__)
        ),
        "runner_sha256": sha(Path(__file__)),
        "device": mx.metal.device_info(),
        "max_tokens": args.max_tokens,
        "draft_length": 16,
        "ngram": 4,
        "repeats": args.repeats,
        "shuffle_seed": 20261117,
        "sampling": "greedy",
        "recovery": "native KVCache trim only; no replay",
        "history": "supplied source code plus already emitted tokens within this request; no future tokens or baseline outputs",
        "wrapper": "stock stream_generate function code with isolated token-generator dispatch extension",
        "timing": "pretokenized prompt through complete stream consumption, detokenization and final synchronization; excludes loading, warmup and clear_cache",
        "scope": "five author-written feasibility cases, not public benchmark or full agent episodes",
        "installed_files_modified": False,
    }
    save(args.out / "manifest.json", manifest)
    print("Loading cached target", flush=True)
    load_started = perf_counter()
    model, tokenizer = load(str(model_path))
    load_s = perf_counter() - load_started
    tasks = []
    for task in TASKS:
        task = dict(task)
        user = task["request"]
        if task["source"]:
            user += "\n\nCurrent code:\n```python\n" + task["source"] + "```"
        messages = [
            {
                "role": "system",
                "content": "Return only the complete Python code in a single Python code block. Do not explain.",
            },
            {"role": "user", "content": user},
        ]
        rendered = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        task["prompt_tokens"] = tokenizer.encode(rendered, add_special_tokens=False)
        task["history_tokens"] = (
            [tokenizer.encode(task["source"], add_special_tokens=False)]
            if task["source"]
            else []
        )
        task["messages"] = messages
        tasks.append(task)
    save(args.out / "tasks.json", tasks)
    print("Warmup: ordinary and fixed history", flush=True)
    for mode in ("ordinary", "fixed_history"):
        run(model, tokenizer, tasks[1], mode, 48)
    validation = []
    for task in tasks:
        a = run(model, tokenizer, task, "ordinary", args.max_tokens)
        b = run(model, tokenizer, task, "fixed_history", args.max_tokens)
        exact = equivalent(a, b)
        validation.append(
            {
                "task": task["id"],
                "exact": exact,
                "ordinary": a,
                "fixed_history": b,
                "difference": None if exact else first_difference(a, b),
            }
        )
        print(
            f"Validation {task['id']}: exact={exact}, ordinary={a['wall_s']:.3f}s, fixed={b['wall_s']:.3f}s",
            flush=True,
        )
        save(args.out / "validation.json", validation)
    reference = {row["task"]: row["ordinary"] for row in validation}
    records = []
    rng = random.Random(20261117)
    for repeat in range(args.repeats):
        order = [
            (task, mode) for task in tasks for mode in ("ordinary", "fixed_history")
        ]
        rng.shuffle(order)
        for task, mode in order:
            record = run(model, tokenizer, task, mode, args.max_tokens)
            record["repeat"] = repeat
            record["exact_to_validation_ordinary"] = equivalent(
                reference[task["id"]], record
            )
            records.append(record)
            save(args.out / "timings.json", records)
            print(
                f"Repeat {repeat + 1} {task['id']} {mode}: {record['wall_s']:.3f}s exact={record['exact_to_validation_ordinary']}",
                flush=True,
            )
    groups = {}
    for group, ids in {
        "all_cases": {t["id"] for t in tasks},
        "history_cases": {t["id"] for t in tasks if t["source"]},
        "behavioral_edits": {t["id"] for t in tasks if t["kind"] == "behavioral_edit"},
        **{t["id"]: {t["id"]} for t in tasks},
    }.items():
        subset = [r for r in records if r["task"] in ids]
        totals = {
            mode: sum(r["wall_s"] for r in subset if r["mode"] == mode)
            for mode in ("ordinary", "fixed_history")
        }
        exact = all(r["exact_to_validation_ordinary"] for r in subset)
        groups[group] = {
            "task_count": len(ids),
            "calls_per_method": len(ids) * args.repeats,
            "summed_wall_s": totals,
            "time_ratio_ordinary_over_fixed": totals["ordinary"]
            / totals["fixed_history"],
            "all_outputs_exact": exact,
            "equivalent_output_speedup_claim_allowed": exact,
        }
    summary = {
        "validation_exact": sum(row["exact"] for row in validation),
        "validation_cases": len(validation),
        "timed_exact_to_reference": sum(
            r["exact_to_validation_ordinary"] for r in records
        ),
        "timed_calls": len(records),
        "peak_mlx_gib": max(r["peak_mlx_gib"] for r in records),
        "groups": groups,
        "load_s": load_s,
        "total_process_work_s": perf_counter() - experiment_started,
        "process_max_rss_gib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 2**30,
        "cold_control_interpretation": "Both modes use the same ordinary API. Any time difference is measurement noise, not drafting benefit.",
        "warning": "Small author-written pilot. Timing ratios for non-exact outputs are diagnostic, not equivalent-output speedups. No confidence intervals or task-success claims.",
    }
    save(args.out / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
