"""Run a small natural-prompt agent pilot on cached Qwen2 and MBPP sanitized.

This is a workflow pilot, not an MBPP+ evaluation or an acceleration benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tiny_llm.agent_pilot import (  # noqa: E402
    extract_code,
    repeat_fraction,
    repair_feedback,
    run_tests,
    task_messages,
)


def generate(model, tokenizer, messages, max_tokens):
    import mlx.core as mx
    from tiny_llm.kv_cache import TinyKvFullCache

    rendered = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt = tokenizer.encode(rendered, add_special_tokens=False)
    cache = [TinyKvFullCache() for _ in range(model.num_hidden_layers)]
    stop_ids = set(getattr(tokenizer, "eos_token_ids", []) or [])
    if tokenizer.eos_token_id is not None:
        stop_ids.add(tokenizer.eos_token_id)
    generated = []
    latencies = []
    offset = 0
    current = prompt
    started = perf_counter()
    for _ in range(max_tokens):
        step_started = perf_counter()
        logits = model(mx.array(current, dtype=mx.int32)[None, :], offset, cache)
        token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(token)
        value = int(token.item())
        latencies.append(perf_counter() - step_started)
        generated.append(value)
        offset += len(current)
        current = [value]
        if value in stop_ids:
            break
    elapsed = perf_counter() - started
    text_ids = generated[:-1] if generated[-1] in stop_ids else generated
    return {
        "response": tokenizer.decode(text_ids),
        "prompt_token_ids": prompt,
        "output_token_ids": generated,
        "prompt_tokens": len(prompt),
        "generated_tokens": len(generated),
        "prefill_s": latencies[0],
        "decode_s": sum(latencies[1:]),
        "generation_wall_s": elapsed,
        "step_times_s": latencies,
        "stop_reason": "eos" if generated[-1] in stop_ids else "length",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--num-tasks", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-repairs", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=384)
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "benches/agent_pilot_20261015"
    )
    args = parser.parse_args()
    if args.max_tokens < 1 or args.max_repairs < 0 or args.num_tasks < 1:
        parser.error("Expected positive task/token limits and nonnegative repairs")

    data_bytes = args.data.read_bytes()
    tasks = sorted(
        (row for row in json.loads(data_bytes) if 11 <= row["task_id"] <= 510),
        key=lambda row: row["task_id"],
    )
    selected = random.Random(args.seed).sample(tasks, args.num_tasks)
    # Verify the test tool before paying model loading and generation costs.
    health = run_tests("def f(): return 4", ["assert f() == 4"], [])
    if not health["passed"]:
        raise RuntimeError(f"Test tool preflight failed: {health}")

    import mlx.core as mx
    from huggingface_hub import snapshot_download
    from mlx_lm import load
    from tiny_llm.models import dispatch_model

    model_id = "Qwen/Qwen2-1.5B-Instruct-MLX"
    snapshot = snapshot_download(model_id, local_files_only=True)
    print(
        f"Tasks selected before inference: {[task['task_id'] for task in selected]}",
        flush=True,
    )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_id,
        "model_revision": Path(snapshot).name,
        "engine": "SiliconLLM Stage 2 KV-cache greedy",
        "seed": args.seed,
        "dataset": "Google MBPP sanitized, original test split, not MBPP+",
        "dataset_url": "https://raw.githubusercontent.com/google-research/google-research/master/mbpp/sanitized-mbpp.json",
        "dataset_sha256": hashlib.sha256(data_bytes).hexdigest(),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "harness_sha256": hashlib.sha256(
            (ROOT / "src/tiny_llm/agent_pilot.py").read_bytes()
        ).hexdigest(),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "task_ids": [task["task_id"] for task in selected],
        "max_repairs": args.max_repairs,
        "max_tokens": args.max_tokens,
        "feedback_tests_per_task": 1,
        "kv_cache_across_turns": False,
        "isolation": "macOS sandbox-exec; network/writes/fork denied; home data restricted; CPU and wall-time limits",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )
    # Preserve exactly the inputs used, excluding reference solutions.
    inputs = [
        {key: task[key] for key in ("task_id", "prompt", "test_list", "test_imports")}
        for task in selected
    ]
    (args.output_dir / "tasks.json").write_text(json.dumps(inputs, indent=2) + "\n")
    load_start = perf_counter()
    base, tokenizer = load(snapshot)
    model = dispatch_model(model_id, base, week=2)
    load_s = perf_counter() - load_start
    generate(model, tokenizer, [{"role": "user", "content": "Say hi."}], 4)
    outcomes = []
    with (args.output_dir / "turns.jsonl").open("w", encoding="utf-8") as trace:
        for task in selected:
            messages = task_messages(task)
            prior_code = []
            task_start = perf_counter()
            turns = []
            for attempt in range(args.max_repairs + 1):
                stats = generate(model, tokenizer, messages, args.max_tokens)
                code = extract_code(stats["response"])
                visible = run_tests(code, task["test_list"][:1], task["test_imports"])
                code_ids = tokenizer.encode(code, add_special_tokens=False)
                row = {
                    "task_id": task["task_id"],
                    "attempt": attempt,
                    "phase": "generation" if attempt == 0 else "repair",
                    **stats,
                    "code": code,
                    "visible_test": visible,
                    "previous_code_repeat_4gram_fraction": repeat_fraction(
                        code_ids, prior_code
                    ),
                }
                trace.write(json.dumps(row) + "\n")
                trace.flush()
                turns.append(row)
                prior_code.append(code_ids)
                print(
                    f"task={task['task_id']} attempt={attempt} phase={row['phase']} visible={visible['status']} tokens={stats['generated_tokens']} model_s={stats['generation_wall_s']:.2f}",
                    flush=True,
                )
                if visible["passed"]:
                    break
                if visible["status"] == "tool_error":
                    raise RuntimeError(f"Test infrastructure failure: {visible}")
                messages.extend(
                    [
                        {"role": "assistant", "content": stats["response"]},
                        {"role": "user", "content": repair_feedback(visible)},
                    ]
                )
            agent_wall_s = perf_counter() - task_start
            # Held-out results never influence repair prompts or stopping.
            heldout = run_tests(code, task["test_list"][1:], task["test_imports"])
            if heldout["status"] == "tool_error":
                raise RuntimeError(f"Held-out infrastructure failure: {heldout}")
            outcome = {
                "task_id": task["task_id"],
                "attempts": len(turns),
                "initial_visible_pass": turns[0]["visible_test"]["passed"],
                "final_visible_pass": visible["passed"],
                "heldout": heldout,
                "final_all_tests_pass": visible["passed"] and heldout["passed"],
                "agent_wall_s": agent_wall_s,
                "model_wall_s": sum(turn["generation_wall_s"] for turn in turns),
                "visible_tool_wall_s": sum(
                    turn["visible_test"]["wall_s"] for turn in turns
                ),
            }
            outcomes.append(outcome)
            (args.output_dir / "outcomes.json").write_text(
                json.dumps(outcomes, indent=2) + "\n"
            )
    trace_rows = [
        json.loads(line)
        for line in (args.output_dir / "turns.jsonl").read_text().splitlines()
    ]
    repair_turns = [row for row in trace_rows if row["phase"] == "repair"]
    summary = {
        "tasks": len(outcomes),
        "llm_calls": len(trace_rows),
        "repair_calls": len(repair_turns),
        "initial_visible_passes": sum(row["initial_visible_pass"] for row in outcomes),
        "final_visible_passes": sum(row["final_visible_pass"] for row in outcomes),
        "final_all_tests_passes": sum(row["final_all_tests_pass"] for row in outcomes),
        "task_outcomes": outcomes,
        "model_load_s": load_s,
        "agent_wall_s": sum(row["agent_wall_s"] for row in outcomes),
        "model_wall_s": sum(row["model_wall_s"] for row in outcomes),
        "visible_tool_wall_s": sum(row["visible_tool_wall_s"] for row in outcomes),
        "heldout_tool_wall_s": sum(row["heldout"]["wall_s"] for row in outcomes),
        "repair_repeat_fraction_mean": sum(
            row["previous_code_repeat_4gram_fraction"] for row in repair_turns
        )
        / len(repair_turns)
        if repair_turns
        else None,
        "peak_mlx_memory_gb": mx.get_peak_memory() / 1024**3,
        "limitations": "Five-task workflow pilot by default; no speculative decoding, acceptance measurement, speedup comparison or MBPP+ score. Per-turn KV cache is rebuilt; model load/warmup and held-out evaluation are excluded from agent wall time.",
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
