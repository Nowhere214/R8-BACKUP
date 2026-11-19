"""Reproduce retained-cache perturbation and check the guarded decoder."""

import json

from run_agent_study import ROOT, load_engines, policy_from, write_json
from tiny_llm.adaptive_decode import HistoryIndex, decode


def main():
    import mlx.core as mx

    run = ROOT / "benches/agent_study_20261021_main"
    traces = [
        json.loads(line)
        for line in (run / "baseline/turns.jsonl").read_text().splitlines()
    ]
    trace = next(
        t for t in traces if t["task_id"] == "humaneval/141" and t["attempt"] == 0
    )
    engine, _, _, tokenizer, _ = load_engines()
    config = json.loads((run / "calibration_guarded.json").read_text())
    prompt = trace["prompt_tokens"]
    cache_a, cache_b = engine.cache(), engine.cache()
    token_a = engine.predict(prompt, 0, cache_a, last_only=True)[-1]
    token_b = engine.predict(prompt, 0, cache_b, last_only=True)[-1]
    prefix = [token_a]
    offset = len(prompt)
    for _ in range(37):
        token_a = engine.predict([token_a], offset, cache_a, last_only=True)[-1]
        token_b = engine.predict([token_b], offset, cache_b, last_only=True)[-1]
        assert token_a == token_b
        prefix.append(token_a)
        offset += 1
    snapshot = engine.snapshot(cache_b)
    candidates = HistoryIndex([prefix]).propose(prefix, 8)
    expected = engine.predict([token_a], offset, cache_a, last_only=True)[-1]
    predicted = engine.predict([token_b, *candidates], offset, cache_b)
    assert predicted[0] != candidates[0], "Expected the original first-token rejection"
    engine.rewind(cache_b, len(candidates))
    differences = []
    for a, b in zip(cache_a, cache_b):
        differences.append(
            {
                "key_max_abs": float(
                    mx.max(mx.abs(a.key_values[0] - b.key_values[0])).item()
                ),
                "value_max_abs": float(
                    mx.max(mx.abs(a.key_values[1] - b.key_values[1])).item()
                ),
            }
        )
    engine.restore(cache_b, snapshot)
    restored_prediction = engine.predict([token_b], offset, cache_b, last_only=True)[-1]
    restored_max_error = max(
        float(mx.max(mx.abs(a.key_values[j] - b.key_values[j])).item())
        for a, b in zip(cache_a, cache_b)
        for j in (0, 1)
    )
    results = []
    for mode in ("baseline", "global", "structure", "adaptive", "acceptance"):
        result = decode(
            engine,
            prompt,
            384,
            {tokenizer.eos_token_id},
            mode=mode,
            history=trace["history"],
            draft_length=16,
            policy=policy_from(config),
        )
        results.append(
            {
                "mode": mode,
                "exact": result.tokens == trace["result"]["tokens"],
                "recovery_steps": result.recovery_steps,
                "wall_s": result.wall_s,
            }
        )
    output = {
        "task_id": trace["task_id"],
        "attempt": 0,
        "verification_output_offset": 38,
        "first_prediction_matches_scalar": predicted[0] == expected,
        "retained_cache_differences": differences,
        "restored_prediction_matches": restored_prediction == expected,
        "restored_cache_max_abs": restored_max_error,
        "guarded_replays": results,
    }
    write_json(run / "rejection_recovery_audit.json", output)
    print(json.dumps(output, indent=2), flush=True)


if __name__ == "__main__":
    main()
