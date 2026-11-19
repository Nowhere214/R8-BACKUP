"""A regression audit: identical accepted tokens need not imply identical FP16 KV."""

import json

from run_agent_study import ROOT, load_engines, write_json
from tiny_llm.adaptive_decode import HistoryIndex


def main():
    import mlx.core as mx

    directory = ROOT / "benches/agent_study_20261021_main"
    trace = next(
        row
        for row in (
            json.loads(line)
            for line in (directory / "baseline/turns.jsonl").read_text().splitlines()
        )
        if row["task_id"] == "mbpp/260" and row["attempt"] == 0
    )
    engine, _, _, _, _ = load_engines()
    prompt = trace["prompt_tokens"]
    a, b = engine.cache(), engine.cache()
    token_a = engine.predict(prompt, 0, a, last_only=True)[-1]
    token_b = engine.predict(prompt, 0, b, last_only=True)[-1]
    prefix = [token_a]
    offset = len(prompt)
    for _ in range(93):
        token_a = engine.predict([token_a], offset, a, last_only=True)[-1]
        token_b = engine.predict([token_b], offset, b, last_only=True)[-1]
        assert token_a == token_b
        prefix.append(token_a)
        offset += 1
    candidates = HistoryIndex([prefix]).propose(prefix, 16)
    assert len(candidates) == 16
    snapshot = engine.snapshot(b)
    predictions = engine.predict([token_b, *candidates], offset, b)
    scalar = []
    current = token_a
    for j in range(17):
        current = engine.predict([current], offset + j, a, last_only=True)[-1]
        scalar.append(current)
    assert predictions[:16] == candidates
    differences = [
        {
            "key_max_abs": float(
                mx.max(mx.abs(x.key_values[0] - y.key_values[0])).item()
            ),
            "value_max_abs": float(
                mx.max(mx.abs(x.key_values[1] - y.key_values[1])).item()
            ),
        }
        for x, y in zip(a, b)
    ]
    engine.restore(b, snapshot)
    current = token_b
    for j in range(17):
        current = engine.predict([current], offset + j, b, last_only=True)[-1]
    restored = max(
        float(mx.max(mx.abs(x.key_values[i] - y.key_values[i])).item())
        for x, y in zip(a, b)
        for i in (0, 1)
    )
    result = {
        "task_id": "mbpp/260",
        "attempt": 0,
        "output_offset": 94,
        "accepted": 16,
        "proposed": 16,
        "local_predictions_equal": predictions == scalar,
        "cache_differences": differences,
        "restored_cache_max_abs": restored,
        "interpretation": "A full token match at this batch does not guarantee a scalar-equivalent retained cache",
    }
    write_json(directory / "full_acceptance_audit.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
