import json
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benches"))
import run_fixed_length_ablation as study  # noqa: E402
from tiny_llm.adaptive_decode import DecodeResult  # noqa: E402


def test_length_ablation_preserves_all_pairs_and_passes_the_requested_length(
    tmp_path, monkeypatch
):
    seen = []

    def fake_decode(engine, prompt, budget, eos, **kwargs):
        seen.append((kwargs["mode"], kwargs["draft_length"], kwargs["history"]))
        assert budget == 384
        assert eos == {9}
        return DecodeResult(tokens=[3, 9])

    monkeypatch.setattr(study, "decode", fake_decode)
    monkeypatch.setattr(study, "policy_from", lambda costs: None)
    turns = [
        {
            "task_id": "task/a",
            "dataset": "mbpp",
            "attempt": 0,
            "phase": "generation",
            "prompt_tokens": [1, 2],
            "history": [[2, 3, 9]],
            "result": {"tokens": [3, 9]},
        }
    ]
    modes = {"baseline": ("baseline", 16), "fixed_k2": ("warm_fixed", 2)}
    output = tmp_path / "replay.jsonl"
    rows = study.measure(
        None, SimpleNamespace(eos_token_id=9), turns, {}, modes, output, 42
    )
    assert len(rows) == 6
    assert Counter((row["mode"], row["rep"]) for row in rows) == Counter(
        (mode, rep) for mode in modes for rep in range(3)
    )
    assert all(row["exact"] for row in rows)
    assert all(
        length == (2 if mode == "warm_fixed" else 16) for mode, length, _ in seen
    )
    assert [json.loads(line) for line in output.read_text().splitlines()] == rows
