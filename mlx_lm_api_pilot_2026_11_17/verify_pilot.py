"""Verify saved pilot outputs and recompute timings without loading MLX."""

from collections import Counter
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUN = ROOT / "runs/trim_k16_n3_confirm"


def read(name):
    return json.loads((RUN / name).read_text())


def main():
    manifest = read("manifest.json")
    validation = read("validation.json")
    records = read("timings.json")
    summary = read("summary.json")
    tasks = read("tasks.json")
    repo = ROOT.parent.parent
    for path, key in [
        (repo / "src/tiny_llm/mlx_lm_history_api.py", "adapter_sha256"),
        (ROOT / "run_pilot.py", "runner_sha256"),
        (
            repo / ".venv/lib/python3.12/site-packages/mlx_lm/generate.py",
            "installed_generate_sha256",
        ),
    ]:
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest[key], key
    expected = Counter(
        {(t["id"], mode): 3 for t in tasks for mode in ("ordinary", "fixed_history")}
    )
    assert Counter((r["task"], r["mode"]) for r in records) == expected
    references = {r["task"]: r["ordinary"] for r in validation}
    for row in validation:
        assert row["exact"]
        for field in ("tokens", "text", "finish_reason", "terminal_token"):
            assert row["ordinary"][field] == row["fixed_history"][field]
    for record in records:
        for field in ("tokens", "text", "finish_reason", "terminal_token"):
            assert record[field] == references[record["task"]][field]
        assert record["finish_reason"] == "stop"
        assert record["stats"]["replay_forwards"] == 0
    selections = {
        "all_cases": {t["id"] for t in tasks},
        "history_cases": {t["id"] for t in tasks if t["source"]},
        "behavioral_edits": {t["id"] for t in tasks if t["kind"] == "behavioral_edit"},
        **{t["id"]: {t["id"]} for t in tasks},
    }
    for key, ids in selections.items():
        totals = {
            mode: sum(
                r["wall_s"] for r in records if r["task"] in ids and r["mode"] == mode
            )
            for mode in ("ordinary", "fixed_history")
        }
        result = summary["groups"][key]
        assert result["summed_wall_s"] == totals
        assert (
            result["time_ratio_ordinary_over_fixed"]
            == totals["ordinary"] / totals["fixed_history"]
        )
    receipt = {
        "status": "passed",
        "tasks": len(tasks),
        "timed_calls": len(records),
        "validation_pairs": len(validation),
        "all_outputs_exact": True,
        "all_outputs_stopped_at_eos": True,
        "replay_forwards": 0,
        "source_hashes_match": True,
        "aggregate_arithmetic": "passed",
        "cache_state_bitwise_equivalence_tested": False,
        "task_success_evaluated": False,
    }
    (RUN / "verification.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
