import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benches"))
from analyze_agent_study import summarize  # noqa: E402
from analyze_agent_closed import summarize_closed  # noqa: E402


def test_paired_aggregation_uses_ratio_of_task_totals_and_groups():
    rows = []
    for task, base, treatment, phase in (
        ("a", 2.0, 1.0, "generation"),
        ("b", 8.0, 8.0, "repair"),
    ):
        for rep in range(2):
            for mode, wall in (("baseline", base), ("adaptive", treatment)):
                rows.append(
                    {
                        "task_id": task,
                        "attempt": 0,
                        "dataset": "mbpp",
                        "phase": phase,
                        "mode": mode,
                        "rep": rep,
                        "exact": True,
                        "wall_s": wall,
                        "prefill_s": 0.1,
                        "decode_s": wall - 0.1,
                        "lookup_s": 0,
                        "decision_s": 0,
                        "draft_s": 0,
                        "verification_s": wall - 0.1,
                        "cache_s": 0,
                        "proposed": 0,
                        "accepted": 0,
                        "speculative_steps": 0,
                        "ordinary_steps": 1,
                    }
                )
    result = summarize(
        rows, [{"task_id": "a", "passed": True}, {"task_id": "b", "passed": False}]
    )
    overall = next(
        r
        for r in result
        if (r["dataset"], r["mode"], r["subset"]) == ("all", "adaptive", "all")
    )
    assert overall["speedup"] == pytest.approx(10 / 9)
    assert overall["tasks"] == 2
    assert overall["timed_runs"] == 4
    successful = next(
        r
        for r in result
        if (r["dataset"], r["mode"], r["subset"]) == ("all", "adaptive", "passed")
    )
    assert successful["speedup"] == 2
    for row in rows:
        if row["mode"] == "baseline":
            row["mode"] = "structure"
    comparison = summarize(rows, reference_mode="structure")
    increment = next(
        row
        for row in comparison
        if row["dataset"] == "all"
        and row["mode"] == "adaptive"
        and row["subset"] == "all"
    )
    assert increment["reference_mode"] == "structure"
    assert increment["speedup"] == pytest.approx(10 / 9)
    rows[0]["exact"] = False
    invalid = next(
        row
        for row in summarize(rows, reference_mode="structure")
        if row["dataset"] == "all"
        and row["mode"] == "adaptive"
        and row["subset"] == "all"
    )
    assert not invalid["sequence_preservation"]
    assert invalid["equivalent_output_speedup"] is None


def test_closed_loop_requires_complete_pairs_and_includes_framework_cost():
    metadata = {
        "task_ids": ["mbpp/a", "mbpp/b"],
        "modes": ["baseline", "adaptive"],
        "reps": 2,
    }
    rows = [
        {
            "task_id": task,
            "dataset": "mbpp",
            "mode": mode,
            "rep": rep,
            "passed": task.endswith("a"),
            "exact": True,
            "calls": 1,
            "model_s": 2 if mode == "baseline" else 1,
            "agent_s": 2.1 if mode == "baseline" else 1.1,
        }
        for task in metadata["task_ids"]
        for mode in metadata["modes"]
        for rep in range(2)
    ]
    result = summarize_closed(rows, metadata)
    item = next(
        row for row in result if row["dataset"] == "all" and row["mode"] == "adaptive"
    )
    assert item["agent_s"]["speedup"] == pytest.approx(2.1 / 1.1)
    assert item["model_s"]["speedup"] == 2
    assert item["exact_episodes"] == item["same_outcome_episodes"] == 4
    with pytest.raises(ValueError, match="Incomplete"):
        summarize_closed(rows[:-1], metadata)
