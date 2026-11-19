import ast
import json
from pathlib import Path


def test_controlled_edits_have_separate_old_and_new_contracts():
    source = Path(__file__).resolve().parents[1] / "benches/controlled_code_edits.json"
    tasks = json.loads(source.read_text())
    assert len(tasks) == len({task["id"] for task in tasks}) == 10
    for task in tasks:
        ast.parse(task["initial_code"])
        assert task["old_tests"] and task["heldout"] and task["change"]
        assert task["visible"] not in task["old_tests"]
        for test in [*task["old_tests"], task["visible"], *task["heldout"]]:
            ast.parse(test)
