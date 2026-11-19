import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_evaluation_split_has_no_development_or_pilot_tasks():
    root = ROOT / "benches/agent_study_20261021_main"
    tasks = json.loads((root / "tasks.json").read_text())
    calibration = json.loads((root / "calibration.json").read_text())
    dev = {t["id"] for t in tasks if t["split"] == "calibration"}
    evaluation = {t["id"] for t in tasks if t["split"] == "evaluation"}
    assert dev == set(calibration["calibration_ids"])
    assert not dev & evaluation
    assert not evaluation & {f"mbpp/{i}" for i in (113, 61, 274, 257, 245)}
    assert len(evaluation) == 90
    assert len(dev) == 10


def test_all_humaneval_visible_tests_have_behavioral_assertions():
    tasks = json.loads(
        (ROOT / "benches/agent_study_20261021_main/tasks.json").read_text()
    )
    human = [t for t in tasks if t["dataset"] == "humaneval"]
    assert len(human) == 30
    for task in human:
        tree = ast.parse(task["test_list"][0])
        assert isinstance(tree.body[0], ast.Assert)
        assert isinstance(tree.body[0].test, ast.Compare)
        assert not task["test_list"][0].startswith("assert callable")
        assert "canonical_solution" not in task
        assert "code" not in task


def test_revised_cohort_is_disjoint_from_screening_and_uses_same_development_set():
    old = json.loads(
        (ROOT / "benches/agent_study_20261021_main/tasks.json").read_text()
    )
    new = json.loads(
        (ROOT / "benches/agent_study_20261021_holdout/tasks.json").read_text()
    )
    first = {task["id"] for task in old if task["split"] == "evaluation"}
    second = {task["id"] for task in new if task["split"] == "evaluation"}
    assert len(first) == 90 and len(second) == 30
    assert not first & second
    assert {task["id"] for task in old if task["split"] == "calibration"} == {
        task["id"] for task in new if task["split"] == "calibration"
    }
    assert all(
        sum(
            task["dataset"] == dataset and task["split"] == "evaluation" for task in new
        )
        == 15
        for dataset in ("mbpp", "humaneval")
    )


def test_feedback_cohort_is_disjoint_from_both_previous_evaluation_cohorts():
    files = [
        ROOT / "benches" / name / "tasks.json"
        for name in (
            "agent_study_20261021_main",
            "agent_study_20261021_holdout",
            "agent_feedback_study_20261021",
        )
    ]
    groups = [
        {
            task["id"]
            for task in json.loads(path.read_text())
            if task["split"] == "evaluation"
        }
        for path in files
    ]
    assert [len(group) for group in groups] == [90, 30, 20]
    assert len(set.union(*groups)) == 140
    assert not groups[-1] & {f"mbpp/{i}" for i in (113, 61, 274, 257, 245)}
