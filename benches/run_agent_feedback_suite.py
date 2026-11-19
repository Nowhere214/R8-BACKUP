"""Validate fallback feedback, then evaluate it on a fresh public-task cohort."""

import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from run_agent_study import ROOT, load_engines, policy_from, write_json
from tiny_llm.adaptive_decode import decode

OUT = ROOT / "benches/agent_feedback_study_20261021"


def status(stage, state, **extra):
    value = {
        "stage": stage,
        "state": state,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    write_json(OUT / "suite_status.json", value)
    print(json.dumps(value), flush=True)


def validate():
    costs = json.loads((OUT / "calibration_history.json").read_text())
    source = ROOT / "benches/agent_study_20261021_holdout/baseline/turns.jsonl"
    known = next(
        row
        for row in (json.loads(line) for line in source.read_text().splitlines())
        if row["task_id"] == "humaneval/17" and row["attempt"] == 1
    )
    engine, _, _, tokenizer, _ = load_engines()
    result = decode(
        engine,
        known["prompt_tokens"],
        384,
        {tokenizer.eos_token_id},
        mode="warm_feedback",
        history=known["history"],
        draft_length=16,
        policy=policy_from(costs),
    )
    regression = {
        "task_id": known["task_id"],
        "attempt": 1,
        "exact": result.tokens == known["result"]["tokens"],
        **result.to_dict(),
    }
    write_json(OUT / "known_case_regression.json", regression)
    if not regression["exact"] or not any(
        event["output_offset"] > 157 for event in result.decisions
    ):
        raise RuntimeError(
            "Fallback-feedback revision failed its known-case regression"
        )
    calibration = (
        ROOT / "benches/agent_study_20261021_main/policy_calibration/turns.jsonl"
    )
    checks = []
    for index, trace in enumerate(
        json.loads(line) for line in calibration.read_text().splitlines()
    ):
        result = decode(
            engine,
            trace["prompt_tokens"],
            384,
            {tokenizer.eos_token_id},
            mode="warm_feedback",
            history=trace["history"],
            draft_length=16,
            policy=policy_from(costs),
        )
        checks.append(
            {
                "task_id": trace["task_id"],
                "attempt": trace["attempt"],
                "exact": result.tokens == trace["result"]["tokens"],
            }
        )
        print(f"fallback-feedback validation {index + 1}/20", flush=True)
    write_json(OUT / "validation_checks.json", checks)
    if not all(row["exact"] for row in checks):
        raise RuntimeError("Fallback-feedback revision failed validation exactness")


def main():
    if "--validation-only" in sys.argv:
        validate()
        return
    old_status = json.loads(
        (ROOT / "benches/agent_study_20261021_holdout/suite_status.json").read_text()
    )
    if old_status["stage"] != "all" or not old_status["state"].startswith("complete"):
        raise RuntimeError(
            "Finish the older serialized suite before starting another GPU study"
        )
    source = OUT / "source"
    source.mkdir(exist_ok=False)
    for relative in (
        "src/tiny_llm/adaptive_decode.py",
        "benches/run_agent_study.py",
        "benches/run_agent_feedback_suite.py",
    ):
        shutil.copy2(ROOT / relative, source / Path(relative).name)
    jobs = [
        ("validation", "benches/run_agent_feedback_suite.py", "--validation-only"),
        ("baseline", "benches/run_agent_study.py", "collect", "--out", str(OUT)),
        (
            "replay",
            "benches/run_agent_study.py",
            "replay",
            "--out",
            str(OUT),
            "--label",
            "replay",
            "--modes",
            "baseline,warm_fixed,warm_adaptive,warm_feedback",
            "--reps",
            "3",
            "--cost-file",
            "calibration_history.json",
        ),
        ("analysis", "benches/analyze_agent_study.py", "--out", str(OUT)),
        (
            "closed",
            "benches/run_agent_study.py",
            "closed",
            "--out",
            str(OUT),
            "--label",
            "closed",
            "--modes",
            "baseline,warm_feedback",
            "--reps",
            "3",
            "--per-dataset",
            "5",
            "--cost-file",
            "calibration_history.json",
        ),
        (
            "closed_analysis",
            "benches/analyze_agent_closed.py",
            "--out",
            str(OUT),
            "--label",
            "closed",
        ),
        ("controlled_edits", "benches/run_controlled_edit_study.py"),
    ]
    for name, *command in jobs:
        status(name, "running")
        completed = subprocess.run([sys.executable, *command], cwd=ROOT)
        if completed.returncode:
            status(name, "failed", returncode=completed.returncode)
            raise SystemExit(completed.returncode)
    status("all", "complete")


if __name__ == "__main__":
    main()
