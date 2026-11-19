"""Run the revised repair-only study on a disjoint, frozen 30-task cohort."""

import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_agent_study import ROOT, load_engines, policy_from, write_json
from tiny_llm.adaptive_decode import decode

OUT = ROOT / "benches/agent_study_20261021_holdout"
SCREEN = ROOT / "benches/agent_study_20261021_main"


def status(stage, state, **extra):
    value = {
        "stage": stage,
        "state": state,
        "updated_utc": datetime.now(timezone.utc).isoformat(),
        **extra,
    }
    write_json(OUT / "suite_status.json", value)
    print(json.dumps(value), flush=True)


def validation_gate():
    costs = json.loads((OUT / "calibration_history.json").read_text())
    traces = [
        json.loads(line)
        for line in (SCREEN / "policy_calibration/turns.jsonl").read_text().splitlines()
    ]
    engine, _, _, tokenizer, _ = load_engines()
    checks = []
    for index, trace in enumerate(traces):
        for mode in ("warm_fixed", "warm_adaptive", "warm_acceptance"):
            result = decode(
                engine,
                trace["prompt_tokens"],
                384,
                {tokenizer.eos_token_id},
                mode=mode,
                history=trace["history"],
                draft_length=16,
                policy=policy_from(costs),
            )
            checks.append(
                {
                    "task_id": trace["task_id"],
                    "attempt": trace["attempt"],
                    "mode": mode,
                    "exact": result.tokens == trace["result"]["tokens"],
                }
            )
        print(f"repair-only validation {index + 1}/{len(traces)}", flush=True)
    write_json(OUT / "validation_checks.json", checks)
    if not all(row["exact"] for row in checks):
        raise RuntimeError("Repair-only methods failed their separate validation gate")


def main():
    if "--validation-only" in sys.argv:
        validation_gate()
        return
    status("baseline", "waiting")
    while not (OUT / "baseline/metadata.json").exists():
        time.sleep(5)
    saved = OUT / "source"
    saved.mkdir(exist_ok=False)
    for relative in (
        "src/tiny_llm/adaptive_decode.py",
        "src/tiny_llm/native_history.py",
        "src/tiny_llm/agent_pilot.py",
        "benches/run_agent_study.py",
        "benches/run_native_history_study.py",
        "benches/run_agent_holdout_suite.py",
    ):
        shutil.copy2(ROOT / relative, saved / Path(relative).name)
    jobs = [
        (
            "screening_analysis",
            "benches/analyze_agent_study.py",
            "--out",
            str(SCREEN),
            "--label",
            "screening_replay",
        ),
        ("full_acceptance_audit", "benches/audit_full_acceptance.py"),
        ("validation", "benches/run_agent_holdout_suite.py", "--validation-only"),
        (
            "paired_replay",
            "benches/run_agent_study.py",
            "replay",
            "--out",
            str(OUT),
            "--label",
            "replay",
            "--modes",
            "baseline,warm_fixed,warm_adaptive,warm_acceptance",
            "--reps",
            "3",
            "--cost-file",
            "calibration_history.json",
        ),
        (
            "replay_analysis",
            "benches/analyze_agent_study.py",
            "--out",
            str(OUT),
            "--label",
            "replay",
        ),
        (
            "closed_loop",
            "benches/run_agent_study.py",
            "closed",
            "--out",
            str(OUT),
            "--label",
            "closed",
            "--modes",
            "baseline,warm_fixed,warm_adaptive",
            "--reps",
            "3",
            "--per-dataset",
            "10",
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
        (
            "native_extension",
            "benches/run_native_history_study.py",
            "--source",
            str(OUT),
            "--out",
            str(OUT / "native_extension"),
            "--per-dataset",
            "10",
            "--reps",
            "3",
        ),
        (
            "native_analysis",
            "benches/analyze_agent_study.py",
            "--out",
            str(OUT / "native_extension"),
        ),
        (
            "small_reference",
            "benches/run_agent_study.py",
            "replay",
            "--out",
            str(SCREEN),
            "--label",
            "reference_comparison",
            "--modes",
            "baseline,small,native",
            "--per-dataset",
            "10",
            "--reps",
            "1",
            "--cost-file",
            "calibration_guarded.json",
        ),
        (
            "small_analysis",
            "benches/analyze_agent_study.py",
            "--out",
            str(SCREEN),
            "--label",
            "reference_comparison",
        ),
    ]
    failed = []
    for name, *command in jobs:
        status(name, "running")
        result = subprocess.run([sys.executable, *command], cwd=ROOT)
        if result.returncode:
            status(name, "failed", returncode=result.returncode)
            failed.append(name)
            if name in {"validation", "paired_replay", "closed_loop"}:
                raise SystemExit(result.returncode)
    status(
        "all",
        "complete" if not failed else "complete_with_failures",
        failed_stages=failed,
    )


if __name__ == "__main__":
    main()
