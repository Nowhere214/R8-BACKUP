"""Freeze a new public-task cohort for fallback-feedback controller testing."""

import json
from pathlib import Path
from types import SimpleNamespace

from run_agent_study import ROOT, prepare, write_json


def main():
    benches = ROOT / "benches"
    previous = []
    for name in ("agent_study_20261021_main", "agent_study_20261021_holdout"):
        previous.extend(json.loads((benches / name / "tasks.json").read_text()))
    exclusions = benches / "agent_feedback_exclusions_20261021.json"
    write_json(
        exclusions,
        [
            {"id": task_id, "split": "evaluation"}
            for task_id in sorted(
                {task["id"] for task in previous if task["split"] == "evaluation"}
            )
        ],
    )
    output = benches / "agent_feedback_study_20261021"
    args = SimpleNamespace(
        out=output,
        mbpp=Path("/private/tmp/siliconllm-sanitized-mbpp.json"),
        humaneval=Path("/private/tmp/siliconllm-humaneval.jsonl.gz"),
        mbpp_count=10,
        humaneval_count=10,
        seed=20261024,
        human_seed=20261024,
        max_tokens=384,
        exclude_manifest=exclusions,
    )
    prepare(args)
    costs = json.loads(
        (benches / "agent_study_20261021_holdout/calibration_history.json").read_text()
    )
    write_json(output / "calibration_history.json", costs)
    write_json(
        output / "revision_protocol.json",
        {
            "reason": "Original cost control can become stuck in ordinary decode without new acceptance feedback",
            "method": "When ordinary decoding already produces the next token, compare it with the first available historical candidate and update acceptance without an extra target call",
            "cohort": "10 MBPP and 10 HumanEval tasks disjoint from all 120 earlier evaluation tasks and the original five-task pilot",
            "modes": ["baseline", "warm_fixed", "warm_adaptive", "warm_feedback"],
            "reps": 3,
            "calibration": "Unchanged independent validation-derived costs and prior 0.99",
            "known_case": "HumanEval 17 repair 1 is a development regression case, not part of the fresh cohort",
            "not_a_novelty_claim": "This is a controller variant and ablation; no claim of first publication or superiority to all prior adaptive methods",
        },
    )


if __name__ == "__main__":
    main()
