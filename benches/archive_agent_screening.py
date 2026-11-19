"""Retain the fully completed first round without claiming three repetitions."""

import json
from collections import Counter

from run_agent_study import ROOT, fingerprint, write_json


def main():
    directory = ROOT / "benches/agent_study_20261021_main"
    source = directory / "guarded_replay/replay.jsonl"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    first = [row for row in rows if row["rep"] == 0]
    modes = ["baseline", "global", "structure", "adaptive", "acceptance"]
    counts = Counter((row["task_id"], row["attempt"], row["mode"]) for row in first)
    if (
        len(first) != 890
        or len(counts) != 890
        or any(value != 1 for value in counts.values())
    ):
        raise ValueError("The first round is not a complete unique set of 890 decodes")
    output = directory / "screening_replay"
    output.mkdir(exist_ok=False)
    with (output / "replay.jsonl").open("w") as stream:
        for row in first:
            stream.write(json.dumps(row) + "\n")
    write_json(
        output / "metadata.json",
        {
            "modes": modes,
            "reps": 1,
            "max_tokens": 384,
            "draft_length": 16,
            "original_planned_reps": 3,
            "original_records": len(rows),
            "source_sha256": fingerprint(source),
            "selection": "Entire completed repetition zero, including all preservation failures",
            "reason": "Stop invalid configurations after one complete screening round; use fresh tasks for revised phase-gated methods",
        },
    )
    validation = json.loads(
        (directory / "policy_calibration/history_results.json").read_text()
    )
    repairs = [row for row in validation if row["attempt"] > 0]
    accepted = sum(row["accepted"] for row in repairs)
    proposed = sum(row["proposed"] for row in repairs)
    costs = json.loads((directory / "calibration_guarded.json").read_text())
    costs.update(
        initial_acceptance=min(0.99, (accepted + 1) / (proposed + 2)),
        calibration_accepted=accepted,
        calibration_proposed=proposed,
        prior_source="Repair requests only from the original ten independent MBPP validation tasks",
        repair_only=True,
    )
    target = ROOT / "benches/agent_study_20261021_holdout"
    write_json(target / "calibration_history.json", costs)
    print(
        {
            "screening_runs": len(first),
            "accepted": accepted,
            "proposed": proposed,
            "prior": costs["initial_acceptance"],
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
