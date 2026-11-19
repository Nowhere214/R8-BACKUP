"""Compute the requested illustrative cost table, not new measurements."""

from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
SOURCE = REPO / "benches/agent_study_20261021_main/calibration.json"
PLACEHOLDERS = REPO / "projects/siliconllm_2026_11_11_predefense/sources/illustrative_placeholders.json"


def fmt(value):
    return str(Decimal(str(value)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))


def main():
    archived = json.loads(SOURCE.read_text())
    illustrative = json.loads(PLACEHOLDERS.read_text())["verification"]
    step = Decimal("18.90")
    lookup = step * Decimal("0.02")
    p = 0.98
    rows = [{"k": 0, "input_tokens": 1, "v_ms": float(step), "l_ms": 0.0,
             "r_ms": 0.0, "g_tokens": 1.0, "c_ms_per_token": float(step),
             "ratio": 1.0, "v_display": fmt(step), "ratio_display": "1.00x"}]
    for k, ratio in zip(illustrative["k"], illustrative["relative_time"]):
        v = step * Decimal(str(ratio))
        r = float(step) * sum((j+1) * p**j * (1-p) for j in range(k))
        g = 1 + sum(p**j for j in range(1, k+1))
        rows.append({"k": k, "input_tokens": k+1, "v_ms": float(v),
                     "l_ms": float(lookup), "r_ms": r, "g_tokens": g,
                     "c_ms_per_token": (float(v)+float(lookup)+r)/g,
                     "ratio": ratio, "v_display": fmt(v), "ratio_display": fmt(ratio)+"x"})
    winner = min(rows, key=lambda row: row["c_ms_per_token"])["k"]
    assert winner == 8
    assert rows[-1]["v_display"] == "38.75"
    data = {
        "status": "Illustrative replay-path calculation; not new inference measurements.",
        "date": "2026-11-18", "step_ms": float(step), "p": p,
        "lookup_ms": float(lookup), "winner": winner, "rows": rows,
        "r_formula": "T_step * sum((j+1) * p**j * (1-p), j=0..k-1)",
        "g_formula": "1 + sum(p**j, j=1..k)",
        "c_formula": "(V(k) + L + R(k)) / G(k)",
        "replay_example": {"accepted": 2, "ordinary_forwards": 3,
                           "conditional_recovery_ms": float(step * 3)},
        "source": {
            "baseline": "User-selected historical reference scale; the archived study is withdrawn.",
            "archived_path": str(SOURCE.relative_to(REPO)),
            "archived_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
            "archived_step_ms": archived["baseline_s"] * 1000,
            "archived_verification_ms_not_used": {k: v*1000 for k,v in archived["verification_s"].items()},
            "ratios_path": str(PLACEHOLDERS.relative_to(REPO)),
            "ratios_sha256": hashlib.sha256(PLACEHOLDERS.read_bytes()).hexdigest(),
            "ratios_are_measurements": False,
        },
        "assumptions": [
            "A constant, independent per-token match probability p is an approximation.",
            "Rejected batches replay j+1 ordinary forwards after j accepted candidates.",
            "Full acceptance has no extra replay in this example.",
            "EOS, output-budget truncation and additional policy overhead are not modeled here.",
            "Native trim requires its own measured recovery-cost model.",
        ],
    }
    (ROOT / "cost_example.json").write_text(json.dumps(data, indent=2)+"\n")
    (ROOT / "cost_data.mjs").write_text("export const data = "+json.dumps(data, indent=2)+";\n")
    print(json.dumps({"step_ms": float(step), "rows": rows, "winner": winner}, indent=2))


if __name__ == "__main__":
    main()
