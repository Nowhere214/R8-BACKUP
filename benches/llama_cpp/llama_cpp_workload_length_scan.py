"""
Run llama.cpp workload length scan with prompt+generation tests.

This uses llama-bench `-pg <prompt,gen>`, which reports a mixed
prompt+generation throughput for that combined workload. It is not the same
metric as the pure decode tok/s reported by mlx-lm stream_generate.

Default:
    pdm run python benches/llama_cpp/llama_cpp_workload_length_scan.py

Outputs:
    benches/llama_cpp/llama_cpp_workload_length_scan.csv
    benches/llama_cpp/llama_cpp_workload_length_scan_summary.csv
"""

from __future__ import annotations

import argparse
import csv
import statistics
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LLAMA_BENCH = ROOT / "tools" / "llama.cpp" / "llama-b9354" / "llama-bench"
OUT_RAW = ROOT / "benches" / "llama_cpp" / "llama_cpp_workload_length_scan.csv"
OUT_SUMMARY = ROOT / "benches" / "llama_cpp" / "llama_cpp_workload_length_scan_summary.csv"

MODEL_PATHS = {
    "qwen2-0.5b": ROOT / "models" / "gguf" / "qwen2-0_5b-instruct-q4_k_m.gguf",
    "qwen2-1.5b": ROOT / "models" / "gguf" / "qwen2-1_5b-instruct-q4_k_m.gguf",
    "qwen2-7b": ROOT / "models" / "gguf" / "qwen2-7b-instruct-q4_k_m.gguf",
}


def parse_csv_from_output(output: str) -> list[dict[str, str]]:
    lines = output.splitlines()
    header_idx = None
    for idx, line in enumerate(lines):
        if line.startswith("build_commit,build_number,"):
            header_idx = idx
            break
    if header_idx is None:
        raise RuntimeError("llama-bench CSV header not found")
    return list(csv.DictReader(lines[header_idx:]))


def run_one(model: str, prompt_len: int, gen_len: int, reps: int, threads: int) -> dict[str, str]:
    model_path = MODEL_PATHS[model]
    if not model_path.exists():
        raise FileNotFoundError(model_path)

    cmd = [
        str(LLAMA_BENCH),
        "-m",
        str(model_path.relative_to(ROOT)),
        "-p",
        "0",
        "-n",
        "0",
        "-pg",
        f"{prompt_len},{gen_len}",
        "-t",
        str(threads),
        "-ngl",
        "99",
        "-r",
        str(reps),
        "-o",
        "csv",
    ]
    print(" ".join(cmd), flush=True)
    completed = subprocess.run(
        cmd,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    rows = parse_csv_from_output(completed.stdout)
    if len(rows) != 1:
        raise RuntimeError(f"expected one llama-bench row, got {len(rows)}")
    row = rows[0]
    row["engine"] = "llama_cpp_metal"
    row["model"] = model
    row["requested_prompt_len"] = str(prompt_len)
    row["requested_gen_len"] = str(gen_len)
    row["repetitions"] = str(reps)
    row["threads_requested"] = str(threads)
    return row


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict[str, str]]) -> list[dict[str, str] | dict[str, float]]:
    out = []
    for row in rows:
        # llama-bench already aggregates repetitions into avg_ts/stddev_ts.
        out.append(
            {
                "engine": row["engine"],
                "model": row["model"],
                "requested_prompt_len": int(row["requested_prompt_len"]),
                "requested_gen_len": int(row["requested_gen_len"]),
                "n": int(row["repetitions"]),
                "mixed_tps_mean": float(row["avg_ts"]),
                "mixed_tps_stdev": float(row["stddev_ts"]),
                "avg_ns": float(row["avg_ns"]),
                "stddev_ns": float(row["stddev_ns"]),
                "model_size_gb": float(row["model_size"]) / 1e9,
                "model_n_params": int(row["model_n_params"]),
                "backend": row["backends"].replace(",", "+"),
                "build_number": row["build_number"],
                "build_commit": row["build_commit"],
            }
        )
    return sorted(out, key=lambda r: (str(r["model"]), int(r["requested_prompt_len"])))


def print_summary(rows: list[dict[str, str] | dict[str, float]]) -> None:
    print()
    print("=" * 98)
    print("LLAMA.CPP WORKLOAD LENGTH SCAN SUMMARY")
    print("=" * 98)
    print(f"{'Model':<12} {'Prompt':>7} {'Gen':>5} {'N':>2} {'Mixed prompt+gen tok/s':>28}")
    print("-" * 98)
    for row in rows:
        print(
            f"{row['model']:<12} {int(row['requested_prompt_len']):>7} "
            f"{int(row['requested_gen_len']):>5} {int(row['n']):>2} "
            f"{float(row['mixed_tps_mean']):>10.1f} ± {float(row['mixed_tps_stdev']):<8.2f}"
        )
    print("=" * 98)


def parse_csv_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def parse_int_csv_list(value: str) -> list[int]:
    return [int(item) for item in parse_csv_list(value)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="qwen2-0.5b,qwen2-1.5b")
    parser.add_argument("--prompt-lens", default="64,128,512")
    parser.add_argument("--gen-len", type=int, default=64)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()

    rows = []
    for model in parse_csv_list(args.models):
        for prompt_len in parse_int_csv_list(args.prompt_lens):
            rows.append(run_one(model, prompt_len, args.gen_len, args.reps, args.threads))

    write_csv(OUT_RAW, rows)
    summary = summarize(rows)
    write_csv(OUT_SUMMARY, summary)
    print_summary(summary)
    print(f"\nRaw     -> {OUT_RAW}")
    print(f"Summary -> {OUT_SUMMARY}")


if __name__ == "__main__":
    main()
