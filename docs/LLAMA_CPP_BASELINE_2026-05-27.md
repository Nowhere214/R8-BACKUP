# llama.cpp Metal Baseline, 2026-05-27

## Setup

- Hardware: Apple M3 Pro, 18 GB unified memory
- llama.cpp: b9354, commit `9777256c3`
- Binary: `tools/llama.cpp/llama-b9354/llama-bench`
- Backend: Metal + BLAS (`-ngl 99`, devices `auto`)
- Models: Qwen2 Instruct GGUF, Q4_K_M
- Workload: `-p 128 -n 128 -t 4`
- Repetitions: `-r 3`

## Command

```bash
tools/llama.cpp/llama-b9354/llama-bench \
  -m models/gguf/qwen2-0_5b-instruct-q4_k_m.gguf,models/gguf/qwen2-1_5b-instruct-q4_k_m.gguf,models/gguf/qwen2-7b-instruct-q4_k_m.gguf \
  -p 128 -n 128 -t 4 -ngl 99 -r 3 -o csv
```

## Results

| Engine | Model | Prefill tok/s | Decode tok/s |
|---|---:|---:|---:|
| llama.cpp Metal | Qwen2 0.5B Q4_K_M | 2,924.7 ± 36.2 | 137.3 ± 1.0 |
| llama.cpp Metal | Qwen2 1.5B Q4_K_M | 1,103.7 ± 8.3 | 71.9 ± 0.5 |
| llama.cpp Metal | Qwen2 7B Q4_K_M | 266.2 ± 2.0 | 22.2 ± 0.03 |

## Immediate Interpretation

- llama.cpp Metal fits cleanly between the Stage 1 reference implementation and mlx-lm.
- For prefill, llama.cpp is close to mlx-lm, especially at 7B.
- For decode, llama.cpp is much faster than the Stage 1 reference path but still slower than mlx-lm.
- The 7B run completed successfully, but memory pressure became visible after the run. Further 7B or long-context experiments should be run after closing large applications.

## Generated Files

- Raw CSV: `benches/llama_cpp/llama_cpp_b9354_qwen2_q4km.csv`
- Summary CSV: `benches/llama_cpp/llama_cpp_b9354_summary.csv`
- Figure: `benches/llama_cpp/phase0_three_engine_llama_cpp.png`
