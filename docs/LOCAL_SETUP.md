# Local setup - what's already done, what you need to run yourself

All the files and dependencies are in place. A few steps must be run in a
regular **Terminal.app / iTerm / zsh** window (outside Cursor's agent shell),
because:

1. The Cursor agent shell runs sandboxed and cannot access the Metal GPU, so
  any command that imports `mlx` crashes at import time.
2. A few items couldn't be written from inside the sandbox (`.vscode/`,
  sometimes `.git/`, ...).

## Done already (inside this repo)

- Cloned upstream `skyzh/tiny-llm` into `/Users/laiqixiao/Documents/SiliconLLM`
- Renamed project to `silicon-llm` in `pyproject.toml`
- Replaced `README.md` with a SiliconLLM-branded one (upstream README content
is reproduced inside)
- Installed `pdm 2.26.8` → `~/.local/bin/pdm`
- Installed managed Python 3.12.13 via pdm
- Created `.venv/` and ran `pdm install` → all deps installed:
  - `mlx 0.29.1`, `mlx-lm`, `torch 2.7.1`, `torchtune 0.6.1`, `torchao 0.12.0`,
  `numpy 2.2.6`, `pytest 8.4.1`, `nanobind 2.4.0`, `pytest-benchmark`
- `silicon-llm 0.1.0` registered as an editable package
- Added `docs/PHASE0_BENCHMARK.md` (the benchmark plan per advisor feedback)

## You should run these yourself (in a normal Terminal, not Cursor)

```bash
# 1) Make pdm visible on every new shell. Add this line to your ~/.zshrc:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

cd ~/Documents/SiliconLLM

# 2) Actual MLX / Metal smoke test (this is the one that crashes in the sandbox)
pdm run check-installation
# Expected output:
#   array([5, 7, 9], dtype=int32)     <- mlx CPU
#   array([5, 7, 9], dtype=int32)     <- mlx GPU
#   tensor([5, 7, 9])                 <- torch CPU

# 3) Sanity-check the reference solution's tests
pdm run test-refsol -- -- -k week_1

# 4) Turn the folder into a git repo (couldn't do this from the sandbox)
git init
git add .
git commit -m "Initial import of skyzh/tiny-llm, renamed to SiliconLLM"
# optionally: git remote add origin <your-fork-url>

# 5) Restore .vscode/ settings (optional; were excluded during extract)
mkdir -p .vscode
cat > .vscode/settings.json <<'EOF'
{
  "python.defaultInterpreterPath": ".venv/bin/python",
  "python.testing.pytestEnabled": true,
  "python.testing.pytestArgs": ["tests", "tests_refsol"]
}
EOF
```

## Download a model to actually run inference

```bash
# Hugging Face CLI
pdm run python -m pip install -U "huggingface_hub[cli]"
# Authenticate (you need a HF account token — free)
pdm run hf auth login

# Small model (0.5B, ~0.4 GB) — use this to smoke-test SiliconLLM
pdm run hf download Qwen/Qwen2-0.5B-Instruct-MLX

# Big model (7B, ~4.3 GB, needs ~20 GB RAM to LOAD in Stage 1 since weights
# get dequantized in-memory) — only do this once you have disk/RAM headroom
pdm run hf download Qwen/Qwen2-7B-Instruct-MLX
```

> **Disk warning**: you had ~18 GB free before setup, ~17 GB free now.
> If you also want to run llama.cpp (Q4_K_M GGUF, ~4.6 GB) and vLLM
> (bf16 HF weights, ~15 GB) for Phase 0, you need to free disk first or
> run Phase 0 with the 0.5B model only. See `docs/PHASE0_BENCHMARK.md`
> Section 3.

## Run the reference implementation end-to-end (after downloading the 0.5B model)

```bash
pdm run main --solution ref --loader week1
# Prompts you for input, then streams out Qwen2's response
```

## Run your own code's tests

```bash
pdm run test --week 1 --day 1
pdm run test --week 1 --day 1 -- -k task_1
```

Right now `src/tiny_llm/` contains only stubs — all the `TODO`-style tasks in
the book will be done there. `src/tiny_llm_ref/` already has the full
reference for weeks 1 and 2.

## Benchmarks

```bash
# The upstream throughput benchmark — use this as the template for Phase 0
pdm run bench --model qwen2-0.5b --solution ref --loader week2
# Output includes prefill tok/s, decode tok/s, total tok/s.
```

For Phase 0 (SiliconLLM vs mlx-lm vs llama.cpp vs vLLM), follow the plan in
`docs/PHASE0_BENCHMARK.md`.