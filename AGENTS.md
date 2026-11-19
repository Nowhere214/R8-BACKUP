# AGENTS.md

## Scope

- This file applies to the entire repository.
- Use this as the default test-running policy for coding agents.

## Objective

- Run and verify tests in a way that matches the book workflow (`book/src/*.md`).
- Prefer `pdm` entrypoints defined in `pyproject.toml`.

## Environment Requirements

- macOS on Apple Silicon is expected by the project.
- Install dependencies first:

```bash
pdm install -v
pdm run check-installation
```

- Optional baseline check from the setup chapter (reference solution, Week 1):

```bash
pdm run test-refsol -- -- -k week_1
```

## Agent Test Workflow

1. Start with the smallest relevant scope (`--week` + `--day`).
2. Use pytest filters via `-- -k ...` to isolate failing tasks.
3. Run broader suites only after targeted tests pass.
4. If extension code changed, rebuild extensions before testing.

## Canonical Commands

Run all tests:

```bash
pdm run test
```

Run a specific chapter/day:

```bash
pdm run test --week <WEEK> --day <DAY>
```

Run with pytest filters:

```bash
pdm run test --week 1 --day 3 -- -k task_2
pdm run test --week 2 --day 2 -- -k cpu
pdm run test --week 2 --day 2 -- -k gpu
```

Run reference-solution tests:

```bash
pdm run test-refsol
pdm run test-refsol --week 2 --day 2 -- -k cpu
```

## Extension Rebuild Rule

Rebuild before tests if these changed:

- `src/extensions/src/*`

Commands:

```bash
pdm run build-ext
```

## Guardrails

- Use `--` before pytest args (`-k`, `-q`, `--collect-only`, etc.).
- `pdm run test --week X --day Y` auto-copies `tests_refsol/test_week_X_day_Y.py` into `tests/`.
- Model-dependent tests (0.5B/1.5B/7B) skip when models are not downloaded locally.

## Git Disk Guard

- This repo has a large `.git/objects/pack/*.pack` file. Git repack / auto-maintenance can create huge `.git/objects/pack/tmp_pack_*` files.
- A background `git add -- ...` has previously tried to add models, build products, and generated outputs, repeatedly creating multi-GB `tmp_pack_*` files.
- `gc.auto = 0` and `maintenance.auto = false` are set locally to reduce automatic repack risk.
- `.gitignore` intentionally excludes `models/`, GGUF files, `.venv/`, build outputs, generated previews, PPTX files, TIFF figures, and llama.cpp binaries.
- Before and after large downloads, skill installs, or Git-heavy operations, check:

```bash
find .git/objects/pack -maxdepth 1 -type f -name 'tmp_pack_*' -print
```

- If stale `tmp_pack_*` files exist and no Git job is running, clean them:

```bash
pdm run python scripts/cleanup_git_tmp_packs.py --force
```

- Do not run `git gc`, `git repack`, or broad Git maintenance in this repo unless there is enough free disk space and the user explicitly wants it.
- Do not run `git add .` or `git add --all` in this repo. Stage only specific source/docs files.
