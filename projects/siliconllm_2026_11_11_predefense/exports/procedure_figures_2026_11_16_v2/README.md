# Procedure figures, version 2

Four minimal English diagrams. Original figures and the reviewed PPT are unchanged.

- `01_mlx_lm_procedure`: ordinary greedy generation and K/V-cache use.
- `02_history_integration_plan`: planned history-policy integration, not a completed stock-API experiment.
- `03_mlx_lm_cache_recovery`: ordinary trimmable target-cache example after partial rejection. A separate draft-model cache is omitted.
- `04_cost_score_and_recovery`: explanation of the existing score, not an evaluated change to the policy.

Each figure has a transparent PNG/SVG and a white-background PDF. The combined PDF contains all four figures. PNG dimensions are 3600 x 1680 pixels.

The technical sources and per-figure bilingual narration remain those in `../../notes/procedure_figures_EN_ZH_2026_11_16.md`. Sources inspected November 16, 2026: mlx-lm `generate.py` and `models/cache.py`, and the local `src/tiny_llm/adaptive_decode.py`. No new measurements or unconditional exact-output guarantee.
