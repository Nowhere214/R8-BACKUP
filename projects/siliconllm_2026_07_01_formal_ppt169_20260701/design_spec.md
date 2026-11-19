# siliconllm_2026_07_01_formal - Design Spec

## I. Project Information

| Item | Value |
| ---- | ----- |
| **Project Name** | siliconllm_2026_07_01_formal |
| **Canvas Format** | PPT 16:9 (1280x720) |
| **Page Count** | 16 |
| **Design Style** | General Consulting + academic systems research |
| **Target Audience** | Research supervisor and lab members |
| **Use Case** | Formal group meeting progress presentation on 2026-07-01 |
| **Created Date** | 2026-07-01 |

---

## II. Canvas Specification

| Property | Value |
| -------- | ----- |
| **Format** | PPT 16:9 |
| **Dimensions** | 1280x720 |
| **viewBox** | `0 0 1280 720` |
| **Margins** | left/right 64px, top 46px, bottom 42px |
| **Content Area** | 1152x632 |

---

## III. Visual Theme

### Theme Style

- **Style**: General Consulting + academic systems research
- **Theme**: Light theme
- **Tone**: formal, measured, data-first, publication-like

### Color Scheme

| Role | HEX | Purpose |
| ---- | --- | ------- |
| **Background** | `#F4EFFA` | Slide background |
| **Secondary bg** | `#FFFFFF` | Figure panels and light cards |
| **Primary** | `#3E1485` | Titles, main dividers, Stage 1 color |
| **Accent** | `#6B4CA3` | Validated baseline and mechanism color |
| **Secondary accent** | `#9E89C2` | Production engine and key number highlight |
| **Body text** | `#1F1633` | Main text |
| **Secondary text** | `#4B405D` | Captions and supporting text |
| **Tertiary text** | `#6F6280` | Footers and caveats |
| **Border/divider** | `#D6CBE8` | Panel borders and dividers |
| **Success** | `#6B4CA3` | Passed checks |
| **Warning** | `#7A4B00` | Caveats and limitations |

No AI image rendering is used. The visual assets are publication-style scientific figures generated from project CSVs.

---

## IV. Typography System

### Font Plan

**Typography direction**: clean technical sans-serif, PowerPoint safe.

| Role | Chinese | English | Fallback tail |
| ---- | ------- | ------- | ------------- |
| **Title** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Body** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Emphasis** | `"Microsoft YaHei"` | `Arial` | `sans-serif` |
| **Code** | — | `Consolas, "Courier New"` | `monospace` |

**Per-role font stacks**

- Title: `Arial, Microsoft YaHei, sans-serif`
- Body: `Arial, Microsoft YaHei, sans-serif`
- Emphasis: `Arial, Microsoft YaHei, sans-serif`
- Code: `Consolas, Courier New, monospace`

### Font Size Hierarchy

**Baseline**: Body font size = 19px

| Purpose | Ratio to body | Project value | Weight |
| ------- | ------------- | ------------- | ------ |
| Cover title | 3.6-4.2x | 70-80px | Bold |
| Section opener | 2.0-2.4x | 40-46px | Bold |
| Page title | 1.7-1.9x | 32-36px | Bold |
| Hero number | 2.0-2.5x | 40-48px | Bold |
| Subtitle | 1.2-1.4x | 23-27px | SemiBold |
| **Body content** | **1x** | **19px** | Regular |
| Annotation / caption | 0.72-0.85x | 14-16px | Regular |
| Footer | 0.55-0.65x | 11-12px | Regular |

Formula rendering policy: `text-only`. There are no complex equations in this deck.

---

## V. Layout Principles

### Page Structure

- **Header area**: 46-112px. Holds page number, section tag, and concise title.
- **Content area**: 530-590px. Holds one main argument per slide.
- **Footer area**: 28-42px. Holds source/caveat when needed.

### Layout Pattern Library

- Use large figure panels with modest annotation, not dense text grids.
- Keep cards to 6-8px corner radius.
- Avoid nested cards.
- Use whitespace and thin dividers rather than heavy decorative backgrounds.
- Use figure pages as the visual center; text should explain the reading of the figure, not repeat every number.

### Spacing Specification

| Element | Current Project |
| ------- | --------------- |
| Safe margin from canvas edge | 64px |
| Content block gap | 28-36px |
| Icon-text gap | 10-14px |
| Card gap | 22-28px |
| Card padding | 20-26px |
| Card border radius | 6-8px |

---

## VI. Icon Usage Specification

### Source

- **Built-in icon library**: `tabler-outline`
- **Stroke width**: 2
- **Usage method**: simple SVG line icons in section markers, validation marks, and pipeline labels.

### Recommended Icon List

| Purpose | Icon Path | Page |
| ------- | --------- | ---- |
| Evidence / chart | `tabler-outline/chart-line` | P02, P07, P08 |
| Engine / compute | `tabler-outline/cpu` | P03, P05 |
| Benchmark table | `tabler-outline/database` | P04, P14 |
| Profiling | `tabler-outline/activity` | P09 |
| Baseline validation | `tabler-outline/circle-check` | P11, P12 |
| Implementation | `tabler-outline/code` | P10 |
| Next step | `tabler-outline/route` | P15 |
| Takeaway | `tabler-outline/target` | P16 |

---

## VII. Visualization Reference List

Catalog read: 71 templates.

The deck uses externally generated publication figures instead of chart templates for numeric plots. Those figures are already calibrated from CSV data and are embedded as no-crop images. Structural pages are custom free-design layouts.

| Page | Template | Path | Summary-quote (verbatim) | Usage |
| ---- | -------- | ---- | ------------------------ | ----- |
| P02 | no-template-match | custom SVG layout | n/a | Evidence-chain overview built as native SVG blocks. |
| P05 | no-template-match | publication figure asset | n/a | Three-engine decode comparison already rendered as a figure. |
| P07 | no-template-match | publication figure asset | n/a | Prompt-length gap figure already rendered as a figure. |
| P09 | no-template-match | publication figure asset | n/a | Decode component profile already rendered as a figure. |
| P12 | no-template-match | publication figure asset | n/a | Self/reference parity figure already rendered as a figure. |
| P14 | no-template-match | publication figure asset | n/a | Reproducibility summary figure already rendered as a figure. |

Runners-up considered:

- `pipeline_with_stages` | rejected for P02: the slide is a research evidence chain, but the custom figure has already encoded the chain and is more compact.
- `bar_chart` | rejected for P05/P09/P12: the data plots were redrawn with scientific-visualization and exported as publication figures.
- `kpi_cards` | rejected for P14: the figure asset already contains the KPI-card structure with publication typography.

---

## VIII. Image Resource List

All figure assets were generated from local CSV summaries with the scientific-visualization workflow and exported as PNG, PDF, SVG, and TIFF. The PPT uses PNG for reliable embedding; vector/PDF/TIFF copies remain in `images/`.

| Filename | Dimensions | Ratio | Purpose | Type | Layout pattern | Acquire Via | Status | Reference | text_policy | page_role |
| -------- | ---------- | ----- | ------- | ---- | -------------- | ----------- | ------ | --------- | ----------- | --------- |
| formal_fig01_evidence_chain.png | 2474x1279 | 1.93 | Opening evidence-chain figure | Scientific Figure | `#19 Image floating in whitespace with thin frame and caption` | user | Existing | Phase 0 to Phase 1 evidence chain | none | local |
| formal_fig02_three_engine_decode.png | 2513x1611 | 1.56 | Three-engine decode comparison | Scientific Figure | `#19 Image floating in whitespace with thin frame and caption` | user | Existing | Decode throughput comparison across Stage 1, llama.cpp, and mlx-lm | none | local |
| formal_fig03_prompt_length_gap.png | 2565x1613 | 1.59 | Prompt-length gap scan | Scientific Figure | `#19 Image floating in whitespace with thin frame and caption` | user | Existing | Decode ratio growth as prompt length increases | none | local |
| formal_fig04_stage1_decode_profile.png | 2709x1545 | 1.75 | Decode component profile | Scientific Figure | `#19 Image floating in whitespace with thin frame and caption` | user | Existing | Full-context decode component timing | none | local |
| formal_fig05_self_reference_parity.png | 2557x1590 | 1.61 | Self/reference parity benchmark | Scientific Figure | `#19 Image floating in whitespace with thin frame and caption` | user | Existing | Self Stage 1 compared with reference Stage 1 | none | local |
| formal_fig06_reproducibility_summary.png | 2328x1149 | 2.03 | Result package summary | Scientific Figure | `#19 Image floating in whitespace with thin frame and caption` | user | Existing | Phase 0 and Phase 1 row counts | none | local |

Reason for no image-as-canvas pages: all image-bearing pages use data figures where exact axis labels and data geometry must remain visible. They are therefore embedded as no-crop scientific figures rather than atmospheric backgrounds.

---

## IX. Content Outline

| Page | Title | Rhythm | Main Message | Visual Plan |
| ---- | ----- | ------ | ------------ | ----------- |
| P01 | Efficient LLM Inference on Apple Silicon | anchor | Formal progress update after the 5/20 ZEMI. | Strong title, simple subtitle, no data overload. |
| P02 | What Changed Since Last Time | dense | The work moved from observing a gap to explaining it and validating a baseline. | `formal_fig01_evidence_chain.png`. |
| P03 | Last ZEMI Finding | dense | The previous major finding was that decode, not prefill, is the largest gap. | Two-column recall: last message vs new work. |
| P04 | Phase 0 Is Now a Reproducible Diagnosis | dense | Phase 0 now has baselines, scans, microbenchmarks, and caveats. | Native metric cards. |
| P05 | A Third Engine Confirms the Decode Story | dense | llama.cpp sits between Stage 1 and mlx-lm, supporting the bottleneck diagnosis. | `formal_fig02_three_engine_decode.png`. |
| P06 | Longer Prompt, Larger Gap | breathing | Prompt length makes the Stage 1 decode gap worse. | Large headline with three numbers. |
| P07 | Prompt-length Scan | dense | The gap grows from about 14-15x to about 58x. | `formal_fig03_prompt_length_gap.png`. |
| P08 | Cost Model Reading | dense | Stage 1 has a measurable per-context-token decode cost; mlx-lm is nearly flat. | Native table with slope values. |
| P09 | Decode Is Not Just Python Loop Overhead | dense | Full-context decode pays MLP, LM head, and attention cost again. | `formal_fig04_stage1_decode_profile.png`. |
| P10 | Why This Matters Before Optimization | dense | A correct baseline must come before optimization claims. | Four-step process diagram. |
| P11 | Phase 1 Implementation Status | dense | Stage 1 components are implemented and targeted tests pass. | Validation checklist. |
| P12 | Self Stage 1 Matches Reference | dense | The self baseline matches the reference on 0.5B and 1.5B smoke. | `formal_fig05_self_reference_parity.png`. |
| P13 | What We Can Claim Now | breathing | The research has a defensible diagnostic and baseline story. | Three claim cards. |
| P14 | Current Progress Summary | dense | The project is ready to move from diagnosis and baseline validation into Stage 2. | Native progress cards. |
| P15 | Next Step: Stage 2 KV-cache | dense | Next work should remove full-context decode first, then re-profile. | Roadmap diagram. |
| P16 | Takeaway | anchor | Measure first, explain second, validate baseline third, optimize fourth. | Large final statement and Q&A. |

---

## X. Speaker Notes Plan

| Item | Value |
| ---- | ----- |
| **Total duration** | 15-18 minutes |
| **Notes style** | bilingual, simple English, short sentences, plain Chinese translation |
| **Purpose** | report progress and justify the next research step |
| **Language rule** | English first, Chinese second, no long sentences, no difficult vocabulary |

---

## XI. Technical Constraints

- SVG viewBox must be `0 0 1280 720`.
- Use only colors, fonts, icons, and images listed in `spec_lock.md`.
- Use raw Unicode characters where needed; avoid HTML named entities.
- No `<style>`, no CSS classes, no `<foreignObject>`, no `rgba()`.
- Scientific figures must be embedded with no-crop behavior.
- Speaker notes must be generated into `notes/total.md`.
- Final export must use `finalize_svg.py` and `svg_to_pptx.py`, not a manual copy pipeline.
