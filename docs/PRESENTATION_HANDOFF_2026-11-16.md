# Presentation Handoff: November 16, 2026

## Latest Files

Project: `projects/siliconllm_2026_11_11_predefense/`.

- PPTX: `exports/Local_LLM_PreDefense_Revised_2026_11_16_Complete.pptx`.
- Script: `exports/Local_LLM_PreDefense_Revised_Script_EN_ZH_2026_11_16.docx`.

Use these together. Both have 31 pages: 26 main, 3 backup, 2 references. Earlier PPTX and DOCX files remain unchanged. The script is English followed immediately by Chinese on every page, with short oral sentences. Main-talk English: 1,194 words; complete script: 1,350 words.

## Changes

Inserted generation procedure at slide 7, native mlx-lm target-cache recovery at slide 15, and evaluation metrics at slide 19. Clarified objective, novelty, target model/controller design, benchmark scope, baseline controls and timing definitions. All later pages and notes are renumbered. Purple design retained. Scientific chart assets and reported values unchanged.

The cost score still describes the current implementation. Restore/replay is not claimed as a universal mlx-lm requirement. Native trim is illustrated separately for a standard trimmable target KV cache; its numerical equivalence to the external experiment remains unverified. History reuse and adaptive length are prior ideas, not claimed as first-ever inventions.

## Evidence Boundaries

Slides 13, 16 and 24 retain explicit ESTIMATED PLACEHOLDER - NOT MEASURED banners. They are illustrative verification costs, a hypothetical controller example, and a planned full mlx-lm API comparison, respectively. They are not experiment results. Earlier placeholder-register page numbers 12/14/21 refer to the preserved 28-slide deck.

Reported results remain the adopted external study, including user corrections. External raw logs and final code are unavailable locally. No new experiments or independent reproduction occurred. Native model/cache results are not a full stock mlx-lm API benchmark. Original and feedback controllers remain separate. Savings mainly concern repeated failed repairs, not improved task success.

## Build and Checks

Native session callbacks: `revision_edit.mjs`; shared diagram functions: `procedure_supplement.mjs`. Narration inputs: `notes/revision_overrides_2026_11_16.json`, `notes/revised_script_en_zh_2026_11_16.json`. Script builder: `build_revised_script.py`. PPTX export: `finalize_revision.mjs`. Content validation: `verify_revision.py`.

All 31 native slide previews and 31 rendered DOCX pages visually inspected. Final PPTX package, geometry, font policy and first-party re-import passed. LibreOffice opened the complete PPTX and produced 31 PDF pages; slides 7/15/19/20/23/29 were visually checked in that PDF. All six embedded chart images match source bytes. Script text, page numbering, language order and slide-note alignment passed. Original deck, short script and source result hashes are unchanged. Native Microsoft PowerPoint was not tested.

QA receipts and internal previews: `.build/revision_20261116/`. The final exported PPTX is the `_Complete.pptx`, not the unembedded intermediate retained under `.build/`. Online export failed; final delivery is the locally exported reviewed state. Do not deliver the unused blank Spaces page or claim a complete cloud copy was saved there.

Final PPTX SHA256: `7acee750a094d07f762d7accd780058da12bca0e6a9d10f665b20b1ed6d92be5`.

Final DOCX SHA256: `1cfe6dced627681ec77b6c87072db8f849d37bbb15d9d6e8f1a605f7c306d35a`.
