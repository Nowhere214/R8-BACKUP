"""Map existing narration to the revised slide order, retaining original data."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".build/revision_20261116"
BUILD.mkdir(parents=True, exist_ok=True)
old_pages = json.loads((ROOT / ".build/slide_payload.json").read_text())
old_notes = json.loads((ROOT / "notes/short_script_en_zh.json").read_text())
delta = json.loads((ROOT / "notes/revision_overrides_2026_11_16.json").read_text())
pages = []
for page, note in zip(old_pages, old_notes):
    old = page["number"]
    narration = delta["old"].get(str(old), note)
    pages.append({"old": old, "title": page["title"], "en": narration["en"], "zh": narration["zh"]})
    key = {6: "generation", 13: "recovery", 16: "metrics"}.get(old)
    if key:
        pages.append({"new": key, **delta["new"][key]})
for n, page in enumerate(pages, 1):
    page["number"] = n
assert len(pages) == 31
assert [p["number"] for p in pages if p.get("old") in (12, 14, 21)] == [13, 16, 24]
text = json.dumps(pages, ensure_ascii=False, indent=2)
(ROOT / "notes/revised_script_en_zh_2026_11_16.json").write_text(text)
(BUILD / "revision_pages.json").write_text(text)
(ROOT / "revision_notes_data.mjs").write_text("export const pages = " + text + ";\n")
manifest = {"total_slides": 31, "main_slides": 26,
            "placeholder_slides": [13, 16, 24], "backup_slides": [27, 28, 29],
            "references": [30, 31], "new_slides": [7, 15, 19],
            "old_to_new": {p["old"]: p["number"] for p in pages if "old" in p},
            "reported_results_sha256": hashlib.sha256((ROOT / "sources/reported_results.json").read_bytes()).hexdigest(),
            "source_pptx_sha256": hashlib.sha256((ROOT / "exports/Local_LLM_Agentic_Workloads_PreDefense_Draft_2026_11_11_Reviewed.pptx").read_bytes()).hexdigest()}
(BUILD / "revision_manifest.json").write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2))
