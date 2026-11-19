"""Update the short per-slide script for the 31-slide revision."""

import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.oxml.ns import qn
from docx.shared import Pt, RGBColor

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "exports/Local_LLM_PreDefense_Short_Script_EN_ZH_2026_11_12.docx"
STEM = "Local_LLM_PreDefense_Revised_Script_EN_ZH_2026_11_16"
OUT = ROOT / "exports" / f"{STEM}.docx"
if OUT.exists():
    raise FileExistsError(OUT)
pages = json.loads((ROOT / "notes/revised_script_en_zh_2026_11_16.json").read_text())
assert len(pages) == 31
doc = Document(SOURCE)
for node in list(doc._element.body):
    if node.tag != qn("w:sectPr"):
        doc._element.body.remove(node)
doc.core_properties.title = "Local LLM Inference Revised Speaker Notes"
doc.core_properties.subject = "Simple English and Chinese notes for the revised 31 slide presentation"
header = doc.sections[0].header.paragraphs[0]
header.text = "Local LLM Inference   Short Speaker Notes   November 16 2026"
header.runs[0].font.name = "Arial"
header.runs[0].font.size = Pt(9)
header.runs[0].font.color.rgb = RGBColor(0, 0, 0)

md = ["# Local LLM Inference Revised Speaker Notes", "", "November 16 2026", "",
      "31 slides. Main talk 1 to 26. Backup 27 to 29. References 30 and 31.",
      "English is followed immediately by Chinese on each slide.",
      "Slides 13, 16 and 24 contain numerical placeholders, not measured results.", ""]
for index, page in enumerate(pages):
    if index:
        doc.add_page_break()
    else:
        doc.add_paragraph("Local LLM Inference Revised Speaker Notes", "Title")
    title = " ".join(re.sub(r"[^A-Za-z0-9 ]", " ", page["title"].replace("'", "")).split())
    doc.add_heading(f"Slide {index + 1} {title}", 1)
    for lang, label in [("en", "English"), ("zh", "中文")]:
        h = doc.add_heading(label, 2)
        if lang == "zh":
            for run in h.runs:
                run.font.name = "Arial Unicode MS"
        for line in page[lang].splitlines():
            p = doc.add_paragraph(line)
            p.paragraph_format.keep_together = True
            p.paragraph_format.widow_control = True
            if lang == "zh":
                for run in p.runs:
                    run.font.name = "Arial Unicode MS"
                    run.font.size = Pt(12)
    md.extend([f"## Slide {index + 1} {page['title']}", "", "### English", "", page["en"],
               "", "### 中文", "", page["zh"], ""])
for style in doc.styles:
    for border in list(style.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
doc.save(OUT)
(ROOT / "notes" / f"{STEM}.md").write_text("\n".join(md))
count = lambda data: sum(len(re.findall(r"\b[\w'-]+\b", page["en"])) for page in data)
stats = {"docx": str(OUT), "pages_expected": 31, "main_english_words": count(pages[:26]),
         "all_english_words": count(pages), "language_order": "English then Chinese on each page",
         "new_slides": [7, 15, 19], "placeholder_slides": [13, 16, 24],
         "source_docx_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest()}
(ROOT / ".build/revision_20261116/script_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
