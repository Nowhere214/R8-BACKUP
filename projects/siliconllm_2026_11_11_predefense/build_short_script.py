"""Create a short bilingual script for the reviewed 28-slide deck."""

import hashlib
import json
import re
from pathlib import Path

from docx import Document
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parent
STEM = "Local_LLM_PreDefense_Short_Script_EN_ZH_2026_11_12"
OUT = ROOT / "exports" / f"{STEM}.docx"
if OUT.exists():
    raise FileExistsError(f"Keep the existing file: {OUT}")

pages = json.loads((ROOT / ".build/slide_payload.json").read_text())
notes = json.loads((ROOT / "notes/short_script_en_zh.json").read_text())
assert len(pages) == len(notes) == 28
doc = Document()
doc.core_properties.title = "Local LLM Inference Speaker Notes"
doc.core_properties.subject = "Short per slide English and Chinese notes for the November 11 pre defense draft"
doc.core_properties.author = "Wang"
section = doc.sections[0]
section.page_width, section.page_height = Inches(8.5), Inches(11)
section.top_margin, section.bottom_margin = Inches(.8), Inches(.65)
section.left_margin = section.right_margin = Inches(.8)

for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2"]:
    style = doc.styles[name]
    style.font.name = "Arial"
    style.font.color.rgb = RGBColor(0, 0, 0)
    style.element.get_or_add_rPr().get_or_add_rFonts().set(qn("w:eastAsia"), "Arial Unicode MS")
normal = doc.styles["Normal"]
normal.font.size = Pt(14)
normal.paragraph_format.line_spacing = 1.15
normal.paragraph_format.space_after = Pt(7)
doc.styles["Title"].font.size = Pt(21)
doc.styles["Title"].paragraph_format.space_after = Pt(16)
doc.styles["Heading 1"].font.size = Pt(18)
doc.styles["Heading 1"].paragraph_format.space_after = Pt(12)
doc.styles["Heading 2"].font.size = Pt(13)
doc.styles["Heading 2"].paragraph_format.space_before = Pt(18)
doc.styles["Heading 2"].paragraph_format.space_after = Pt(8)
doc.styles["Heading 2"].paragraph_format.keep_with_next = True

header = section.header.paragraphs[0]
header.text = "Local LLM Inference   Short Speaker Notes   November 12 2026"
header.runs[0].font.size = Pt(9)
footer = section.footer.paragraphs[0]
footer.alignment = 2
field = OxmlElement("w:fldSimple")
field.set(qn("w:instr"), "PAGE")
footer._p.append(field)

md = ["# Local LLM Inference Short Speaker Notes", "", "November 12 2026", "",
      "For the reviewed November 11 deck. Slides 1 to 23 are the main talk, 24 to 26 are backup, and 27 to 28 are references.",
      "English is immediately followed by Chinese on each slide. Slides 12, 14 and 21 use numerical placeholders, not measurements.", ""]
for i, (page, note) in enumerate(zip(pages, notes)):
    assert note["slide"] == page["number"] == i + 1
    if i:
        doc.add_page_break()
    else:
        doc.add_paragraph("Local LLM Inference Speaker Notes", "Title")
    clean_title = " ".join(re.sub(r"[^A-Za-z0-9 ]", " ", page["title"].replace("'", "")).split())
    doc.add_heading(f"Slide {i + 1} {clean_title}", 1)
    for lang, label in [("en", "English"), ("zh", "中文")]:
        heading = doc.add_heading(label, 2)
        if lang == "zh":
            for run in heading.runs:
                run.font.name = "Arial Unicode MS"
        for line in note[lang].splitlines():
            p = doc.add_paragraph(line)
            p.paragraph_format.keep_together = True
            p.paragraph_format.widow_control = True
            if lang == "zh":
                for run in p.runs:
                    run.font.name = "Arial Unicode MS"
                    run.font.size = Pt(12)
    md.extend([f"## Slide {i + 1} {page['title']}", "", "### English", "", note["en"], "", "### 中文", "", note["zh"], ""])

for style in doc.styles:
    for border in list(style.element.iter(qn("w:pBdr"))):
        border.getparent().remove(border)
doc.save(OUT)
(ROOT / "notes" / f"{STEM}.md").write_text("\n".join(md))
count = lambda selection: sum(len(re.findall(r"\b[\w'-]+\b", n["en"])) for n in selection)
stats = {"docx": str(OUT), "main_english_words": count(notes[:23]), "all_english_words": count(notes),
         "pages_expected": 28, "main_slides": 23, "backup_slides": [24, 25, 26], "reference_slides": [27, 28],
         "placeholder_slides": [12, 14, 21], "language_order": "English then Chinese for each slide",
         "deck_sha256": hashlib.sha256((ROOT / "exports/Local_LLM_Agentic_Workloads_PreDefense_Draft_2026_11_11_Reviewed.pptx").read_bytes()).hexdigest()}
(ROOT / ".build/short_script_stats.json").write_text(json.dumps(stats, indent=2))
print(json.dumps(stats, indent=2))
