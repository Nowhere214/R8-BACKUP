"""Check page mapping, retained charts, narration and placeholder boundaries."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import posixpath
import re
import unicodedata
import xml.etree.ElementTree as ET
from zipfile import ZipFile

from pypdf import PdfReader

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / ".build/revision_20261116"
PPT = ROOT / "exports/Local_LLM_PreDefense_Revised_2026_11_16_Complete.pptx"
DOC = ROOT / "exports/Local_LLM_PreDefense_Revised_Script_EN_ZH_2026_11_16.docx"
pages = json.loads((ROOT / "notes/revised_script_en_zh_2026_11_16.json").read_text())
manifest = json.loads((BUILD / "revision_manifest.json").read_text())
stats = json.loads((BUILD / "script_stats.json").read_text())
digest = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
normal = lambda text: re.sub(r"\s+", "", unicodedata.normalize("NFKC", text))
assert digest(ROOT / "exports/Local_LLM_Agentic_Workloads_PreDefense_Draft_2026_11_11_Reviewed.pptx") == manifest["source_pptx_sha256"]
assert digest(ROOT / "sources/reported_results.json") == manifest["reported_results_sha256"]
assert digest(ROOT / "exports/Local_LLM_PreDefense_Short_Script_EN_ZH_2026_11_12.docx") == stats["source_docx_sha256"]
ns = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main",
      "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
      "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships"}
slide_texts, placeholders, pictures, table_owners = [], [], {}, []
with ZipFile(PPT) as package:
    assert package.testzip() is None
    for number, page in enumerate(pages, 1):
        xml = ET.fromstring(package.read(f"ppt/slides/slide{number}.xml"))
        texts = [e.text or "" for e in xml.findall(".//a:t", ns)]
        full = "\n".join(texts)
        slide_texts.append(full)
        assert page["title"] in full
        assert f"{number:02d} / 31" in full
        assert "numerical placeholders on slides 13, 16 and 24" in full
        assert not re.search(r"\d\d / 28", full)
        if "ESTIMATED PLACEHOLDER - NOT MEASURED" in full:
            placeholders.append(number)
        if xml.findall(".//a:tbl", ns):
            table_owners.append(number)
        rels = {e.attrib["Id"]: e.attrib for e in ET.fromstring(package.read(f"ppt/slides/_rels/slide{number}.xml.rels"))}
        hashes = []
        for blip in xml.findall(".//p:pic//a:blip", ns):
            rel = rels[blip.attrib[f"{{{ns['r']}}}embed"]]
            assert rel.get("TargetMode") != "External"
            target = posixpath.normpath(str(PurePosixPath("ppt/slides") / rel["Target"])).lstrip("/")
            hashes.append(hashlib.sha256(package.read(target)).hexdigest())
        if hashes:
            pictures[number] = hashes
        notes = ET.fromstring(package.read(f"ppt/notesSlides/notesSlide{number}.xml"))
        note_text = "\n".join(e.text or "" for e in notes.findall(".//a:t", ns))
        assert note_text.index("English") < note_text.index("中文")
        assert normal(page["en"]) in normal(note_text)
        assert normal(page["zh"]) in normal(note_text)
assert placeholders == [13, 16, 24]
assert table_owners == [9, 12, 17, 18, 19]
keys = {20: "overall", 21: "repair", 22: "agent", 23: "native", 28: "fixed_lengths", 29: "feedback"}
assert set(pictures) == set(keys)
for number, key in keys.items():
    assert pictures[number] == [digest(ROOT / f"images/{key}.png")]
with ZipFile(DOC) as package:
    assert package.testzip() is None
script_pdf = next((BUILD / "script_preview").glob("*.pdf"))
pdf = PdfReader(script_pdf)
assert len(pdf.pages) == 31
for number, (pdf_page, source) in enumerate(zip(pdf.pages, pages), 1):
    text = pdf_page.extract_text()
    assert re.search(rf"\bSlide {number}\b", text)
    assert text.index("English") < text.index("中文")
    for lang in ["en", "zh"]:
        for line in source[lang].splitlines():
            assert normal(line) in normal(text), (number, lang, line)

report = {"slides": 31, "script_pdf_pages": 31, "slide_and_script_notes_match": True,
          "placeholder_slides": placeholders, "native_table_slides": table_owners,
          "embedded_chart_slides": sorted(pictures), "all_chart_bytes_match_existing_assets": True,
          "source_pptx_unchanged": True, "source_script_unchanged": True, "reported_data_unchanged": True,
          "script_text_and_language_order": "passed", "pptx_sha256": digest(PPT), "docx_sha256": digest(DOC)}
lo_pdf = BUILD / "Local_LLM_PreDefense_Revised_2026_11_16_Complete.pdf"
if lo_pdf.exists():
    exported = PdfReader(lo_pdf)
    assert len(exported.pages) == 31
    for n, pdf_page in enumerate(exported.pages, 1):
        text = pdf_page.extract_text()
        assert f"{n:02d} / 31" in text
    report["libreoffice_pdf_pages"] = 31
(BUILD / "content_verification.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
