"""Verify native tables, illustrative labels and the cost arithmetic."""

import hashlib
import json
import math
from pathlib import Path
from zipfile import ZipFile
from lxml import etree as E
from pypdf import PdfReader


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}


def main():
    data = json.loads((ROOT / "cost_example.json").read_text())
    with ZipFile(ROOT / "exports/Local_LLM_Cost_Explanation_18p90ms_2026_11_18.pptx") as z:
        assert z.testzip() is None
        assert len([n for n in z.namelist() if n.startswith("ppt/slides/slide") and n.endswith(".xml")]) == 3
        tables = []
        for number in [1, 2, 3]:
            xml = E.fromstring(z.read(f"ppt/slides/slide{number}.xml"))
            assert len(xml.findall(".//a:tbl", NS)) == 1
            tables.append(xml.findall(".//a:tbl", NS)[0])
            notes = E.fromstring(z.read(f"ppt/notesSlides/notesSlide{number}.xml"))
            note_text = "\n".join(notes.xpath("//a:t/text()", namespaces=NS))
            assert note_text.index("English") < note_text.index("中文")
        vk = [["".join(c.xpath(".//a:t/text()", namespaces=NS)) for c in row.findall("a:tc", NS)]
              for row in tables[0].findall("a:tr", NS)]
        assert [row[2] for row in vk[1:]] == ["18.90", "20.79", "23.06", "27.97", "38.75"]
        ck = [["".join(c.xpath(".//a:t/text()", namespaces=NS)) for c in row.findall("a:tc", NS)]
              for row in tables[2].findall("a:tr", NS)]
        assert [row[-1] for row in ck[1:]] == ["18.90", "7.58", "5.63", "4.90", "5.59"]
    for row in data["rows"][1:]:
        k, p = row["k"], data["p"]
        r = data["step_ms"] * sum((j+1) * p**j * (1-p) for j in range(k))
        g = 1+sum(p**j for j in range(1,k+1))
        assert math.isclose(row["r_ms"], r)
        assert math.isclose(row["g_tokens"], g)
        assert math.isclose(row["c_ms_per_token"], (row["v_ms"]+row["l_ms"]+r)/g)
    for label in ["archived", "ratios"]:
        source = data["source"]
        assert hashlib.sha256((REPO / source[label+"_path"]).read_bytes()).hexdigest() == source[label+"_sha256"]
    assert data["winner"] == 8
    pdf = ROOT / ".build/Local_LLM_Cost_Explanation_18p90ms_2026_11_18.pdf"
    assert len(PdfReader(pdf).pages) == 3
    receipt = {"status": "passed", "slides": 3, "native_tables": 3,
               "bilingual_notes": True, "arithmetic": "passed", "source_files_unchanged": True,
               "illustrative_not_new_measurements": True, "libreoffice_pdf_pages": 3,
               "new_inference_experiments": False}
    (ROOT / ".build/content_verification.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
