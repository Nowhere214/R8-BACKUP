"""Check draft labels and source separation without claiming data reproduction."""

import hashlib
import json
from pathlib import Path
from xml.etree import ElementTree as ET
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parent
PPTX = ROOT / "exports/Local_LLM_Agentic_Workloads_PreDefense_Draft_2026_11_11_Reviewed.pptx"
NS = {"a": "http://schemas.openxmlformats.org/drawingml/2006/main"}
PLACEHOLDER_SLIDES = {12, 14, 21}
texts = []
with ZipFile(PPTX) as z:
    assert z.testzip() is None
    for n in range(1, 29):
        tree = ET.fromstring(z.read(f"ppt/slides/slide{n}.xml"))
        text = "\n".join(e.text or "" for e in tree.findall(".//a:t", NS))
        texts.append(text)
        assert f"{n:02d} / 28" in text
        assert ("ESTIMATED PLACEHOLDER - NOT MEASURED" in text) == (n in PLACEHOLDER_SLIDES)
        assert "Claude" not in text and "user updates" not in text
        if n in PLACEHOLDER_SLIDES:
            note = ET.fromstring(z.read(f"ppt/notesSlides/notesSlide{n}.xml"))
            assert "ILLUSTRATIVE" in " ".join(e.text or "" for e in note.findall(".//a:t", NS)).upper()
    assert "Pending" in texts[20]
    assert "100" in texts[20] and "85" in texts[20] and "75" in texts[20]
    assert "not a full stock mlx-lm API benchmark" in texts[19]
    assert all("REPORTED" in texts[i - 1] for i in [17, 18, 19, 20, 25, 26])

source = ROOT / "sources/reported_results.json"
original = ROOT.parents[1] / "projects/siliconllm_2026_10_22_formal_ppt169_20261021/sources/reported_results.json"
assert source.read_bytes() == original.read_bytes()
data = json.loads(source.read_text())
assert data["overall"]["speedup"] == [1.0, 1.45, 1.65]
assert data["repair"]["speedup"] == [1.0, 2.05, 2.86]

# Independently recalculate all illustrative policy scores and choices.
payload = json.loads((ROOT / ".build/slide_payload.json").read_text())
page14 = "\n".join(e["text"] for e in payload[13]["elements"] if e["type"] == "text")
choices = []
for p in [.55, .90, .98]:
    costs = {}
    for k, v in zip([2, 4, 8, 16], [1.10, 1.22, 1.48, 2.05]):
        probs = [p**j * (1 - p) for j in range(k)]
        replay = sum((j + 1) * prob for j, prob in enumerate(probs))
        gain = sum(p**j for j in range(k + 1))
        costs[k] = (v + .02 + replay) / gain
        assert f"{costs[k]:.2f}" in page14
    best = min(costs, key=costs.get)
    choices.append(0 if costs[best] >= .95 else best)
assert choices == [0, 4, 8]

pdf = PPTX.with_suffix(".pdf")
pdf_check = {"present": pdf.exists()}
if pdf.exists():
    from pypdf import PdfReader
    reader = PdfReader(pdf)
    assert len(reader.pages) == 28
    for i, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        assert f"{i:02d} / 28" in text
        assert ("ESTIMATED PLACEHOLDER" in text) == (i in PLACEHOLDER_SLIDES)
    pdf_check.update({"pages": 28, "text_status_checks": "pass"})

result = {"pptx": str(PPTX), "sha256": hashlib.sha256(PPTX.read_bytes()).hexdigest(),
          "slides": 28, "placeholder_slides": sorted(PLACEHOLDER_SLIDES),
          "reported_source_unchanged": True, "placeholder_arithmetic": "pass",
          "numerical_provenance": "Reported study aggregates and hypothetical examples remain separate.",
          "pdf": pdf_check, "native_powerpoint_opening": "not tested",
          "experiment_reproduction": "not performed"}
(ROOT / ".build/content_validation.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
