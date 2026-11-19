from __future__ import annotations

import copy
import mimetypes
import shutil
import zipfile
from pathlib import Path
import xml.etree.ElementTree as ET


SRC = Path("/Users/laiqixiao/Downloads/ZEMI05.20.WANG.pptx")
OUT = Path("/Users/laiqixiao/Downloads/ZEMI05.20.WANG.SiliconLLM.pptx")

REPO = Path("/Users/laiqixiao/Documents/SiliconLLM")
ASSETS = REPO / "benches" / "presentation_assets"
PHASE0 = REPO / "benches" / "phase0_n3.png"

PML = "http://schemas.openxmlformats.org/presentationml/2006/main"
AML = "http://schemas.openxmlformats.org/drawingml/2006/main"
RML = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PKG_REL = "http://schemas.openxmlformats.org/package/2006/relationships"
CT = "http://schemas.openxmlformats.org/package/2006/content-types"
EP = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"

NS = {"p": PML, "a": AML, "r": RML, "rel": PKG_REL, "ct": CT}

ET.register_namespace("p", PML)
ET.register_namespace("a", AML)
ET.register_namespace("r", RML)
ET.register_namespace("", PKG_REL)


SLIDES = [
    {
        "template": 1,
        "title": "Efficient LLM Inference on Apple Silicon",
        "subtitle": "SiliconLLM Phase 0 Benchmark Results\nWANG Peiran\n2026.05.20",
    },
    {
        "template": 2,
        "body": [
            "Talk Outline:",
            "Introduction: Why local LLM inference matters",
            "Method: Controlled benchmark design on Apple Silicon",
            "Result: Preliminary Phase 0 measurements",
            "Discussion: What the results imply and what comes next",
            "",
            "Target: 15-20 minutes",
        ],
    },
    {
        "template": 3,
        "title": "Introduction: Why This Research?",
        "body": [
            "Motivation:",
            "- LLM inference is becoming a major system cost after training.",
            "- Local inference on consumer hardware is increasingly important for privacy, latency, and accessibility.",
            "- Apple Silicon is widely available, but its LLM serving behavior is less studied than CUDA GPUs.",
            "",
            "Research direction:",
            "- Use controlled measurement before building new optimizations.",
        ],
    },
    {
        "template": 3,
        "title": "Research Problem",
        "body": [
            "Problem:",
            "- Production engines such as mlx-lm are fast, but their speed comes from many stacked optimizations.",
            "- A single end-to-end number does not explain where the performance gap comes from.",
            "",
            "Key question:",
            "- Which part of the inference pipeline creates the largest gap between an unoptimized implementation and a production engine?",
        ],
    },
    {
        "template": 3,
        "title": "Project Framing: SiliconLLM",
        "body": [
            "SiliconLLM:",
            "- A personal research branch based on the open-source tiny-llm teaching project.",
            "- tiny-llm is used as a controlled experimental platform, not claimed as original project infrastructure.",
            "- The current stage focuses on measurement, reproducibility, and baseline construction.",
            "",
            "Why this platform:",
            "- The implementation is small enough to open the black box of LLM inference.",
        ],
    },
    {
        "template": 3,
        "title": "Research Questions",
        "body": [
            "Q1: Where is the performance gap?",
            "- Prefill, decode, memory usage, or end-to-end latency?",
            "",
            "Q2: Which optimizations matter most?",
            "- KV cache, quantized matmul, Flash Attention, batching, and chunked prefill.",
            "",
            "Q3: Does Apple Silicon behave like CUDA reports?",
            "- Compare optimization gains and explain hardware-specific differences.",
        ],
    },
    {
        "template": 3,
        "title": "Method: Experimental Setup",
        "body": [
            "Hardware:",
            "- MacBook Pro with Apple Silicon and 18 GB unified memory.",
            "",
            "Models:",
            "- Qwen2-0.5B-Instruct",
            "- Qwen2-1.5B-Instruct",
            "- Qwen2-7B-Instruct for production-engine baseline",
            "",
            "Engines:",
            "- SiliconLLM Stage 1 reference implementation",
            "- mlx-lm production engine",
        ],
    },
    {
        "template": 3,
        "title": "Method: Benchmark Design",
        "body": [
            "Controlled comparison:",
            "- Same machine",
            "- Same model family",
            "- Same prompt and generation workload",
            "- N=3 repeated runs for each configuration",
            "",
            "Metrics:",
            "- Prefill throughput",
            "- Decode throughput",
            "- End-to-end wall time",
            "- Peak memory usage",
        ],
    },
    {
        "template": 3,
        "title": "Method: Why Separate Prefill and Decode?",
        "body": [
            "Prefill:",
            "- Processes the input prompt in parallel.",
            "- Often dominated by large matrix operations.",
            "",
            "Decode:",
            "- Generates tokens step by step.",
            "- More sensitive to KV cache, attention recomputation, Python overhead, and kernel launch overhead.",
            "",
            "This separation is essential for diagnosing the bottleneck.",
        ],
    },
    {
        "template": 3,
        "title": "Result: Overall Throughput",
        "body": [
            "Main observation:",
            "- The production engine is consistently faster.",
            "- The gap is much larger in decode than in prefill.",
            "- The Stage 1 reference implementation cannot safely provide a measured 7B datapoint on this 18 GB machine.",
        ],
        "image": PHASE0,
    },
    {
        "template": 3,
        "title": "Result: The Gap Is Concentrated in Decode",
        "body": [
            "Measured speedup of mlx-lm over Stage 1 reference:",
            "- Prefill gap: about 1.3-1.6x",
            "- Decode gap: about 29-33x",
            "",
            "Interpretation:",
            "- The main bottleneck is not simply raw matrix throughput.",
            "- It is concentrated in autoregressive generation.",
        ],
        "image": ASSETS / "phase0_gap_ratios.png",
    },
    {
        "template": 3,
        "title": "Result: End-to-End Wall Time",
        "body": [
            "Same benchmark prompt:",
            "- Qwen2 0.5B Stage 1 ref: 66.8 seconds",
            "- Qwen2 0.5B mlx-lm: 2.5 seconds",
            "- Qwen2 1.5B Stage 1 ref: 172.7 seconds",
            "- Qwen2 1.5B mlx-lm: 5.9 seconds",
            "",
            "This converts throughput gaps into user-visible latency.",
        ],
        "image": ASSETS / "phase0_wall_time.png",
    },
    {
        "template": 3,
        "title": "Result: Memory and Feasibility Boundary",
        "body": [
            "Peak memory:",
            "- Stage 1 ref grows from 1.9 GB to 4.7 GB between 0.5B and 1.5B.",
            "- mlx-lm runs Qwen2 7B at about 4.8 GB peak memory.",
            "",
            "Conclusion:",
            "- Optimization affects not only speed, but also whether the model can run at all.",
        ],
        "image": ASSETS / "phase0_memory.png",
    },
    {
        "template": 3,
        "title": "Result: Repeatability Check",
        "body": [
            "Why N=3 matters:",
            "- Earlier N=1 measurements produced a misleading prefill trend.",
            "- Repeating the benchmark changed the interpretation.",
            "- The current Phase 0 results report mean and standard deviation.",
            "",
            "Methodological finding:",
            "- High-overhead inference paths should not be evaluated with a single run.",
        ],
        "image": ASSETS / "phase0_repeatability_cv.png",
    },
    {
        "template": 3,
        "title": "Result: 7B Boundary for Stage 1 Reference",
        "body": [
            "The missing 7B point is itself informative:",
            "- It was not reported as a measured result because the memory estimate approaches the 18 GB machine limit.",
            "- Extrapolation suggests decode would be around 1 token/sec.",
            "- This motivates later optimization work before large-model experiments.",
        ],
        "image": ASSETS / "phase0_stage1_ref_7b_boundary.png",
    },
    {
        "template": 3,
        "title": "Discussion: What Explains the Gap?",
        "body": [
            "Likely causes:",
            "- No KV cache in the Stage 1 reference path",
            "- Repeated attention computation during decode",
            "- Less optimized quantized matrix multiplication",
            "- Python-level overhead and less kernel fusion",
            "",
            "Important nuance:",
            "- The data shows where the gap appears; profiler work is needed to assign exact percentages.",
        ],
    },
    {
        "template": 3,
        "title": "Discussion: Next Steps",
        "body": [
            "Immediate next steps:",
            "- Add llama.cpp Metal as a third production baseline.",
            "- Run workload-length scans for prompt and generation length.",
            "- Use MLX profiler to decompose the decode gap.",
            "",
            "Longer-term plan:",
            "- Implement Stage 1 components in src/tiny_llm.",
            "- Add KV cache, quantization, and attention optimizations.",
            "- Measure the marginal contribution of each optimization.",
        ],
    },
    {
        "template": 12,
        "title": "THANK YOU FOR LISTENING",
    },
]


def qn(ns: str, name: str) -> str:
    return f"{{{ns}}}{name}"


def read_xml(files: dict[str, bytes], path: str) -> ET.Element:
    return ET.fromstring(files[path])


def xml_bytes(root: ET.Element) -> bytes:
    return ET.tostring(root, encoding="UTF-8", xml_declaration=True)


def set_xfrm(shape: ET.Element, x: int, y: int, cx: int, cy: int) -> None:
    xfrm = shape.find(".//a:xfrm", NS)
    if xfrm is None:
        return
    off = xfrm.find("a:off", NS)
    ext = xfrm.find("a:ext", NS)
    if off is not None:
        off.set("x", str(x))
        off.set("y", str(y))
    if ext is not None:
        ext.set("cx", str(cx))
        ext.set("cy", str(cy))


def find_text_shapes(slide: ET.Element) -> list[ET.Element]:
    shapes = []
    for shape in slide.findall(".//p:sp", NS):
        if shape.find(".//a:t", NS) is not None:
            shapes.append(shape)
    return shapes


def replace_text(shape: ET.Element, lines: list[str], size: int | None = None) -> None:
    tx_body = shape.find("p:txBody", NS)
    if tx_body is None:
        return
    body_pr = tx_body.find("a:bodyPr", NS)
    lst_style = tx_body.find("a:lstStyle", NS)
    sample_rpr = copy.deepcopy(shape.find(".//a:rPr", NS))
    if sample_rpr is None:
        sample_rpr = ET.Element(qn(AML, "rPr"))
    if size is not None:
        sample_rpr.set("sz", str(size))

    for child in list(tx_body):
        tx_body.remove(child)
    if body_pr is not None:
        tx_body.append(body_pr)
    if lst_style is not None:
        tx_body.append(lst_style)

    for line in lines:
        p = ET.Element(qn(AML, "p"))
        r = ET.SubElement(p, qn(AML, "r"))
        r.append(copy.deepcopy(sample_rpr))
        t = ET.SubElement(r, qn(AML, "t"))
        t.text = line
        tx_body.append(p)


def set_slide_text(slide: ET.Element, spec: dict) -> None:
    shapes = find_text_shapes(slide)
    if spec["template"] == 1:
        replace_text(shapes[0], [spec["title"]], size=3000)
        replace_text(shapes[1], spec["subtitle"].splitlines(), size=1800)
        return
    if spec["template"] == 2:
        replace_text(shapes[0], spec["body"], size=2100)
        set_xfrm(shapes[0], 700000, 1300000, 10800000, 4200000)
        return
    if spec["template"] == 12:
        replace_text(shapes[0], [spec["title"]], size=3200)
        return

    title_shape = shapes[0]
    body_shape = shapes[1]
    replace_text(title_shape, [spec["title"]], size=2700)
    replace_text(body_shape, spec["body"], size=1650 if "image" in spec else 1950)
    set_xfrm(title_shape, 360000, 0, 11000000, 720000)
    if "image" in spec:
        set_xfrm(body_shape, 360000, 980000, 3600000, 4850000)
    else:
        set_xfrm(body_shape, 360000, 1100000, 11100000, 4600000)


def make_picture(image_rel_id: str, name: str, x: int, y: int, cx: int, cy: int, pic_id: int) -> ET.Element:
    pic = ET.Element(qn(PML, "pic"))
    nv = ET.SubElement(pic, qn(PML, "nvPicPr"))
    c_nv_pr = ET.SubElement(nv, qn(PML, "cNvPr"))
    c_nv_pr.set("id", str(pic_id))
    c_nv_pr.set("name", name)
    ET.SubElement(nv, qn(PML, "cNvPicPr"))
    ET.SubElement(nv, qn(PML, "nvPr"))

    blip_fill = ET.SubElement(pic, qn(PML, "blipFill"))
    blip = ET.SubElement(blip_fill, qn(AML, "blip"))
    blip.set(qn(RML, "embed"), image_rel_id)
    stretch = ET.SubElement(blip_fill, qn(AML, "stretch"))
    ET.SubElement(stretch, qn(AML, "fillRect"))

    sp_pr = ET.SubElement(pic, qn(PML, "spPr"))
    xfrm = ET.SubElement(sp_pr, qn(AML, "xfrm"))
    off = ET.SubElement(xfrm, qn(AML, "off"))
    off.set("x", str(x))
    off.set("y", str(y))
    ext = ET.SubElement(xfrm, qn(AML, "ext"))
    ext.set("cx", str(cx))
    ext.set("cy", str(cy))
    prst = ET.SubElement(sp_pr, qn(AML, "prstGeom"))
    prst.set("prst", "rect")
    ET.SubElement(prst, qn(AML, "avLst"))
    return pic


def add_image_to_slide(slide: ET.Element, rels: ET.Element, image_name: str, pic_id: int) -> None:
    rel_id = "rIdImage1"
    rel = ET.SubElement(rels, qn(PKG_REL, "Relationship"))
    rel.set("Id", rel_id)
    rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image")
    rel.set("Target", f"../media/{image_name}")

    sp_tree = slide.find(".//p:spTree", NS)
    if sp_tree is None:
        return
    sp_tree.append(
        make_picture(
            rel_id,
            image_name,
            x=4300000,
            y=1060000,
            cx=7500000,
            cy=4550000,
            pic_id=pic_id,
        )
    )


def add_content_type(root: ET.Element, part_name: str, content_type: str) -> None:
    for override in root.findall("ct:Override", NS):
        if override.get("PartName") == part_name:
            override.set("ContentType", content_type)
            return
    override = ET.SubElement(root, qn(CT, "Override"))
    override.set("PartName", part_name)
    override.set("ContentType", content_type)


def ensure_png_default(root: ET.Element) -> None:
    for default in root.findall("ct:Default", NS):
        if default.get("Extension") == "png":
            return
    default = ET.SubElement(root, qn(CT, "Default"))
    default.set("Extension", "png")
    default.set("ContentType", "image/png")


def main() -> None:
    with zipfile.ZipFile(SRC, "r") as zin:
        files = {name: zin.read(name) for name in zin.namelist()}

    content_types = read_xml(files, "[Content_Types].xml")
    ensure_png_default(content_types)

    app_props = read_xml(files, "docProps/app.xml")
    slides_prop = app_props.find(f"{{{EP}}}Slides")
    if slides_prop is not None:
        slides_prop.text = str(len(SLIDES))

    pres = read_xml(files, "ppt/presentation.xml")
    pres_rels = read_xml(files, "ppt/_rels/presentation.xml.rels")
    ext_lst = pres.find("p:extLst", NS)
    if ext_lst is not None:
        pres.remove(ext_lst)
    sld_id_lst = pres.find("p:sldIdLst", NS)
    if sld_id_lst is None:
        raise RuntimeError("presentation.xml has no p:sldIdLst")
    for child in list(sld_id_lst):
        sld_id_lst.remove(child)

    old_slide_names = [name for name in files if name.startswith("ppt/slides/slide") and name.endswith(".xml")]
    old_slide_rel_names = [name for name in files if name.startswith("ppt/slides/_rels/slide") and name.endswith(".xml.rels")]
    for name in old_slide_names + old_slide_rel_names:
        files.pop(name, None)

    for rel in list(pres_rels):
        if rel.get("Type") == "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide":
            pres_rels.remove(rel)

    image_counter = 6
    for idx, spec in enumerate(SLIDES, start=1):
        template_num = spec["template"]
        with zipfile.ZipFile(SRC, "r") as zin:
            slide = ET.fromstring(zin.read(f"ppt/slides/slide{template_num}.xml"))
            rels = ET.fromstring(zin.read(f"ppt/slides/_rels/slide{template_num}.xml.rels"))

        for rel in list(rels):
            if rel.get("Type") == "http://schemas.openxmlformats.org/officeDocument/2006/relationships/notesSlide":
                rels.remove(rel)

        set_slide_text(slide, spec)

        if "image" in spec:
            image_path = Path(spec["image"])
            image_name = f"image{image_counter}.png"
            files[f"ppt/media/{image_name}"] = image_path.read_bytes()
            add_image_to_slide(slide, rels, image_name, pic_id=100 + idx)
            image_counter += 1

        files[f"ppt/slides/slide{idx}.xml"] = xml_bytes(slide)
        files[f"ppt/slides/_rels/slide{idx}.xml.rels"] = xml_bytes(rels)
        add_content_type(
            content_types,
            f"/ppt/slides/slide{idx}.xml",
            "application/vnd.openxmlformats-officedocument.presentationml.slide+xml",
        )

        rel_id = f"rId{100 + idx}"
        rel = ET.SubElement(pres_rels, qn(PKG_REL, "Relationship"))
        rel.set("Id", rel_id)
        rel.set("Type", "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide")
        rel.set("Target", f"slides/slide{idx}.xml")

        sld_id = ET.SubElement(sld_id_lst, qn(PML, "sldId"))
        sld_id.set("id", str(255 + idx))
        sld_id.set(qn(RML, "id"), rel_id)

    files["[Content_Types].xml"] = xml_bytes(content_types)
    files["ppt/presentation.xml"] = xml_bytes(pres)
    files["ppt/_rels/presentation.xml.rels"] = xml_bytes(pres_rels)
    files["docProps/app.xml"] = xml_bytes(app_props)

    backup = OUT.with_suffix(".backup.pptx")
    if OUT.exists():
        shutil.copy2(OUT, backup)

    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zout:
        for name, data in files.items():
            zout.writestr(name, data)

    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
