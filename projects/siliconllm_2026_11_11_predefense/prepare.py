"""Prepare a pre-defense draft, keeping reported data and placeholders separate."""

import json
import math
import shutil
import sys
from html import escape
from pathlib import Path

from PIL import ImageFont

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
OLD = REPO / "projects/siliconllm_2026_10_22_formal_ppt169_20261021"
for directory in ["sources", "images", "svg_output", "notes", ".build", "exports"]:
    (ROOT / directory).mkdir(parents=True, exist_ok=True)

shutil.copyfile(OLD / "sources/reported_results.json", ROOT / "sources/reported_results.json")
DATA = json.loads((ROOT / "sources/reported_results.json").read_text())
PLACEHOLDERS = {
    "status": "Illustrative numerical placeholders, NOT measurements or fitted predictions.",
    "verification": {"k": [2, 4, 8, 16], "relative_time": [1.10, 1.22, 1.48, 2.05],
                     "ordinary_step": 1.0, "lookup": 0.02},
    "full_mlx_lm": {"methods": ["Stock mlx-lm greedy", "mlx-lm + fixed history", "mlx-lm + adaptive history"],
                    "relative_time": [100.0, 85.0, 75.0],
                    "purpose": "Layout and comparison design only; integration and timing pending."},
}
(ROOT / "sources/illustrative_placeholders.json").write_text(json.dumps(PLACEHOLDERS, indent=2))

SKILL = Path("/Users/laiqixiao/.codex/skills/scientific-visualization")
sys.path.insert(0, str(SKILL / "scripts"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from figure_export import save_publication_figure
from style_presets import apply_publication_style

apply_publication_style("default")
plt.rcParams.update({"font.family": "Arial", "font.size": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 14, "ytick.labelsize": 15})
BG, PURPLE, INK, MUTED = "#F4EFFA", "#3E1485", "#1F1633", "#4B405D"
BLUE, GREEN, ORANGE, GRAY = "#0072B2", "#009E73", "#D55E00", "#7A7A7A"


def figure(key, field="speedup"):
    row = DATA[key]
    fig, ax = plt.subplots(figsize=(8.2, 4.5))
    fig.set_layout_engine("none")
    values = row[field]
    bars = ax.barh(row["methods"], values, color=[GRAY, BLUE, GREEN, ORANGE][:len(values)], height=0.53)
    ax.invert_yaxis()
    ax.set_xlim(0, max(values) * 1.25)
    ax.set_xlabel("Sum of per-task mean agent times (s)" if field == "seconds"
                  else "Speedup over ordinary decoding")
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.6)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=10)
    for bar, value in zip(bars, values):
        label = f"{value:.2f} s" if field == "seconds" else f"{value:.2f}x"
        if key == "repair" and value == 2.86:
            label = "~2.86x"
        ax.text(value + max(values) * 0.03, bar.get_y() + bar.get_height() / 2,
                label, va="center", fontsize=16, fontweight="bold")
    fig.subplots_adjust(left=0.30, right=0.98, bottom=0.20, top=0.97)
    save_publication_figure(fig, ROOT / "images" / key, formats=["png", "pdf", "svg"], dpi=300)
    plt.close(fig)


for key in ["overall", "repair", "native", "fixed_lengths", "feedback"]:
    figure(key)
figure("agent", "seconds")

REFS = {
    1: ("Leviathan, Kalman and Matias. Fast Inference from Transformers via Speculative Decoding. ICML 2023.",
        "https://arxiv.org/abs/2211.17192"),
    2: ("Apoorv Saxena. Prompt Lookup Decoding. Software repository, 2023.",
        "https://github.com/apoorvumang/prompt-lookup-decoding"),
    3: ("Wang et al. EfficientEdit: Accelerating Code Editing via Edit-Oriented Speculative Decoding. ASE 2025.",
        "https://arxiv.org/abs/2506.02780"),
    4: ("Lee et al. AgSpec: Pushing the Limits of Retrieval-Based Speculative Decoding in Coding Agent Pipelines. 2026.",
        "https://arxiv.org/abs/2610.01108"),
    5: ("Mamou et al. Dynamic Speculation Lookahead Accelerates Speculative Decoding of Large Language Models. 2024.",
        "https://arxiv.org/abs/2405.04304"),
    6: ("MLX LM. Language model inference and fine-tuning on Apple Silicon. Software repository.",
        "https://github.com/ml-explore/mlx-lm"),
    7: ("Austin et al. Program Synthesis with Large Language Models. MBPP, 2021.",
        "https://arxiv.org/abs/2108.07732"),
    8: ("Chen et al. Evaluating Large Language Models Trained on Code. HumanEval, 2021.",
        "https://arxiv.org/abs/2107.03374"),
    9: ("Alex Chi. tiny-llm: A course on LLM serving. Software repository.",
        "https://github.com/skyzh/tiny-llm"),
}
(ROOT / "sources/references.json").write_text(json.dumps(REFS, indent=2))

FONTS = Path("/System/Library/Fonts/Supplemental")


def font(size, bold=False, mono=False):
    name = "Courier New" if mono else "Arial"
    return ImageFont.truetype(str(FONTS / f"{name}{' Bold' if bold else ''}.ttf"), int(size * 4))


PAGES = []


class Slide:
    def __init__(self, title, section, notes, refs=(), placeholder=False, reported=False):
        self.number = len(PAGES) + 1
        self.title, self.section, self.notes = title, section, notes
        self.refs, self.placeholder, self.reported = refs, placeholder, reported
        self.elements = []
        self.rect(0, 0, 1280, 720, BG)
        self.rect(0, 0, 1280, 118, PURPLE)
        self.text(title, 52, 70, 34, color="#FFFFFF", bold=True, max_width=1090)
        self.rect(1178, 0, 102, 118, "#9E89C2")
        self.text(str(self.number), 1200 if self.number >= 10 else 1208, 75, 38, color="#FFFFFF", bold=True, max_width=65)
        if placeholder:
            self.rect(52, 136, 1176, 42, "#FFF1D6")
            self.text("ESTIMATED PLACEHOLDER - NOT MEASURED", 68, 164, 22, bold=True, color="#9B4300")
        else:
            self.text(section.upper(), 56, 153, 17, color="#6B4CA3", bold=True)
        self.line(54, 657, 1226, 657, "#D6CBE8", 1)
        label = "Working draft - numerical placeholders on slides 12, 14 and 21"
        self.text(label, 56, 688, 14, color="#6F6280")
        PAGES.append(self)

    def text(self, text, x, y, size=26, color=INK, bold=False, mono=False, max_width=1160):
        width = font(size, bold, mono).getlength(text) / 4
        if width > max_width or x + width > (1278 if y < 118 else 1230):
            raise ValueError(f"Slide {self.number}: text overflow ({width}): {text}")
        self.elements.append({"type": "text", "text": text, "font": "Courier New" if mono else "Arial",
                              "width": math.ceil(width) + 6,
                              "attrs": {"x": x, "y": y, "font-size": size, "fill": color,
                                        "font-weight": "700" if bold else "400"}})

    def para(self, text, x, y, width, size=25, leading=37, color=INK, bold=False, mono=False):
        words = text.split()
        lines, current = [], ""
        for word in words:
            trial = f"{current} {word}".strip()
            if font(size, bold, mono).getlength(trial) / 4 > width:
                if not current:
                    raise ValueError(word)
                lines.append(current)
                current = word
            else:
                current = trial
        if current:
            lines.append(current)
        for line in lines:
            self.text(line, x, y, size, color=color, bold=bold, mono=mono, max_width=width)
            y += leading
        return y

    def rect(self, x, y, w, h, fill, stroke=None):
        self.elements.append({"type": "rect", "attrs": {"x": x, "y": y, "width": w, "height": h,
                              "fill": fill, **({"stroke": stroke} if stroke else {})}})

    def line(self, x1, y1, x2, y2, color=PURPLE, weight=2):
        self.elements.append({"type": "line", "attrs": {"x1": x1, "y1": y1, "x2": x2, "y2": y2,
                              "stroke": color, "stroke-width": weight}})

    def arrow(self, x1, y1, x2, y2, color=PURPLE):
        self.line(x1, y1, x2, y2, color, 3)
        angle = math.atan2(y2 - y1, x2 - x1)
        for side in [-1, 1]:
            a = angle + side * 0.52
            self.line(x2, y2, x2 - 12 * math.cos(a), y2 - 12 * math.sin(a), color, 3)

    def label(self, text, x, y, color=PURPLE, size=20):
        self.text(text, x, y, size, color=color, bold=True)

    def takeaway(self, text):
        self.line(56, 579, 1224, 579, "#B9A6D3", 2)
        self.para(text, 56, 614, 1150, size=24, leading=29, bold=True)

    def chart(self, key, x=56, y=186, w=760, h=384):
        self.elements.append({"type": "image", "path": str(ROOT / "images" / f"{key}.png"),
                              "attrs": {"x": x, "y": y, "width": w, "height": h}})

    def table(self, headers, rows, x, y, widths, row_h=58, size=23):
        w = sum(widths)
        self.rect(x, y, w, row_h, "#E4D9F3")
        all_rows = [headers] + rows
        for i, row in enumerate(all_rows):
            yy = y + i * row_h
            if i:
                self.line(x, yy, x + w, yy, "#D6CBE8", 1)
            xx = x
            for value, width in zip(row, widths):
                self.text(str(value), xx + 14, yy + row_h / 2 + size * .34, size,
                          bold=i == 0, max_width=width - 26)
                xx += width
        self.line(x, y + len(all_rows) * row_h, x + w, y + len(all_rows) * row_h, "#B9A6D3", 1)


s = Slide("Master's Thesis Pre-Defense", "Introduction",
          "This working draft follows a problem-method-evaluation structure. Numerical placeholders on slides 12, 14 and 21 are hypothetical assumptions, not experimental findings. The title is the thesis title supplied by the researcher.")
for line, y in zip(["Performance Analysis and", "Adaptive Optimization of", "Local LLM Inference", "for Agentic Workloads"], [240, 306, 372, 438]):
    s.text(line, 64, y, 49, bold=True)
s.text("Wang", 68, 528, 27)
s.text("November 11, 2026", 68, 567, 23, color=MUTED)
s.text("Working draft for discussion", 68, 616, 21, color=MUTED)

s = Slide("What Is a Local Code Agent?", "Background",
          "A local code agent combines a locally running language model with tools such as a test runner. The workflow can avoid sending source code to a remote inference service, provided no remote tools are used. Local execution still needs an appropriate model, and it shares the machine's compute and memory with other applications.")
s.para("A language model writes code. An agent also runs tools and uses their results.", 56, 211, 1140, 30, 43, bold=True)
for x, label, a, b in [(56, "Code model", "Reads a request", "Generates code"),
                       (456, "Tools", "Run Python tests", "Return error messages"),
                       (856, "Local computer", "Runs model and tools", "Uses local resources")]:
    s.label(label, x, 335, size=25)
    s.para(a + ". " + b + ".", x, 380, 340, 25, 36)
s.label("Why local?", 56, 490)
s.para("Keep inference local and work without a remote model service. The trade-off is limited compute and shared memory.", 56, 529, 1145, 25, 35)
s.takeaway("This study focuses on the time users spend waiting for local code generation.")

s = Slide("LLM-Based Software Development Workflow", "Background",
          "The user supplies a task. The model generates code and a tool executes visible tests. If a test fails, the repair prompt includes the previous code and its error feedback. The model generates a revised response, and the tool runs tests again. In the evaluated pilot-style workflow, at most two repairs are allowed. Hidden scoring tests never enter the prompt.")
for x, label, detail in [(56, "Generate code", "Task and example"), (356, "Run tests", "Execute code"),
                          (656, "Read errors", "Test feedback"), (956, "Repair", "Old code + errors")]:
    s.rect(x, 254, 268, 112, "#E4D9F3")
    s.text(label, x + 17, 296, 26, bold=True, max_width=236)
    s.text(detail, x + 17, 338, 20, color=MUTED, max_width=236)
for x in [326, 626, 926]:
    s.arrow(x, 309, x + 26, 309)
s.line(1090, 369, 1090, 422, PURPLE, 3)
s.line(1090, 422, 490, 422, PURPLE, 3)
s.arrow(490, 422, 490, 369)
s.text("Fail: test the revised code again", 580, 461, 24, color=PURPLE)
s.arrow(490, 254, 490, 205, GREEN)
s.text("Pass: return the result", 520, 211, 24, color=GREEN)
s.para("Repair input: task + previous code + test feedback", 56, 530, 1135, 27, 38)
s.takeaway("The repair stage generates revised code, then returns to testing.")

s = Slide("Research Problem and Goal", "Introduction",
          "This work studies inference latency in local code-agent repair. A model may regenerate unchanged parts of an earlier solution, yet ordinary decoding still processes output sequentially. The goal is to reduce model and agent wall time while preserving the ordinary greedy output in the evaluated runs. This does not claim to repair bugs that the underlying model cannot repair.")
s.label("Problem", 56, 214, size=25)
s.para("Each repair can generate a full response again, including code that already exists in the history.", 56, 260, 1120, 30, 43)
s.para("Repeated model calls add to the user's waiting time on a local machine.", 56, 367, 1120, 28, 40)
s.label("Goal", 56, 464, size=25)
s.para("Make local code agents faster without changing the code they generate.", 56, 509, 1120, 31, 44, bold=True)
s.takeaway("Optimize inference time. Keep task success and output preservation as separate checks.")

s = Slide("Why Repair Offers Reusable Code", "Background",
          "This is a schematic example, not an experimental trace. A repair may alter one expression while retaining the surrounding code. The old version can provide candidate tokens for the unchanged region. The target model must still verify them. The reuse opportunity is not a guarantee that any particular repair succeeds.", refs=(2, 3, 4))
s.label("Previous code", 56, 209, size=24)
s.label("Revised code", 660, 209, size=24)
before = ["def is_even(n):", "    value = n % 2", "    return value == 1"]
after = ["def is_even(n):", "    value = n % 2", "    return value == 0"]
for x, lines in [(56, before), (660, after)]:
    s.rect(x, 236, 560, 200, "#FFFFFF", "#D6CBE8")
    for i, line in enumerate(lines):
        if i == 2:
            s.rect(x + 10, 364, 540, 43, "#FFF1D6")
        s.text(line, x + 18, 284 + i * 53, 24, mono=True, max_width=525)
s.text("Visible test: is_even(2) should return True.", 56, 486, 25)
s.text("Reuse the matching prefix. Generate and check the changed part.", 56, 539, 25)
s.takeaway("Earlier code is a cheap source of candidate tokens. [2-4]")

s = Slide("How an LLM Generates Code", "Background",
          "A token is a small text unit, not necessarily a whole word. In ordinary greedy decoding, the model selects the highest-scoring next token at each step. KV cache stores intermediate attention keys and values for earlier tokens, avoiding recomputation of the full prefix. It does not remove the next-token model steps. The diagram uses text fragments only to explain the sequential process.")
s.para("A token is a small piece of text. Ordinary decoding adds one token per step.", 56, 214, 1145, 30, 43)
s.text("Greedy decoding selects the highest-scoring next token.", 56, 273, 24, color=MUTED)
for x, token in [(56, "def"), (292, " is_even"), (528, "("), (764, "n"), (1000, "):")]:
    s.rect(x, 315, 200, 72, "#E4D9F3")
    s.text(token, x + 24, 361, 28, mono=True)
    if x < 1000:
        s.arrow(x + 204, 350, x + 230, 350)
s.label("KV cache", 56, 466, size=25)
s.para("Stores intermediate results for earlier tokens. This saves repeated work, but the model still runs for each new token.", 56, 509, 1130, 27, 39)
s.takeaway("Long responses still require many sequential model steps.")

s = Slide("Speculative Decoding", "Related techniques",
          "Speculative decoding uses cheap draft proposals and a target model that verifies several proposed tokens together. A draft model is usually a smaller and faster model, while the target is the model whose output is desired. This study uses greedy matching: retain the matching prefix and stop accepting at the first mismatch. The original paper also addresses distribution-preserving sampling. Saved target steps must outweigh proposal, verification and other overheads.", refs=(1,))
s.para("A fast draft model proposes tokens. The target is the model whose output we want. [1]", 56, 211, 1130, 29, 42)
for x, label, detail in [(56, "Draft model", "Propose several tokens"), (466, "Target model", "Check the draft in one batch"),
                          (876, "Output", "Keep the matching prefix")]:
    s.label(label, x, 356, size=25)
    s.para(detail, x, 404, 332, 26, 38)
s.arrow(356, 375, 446, 375)
s.arrow(766, 375, 856, 375)
s.para("Batch checking can advance several tokens. A mismatch ends the accepted prefix.", 56, 523, 1130, 26, 37)
s.takeaway("Saved target steps must outweigh drafting and verification costs.")

s = Slide("Existing Reuse and Adaptive Methods", "Related work",
          "Prompt Lookup Decoding retrieves candidate continuations from existing prompt text. EfficientEdit reuses source code and includes edit-oriented drafting and dynamic verification. AgSpec builds retrieval corpora for coding agents and adapts draft length using feedback. DISCO studies dynamic lookahead. The present comparison is against controlled in-project baselines, not reproduced implementations of all these systems. General history reuse and dynamic length are therefore not claimed as first-ever ideas.", refs=(2, 3, 4, 5))
s.table(["Prior work", "Idea used in the literature"], [
    ["Prompt Lookup [2]", "Copy continuations from existing text"],
    ["EfficientEdit [3]", "Reuse code with edit-oriented drafting"],
    ["AgSpec [4]", "Agent history and adaptive draft length"],
    ["DISCO [5]", "Choose speculation lookahead dynamically"],
], 56, 210, [350, 818], row_h=66, size=24)
s.takeaway("Our focus: a time-based decision rule for local history drafting, with explicit cost accounting.")

s = Slide("Proposed Method: Cost-Aware Selection", "Method",
          "The proposal is a lightweight policy on top of history retrieval and target verification. It scores feasible lengths using estimated time per output token, including verification, lookup and the current implementation's expected recovery. k=0 means ordinary decoding. Its contribution must be argued through this concrete design and controlled evidence, not simply the existence of adaptive length. Cost validation and standard API integration are still pending in this draft.", refs=(2, 4, 5))
s.para("At each step, decide whether to reuse history and how much to reuse.", 56, 213, 1130, 31, 44, bold=True)
for x, title, body in [(56, "Observe", "Retrieve a candidate and track recent acceptance."),
                        (456, "Compare", "Estimate time per output token for each action."),
                        (856, "Choose", "Use ordinary decoding or k = 2, 4, 8, 16.")]:
    s.label(title, x, 359, size=26)
    s.para(body, x, 407, 340, 26, 39)
s.takeaway("The proposed component is the action-selection policy, not history lookup itself.")

s = Slide("Historical Draft Retrieval", "Existing component",
          "The history lookup follows an existing n-gram continuation idea. The local implementation matches a recent suffix, retrieves following tokens and caps candidates by available history. It does not access held-out tests. Repair requests use earlier generated code as history; cold first generations take the ordinary path. All controlled history methods share the same retrieval source.", refs=(2,))
s.label("Existing n-gram lookup idea [2]", 56, 211, size=25)
for y, number, title, detail in [
    (288, "1", "Match the current suffix", "Search for the recent output pattern in earlier code."),
    (386, "2", "Read its continuation", "Use the following historical tokens as candidates."),
    (484, "3", "Pass candidates to the policy", "The policy decides how many tokens to verify."),
]:
    s.text(number, 60, y, 34, color=PURPLE, bold=True)
    s.text(title, 114, y, 27, bold=True)
    s.text(detail, 114, y + 38, 24, color=MUTED)
s.takeaway("Lookup proposes text. Target verification determines what can enter the output.")

s = Slide("Estimating Time per Output Token", "Method",
          "First ask how much time a step costs and how far it advances the output. C(k) equals estimated verification plus lookup plus expected recovery divided by expected output advance. T_step is the ordinary single-step time. This is a proxy score, not a validated latency predictor. Policy overhead, cache handling and end-of-sequence effects need accounting checks. Offline step and verification calibration are fixed in the current policy; acceptance and lookup estimates update online.")
s.para("A longer draft is useful only if the extra progress is worth the extra time.", 56, 211, 1120, 30, 42)
s.text("C(k) = [ V(k) + L + R(k) ] / G(k)", 68, 341, 38, mono=True, color=PURPLE, bold=True)
s.table(["Term", "Meaning"], [["V(k)", "Time to verify k candidate tokens"],
                              ["L", "History lookup time"],
                              ["R(k)", "Expected recovery time in this implementation"],
                              ["G(k)", "Expected number of output tokens advanced"]],
        56, 366, [185, 983], row_h=40, size=22)
s.takeaway("Compare the score with ordinary decoding. Select reuse only when it is cheaper.")

s = Slide("Verification Cost by Draft Length", "Cost validation", placeholder=True,
          notes="ALL NUMBERS ON THIS SLIDE ARE ILLUSTRATIVE PLACEHOLDERS. They are not measured or fitted from old experiments. Assume ordinary step=1 and verification ratios 1.10, 1.22, 1.48 and 2.05 solely to lay out the planned result table. V(k) includes a batch of current token plus k candidates. Replace these with synchronized repeated timings across context lengths before drawing any Apple Silicon cost conclusion. Do not assume server GPUs have perfectly constant verification cost either.")
s.text("Illustrative normalized time: one ordinary step = 1.00", 56, 218, 26)
s.table(["Draft length k", "Verified input tokens", "V(k), assumed"],
        [[str(k), str(k + 1), f"{v:.2f}"] for k, v in zip([2, 4, 8, 16], [1.10, 1.22, 1.48, 2.05])],
        56, 251, [330, 470, 368], row_h=54, size=24)
s.para("Planned measurement: warm up, synchronize, repeat each length, and vary context length.", 56, 561, 1140, 24, 34)
s.text("These values do not establish how verification cost scales on Apple Silicon.", 56, 626, 23, color="#9B4300", bold=True)

s = Slide("Cache Recovery: What We Count", "Method and implementation check",
          "The current local code snapshots the cache before batched verification. On rejection it restores that snapshot and replays the retained prefix through ordinary single-token forwards. Earlier diagnostics reported shape-related floating-point differences in retained KV values. This explains the implementation choice but is not a universal requirement of speculative decoding. It remains necessary to test whether native cache trimming, a different verification path or another precision avoids replay while preserving complete output. Do not treat replay as a hardware law.")
s.label("Current implementation", 56, 212, size=25)
s.para("On rejection: restore the earlier cache, then recompute the retained prefix through ordinary model steps.", 56, 257, 1135, 28, 40)
s.label("Why it was added", 56, 378, size=25)
s.para("Earlier diagnostics found cache-state differences after batched verification. Replay aimed to recover the ordinary path.", 56, 423, 1135, 27, 39)
s.takeaway("Replay may be implementation-specific. Check if a correct native path can avoid it.")

s = Slide("Dynamic Policy: A Worked Example", "Method example", placeholder=True,
          notes="HYPOTHETICAL DECISION EXAMPLE, NOT AN EXPERIMENT. Use the illustrative V(k) from slide 12, L=0.02 and T_step=1. R(k)=T_step*sum_{j=0}^{k-1}(j+1)*p^j*(1-p), G(k)=1+sum_{j=1}^k p^j. At p=.55 all scores exceed the ordinary path. At p=.9 k4 has the lowest score. At p=.98 k8 has the lowest score. The 5% estimated margin means reuse must score below .95. The calculation assumes geometric acceptance and no truncation. This only explains how the policy operates.")
s.text("Assume ordinary step = 1.00 and lookup = 0.02", 56, 217, 26)
rows = []
for p in [.55, .90, .98]:
    costs = []
    for k, verify in zip([2, 4, 8, 16], [1.10, 1.22, 1.48, 2.05]):
        gain = 1 + sum(p**j for j in range(1, k + 1))
        replay = sum((j + 1) * p**j * (1 - p) for j in range(k))
        costs.append((verify + .02 + replay) / gain)
    best = [2, 4, 8, 16][min(range(4), key=lambda i: costs[i])]
    rows.append([f"{p:.2f}"] + [f"{c:.2f}" for c in costs] + ["Ordinary" if min(costs) >= .95 else f"k = {best}"])
s.table(["Acceptance p", "k=2", "k=4", "k=8", "k=16", "Choice"], rows,
        56, 254, [260, 164, 164, 164, 164, 252], row_h=61, size=23)
s.para("Lower score is better. Use a draft only if its estimated cost is at least 5% below ordinary decoding.",
       56, 552, 1140, 24, 35)
s.text("Assumed inputs explain the rule. They do not validate its predictions.", 56, 628, 23, color="#9B4300", bold=True)

s = Slide("Experimental Setup", "Evaluation",
          "Reported main evaluation uses a custom SiliconLLM synchronous decoding loop based on tiny-llm, with Qwen2-1.5B-Instruct-MLX 4-bit group64 on Apple Silicon with 18 GiB unified memory. Main replay: 30 tasks, 64 fixed requests, three shuffled repetitions. Agent episodes: 20 tasks, three repetitions, up to two repairs, 384 tokens per call. The native extension uses mlx-lm model and cache in a shared synchronous loop. Full stock API integration and timings remain planned. Raw records for the reported external run are not locally available and have not been independently verified here.", refs=(6, 7, 8, 9))
s.table(["Item", "Configuration"], [
    ["Hardware / target", "Apple Silicon, 18 GiB / Qwen2-1.5B, 4-bit"],
    ["Main inference path", "SiliconLLM custom synchronous loop [9]"],
    ["Native extension", "mlx-lm model + cache, shared loop [6]"],
    ["Tasks", "MBPP sanitized [7] and HumanEval [8]"],
    ["Main replay / agent", "30 tasks, 64 requests / 20 tasks; 3 repeats"],
    ["Decoding / repairs", "Greedy; 384 tokens per call; up to 2 repairs"],
], 56, 207, [340, 828], row_h=52, size=23)
s.takeaway("The full mlx-lm generation API is a separate planned comparison, not the main engine.")

s = Slide("Baselines and Fair Comparison", "Evaluation",
          "Ordinary greedy decoding with persistent KV cache is the reference. Fixed-history decoding uses at most 16 candidates. Adaptive decoding selects k0,2,4,8,16. These controlled comparisons share prompt, model, history source, verifier and recovery path. Model wall time and agent wall time are separate outcomes. Compare full token sequences and final task outcomes, not just token-level local agreement. The reported main runs matched, but this is not an unconditional numerical guarantee.")
s.table(["Method", "Next decoding action"], [
    ["Ordinary KV", "One ordinary greedy step"],
    ["Fixed history", "Use available history, up to k = 16"],
    ["Adaptive history", "Choose k = 0, 2, 4, 8 or 16"],
], 56, 213, [360, 808], row_h=67, size=25)
s.para("Keep model, prompts, history and verification path the same.", 56, 530, 1130, 26, 37)
s.takeaway("Measure wall time and check complete outputs. Fewer target calls alone do not prove speedup.")

s = Slide("Model-Time Results", "Reported results", reported=True,
          notes="Reported point estimates from the adopted study report, not a new local reproduction. Main 30-task queue, 64 requests, three shuffled repetitions. Ordinary1.00, fixed1.45, dynamic1.65. Dynamic/fixed=1.1379, about14% higher speed, not14% lower time. Do not infer confidence intervals from these aggregate bars. Full output matches were reported for the tested runs. Underlying external per-run records still need archiving locally.")
s.chart("overall")
s.label("Main replay", 854, 235, size=24)
s.para("64 fixed requests, three repeats. Complete outputs matched in reported runs.", 854, 281, 350, 24, 34)
s.text("1.65 / 1.45", 854, 420, 34, color=GREEN, bold=True)
s.para("About 14% higher speed than fixed history in this queue.", 854, 465, 350, 24, 36)
s.takeaway("Reported overall speedup: fixed history 1.45x; adaptive history 1.65x.")

s = Slide("Where the Speedup Comes From", "Reported repair results", reported=True,
          notes="Repair-only speedups adopted from the user's explicit correction: fixed2.05 and dynamic approximately2.86, replacing the draft report's2.20. First generation uses the same ordinary algorithm, so overall gain is lower than repair-only gain. Public task savings were dominated by unsuccessful repair attempts repeating earlier code. These results are latency findings, not evidence of improved repair success.")
s.chart("repair")
s.label("Repair calls only", 854, 235, size=24)
s.para("Earlier code is available as draft history.", 854, 282, 350, 25, 37)
s.para("Cold first generation stays on the ordinary path.", 854, 395, 350, 25, 37)
s.takeaway("Large savings mainly came from repeated code in failed repairs, not more solved tasks.")

s = Slide("Agent Workflow Results", "Reported end-to-end results", reported=True,
          notes="20 actual tasks, three repeats per method. The displayed seconds are sums of per-task mean agent times, not one task's average or the sum of all raw repetitions. Ordinary201.51, fixed139.94, dynamic124.39 seconds. Speedups1.440 and1.620. Timing includes prompt processing, visible tests and logging; model loading, warmup and final hidden scoring are excluded. Outputs and task outcomes matched in the reported episodes.")
s.chart("agent")
s.label("Agent wall time", 854, 235, size=24)
s.para("20 tasks, three repeats. Includes visible tests and logging.", 854, 282, 350, 25, 37)
s.text("1.62x", 854, 439, 43, color=GREEN, bold=True)
s.para("Adaptive vs ordinary", 854, 484, 350, 24, 35)
s.takeaway("The reported decoding gain also reduces time in this small agent workflow.")

s = Slide("Native MLX Model and Cache", "Reported implementation extension", reported=True,
          notes="The native extension uses the native mlx-lm model and non-rotating cache through the project's shared synchronous loop. Its own ordinary path is the normalization reference. Reported fixed1.34, dynamic1.60, acceptance1.14. This is not the stock streaming API and does not establish superiority to the fastest mlx-lm configuration. These are external reported values, not newly collected here.", refs=(6,))
s.chart("native")
s.label("Native components [6]", 854, 235, size=23)
s.para("mlx-lm model and cache inside a shared synchronous loop.", 854, 282, 350, 25, 37)
s.para("Baseline: the native ordinary path in this same loop.", 854, 427, 350, 24, 36)
s.takeaway("This tests a second model/cache path. It is not a full stock mlx-lm API benchmark.")

s = Slide("Full mlx-lm API Comparison", "Planned standard-engine evaluation", placeholder=True,
          notes="ALL PERFORMANCE NUMBERS HERE ARE ILLUSTRATIVE PLACEHOLDERS, not measurements or calibrated forecasts. Assume normalized model times100,85,75 merely to complete the intended result layout. They are not copied from the native-cache result, and no speedup over stock mlx-lm is established. Baseline will use an explicitly versioned stock greedy generation API; method integration and fair timing boundaries still need implementation review. Match model, quantization, prompts, output length and stopping; measure prefill and decode separately, synchronize, warm up and randomize repeated runs. Complete-output comparisons must be recorded separately from timing.", refs=(6,))
s.text("Illustrative normalized model time: stock greedy = 100", 56, 217, 26)
s.table(["Planned configuration", "Time, assumed", "Output check"], [
    ["Stock mlx-lm greedy [6]", "100", "Pending"],
    ["mlx-lm + fixed history", "85", "Pending"],
    ["mlx-lm + adaptive history", "75", "Pending"],
], 56, 254, [625, 278, 265], row_h=65, size=24)
s.para("Planned controls: same checkpoint, prompts, stopping rules and warmup. Report prefill and decode time separately.",
       56, 552, 1145, 24, 34)
s.text("Integration and measurements pending. These values establish no engine advantage.", 56, 628, 22, color="#9B4300", bold=True)

s = Slide("Limitations and Remaining Checks", "Discussion",
          "Limitations of the adopted results include one hardware/model setting, repeated failed repairs, incomplete access to external raw logs and lack of a full stock API comparison. Placeholder verification timings are not evidence. Recovery necessity and cost-model prediction quality remain open. A follow-up should test useful successful edits, repeated sessions and sensitivity to acceptance and context. Report adverse cohorts as well as positive ones.")
items = [
    ("Workload", "Most public-task savings came from unsuccessful repairs that repeated code."),
    ("Cost model", "Measure V(k) and check predicted choices against actual latency."),
    ("Recovery", "Test whether a correct native implementation can avoid replay."),
    ("Engine", "Complete the stock mlx-lm API comparison with matching settings."),
    ("Generality", "Repeat across sessions, models and longer successful edits."),
]
for i, (label, body) in enumerate(items):
    y = 219 + i * 76
    s.label(label, 56, y, size=24)
    s.para(body, 270, y, 940, 24, 32)
s.takeaway("Results are limited to the tested model, machine and task queues.")

s = Slide("Conclusions", "Conclusion",
          "The thesis examines local inference for agentic repair workloads. It implements a history-based time-scoring policy and controlled comparisons. Reported results show1.45x fixed and1.65x adaptive model speedup on the main queue, and1.62x adaptive agent speedup. These values have limited scope and raw external records remain pending local archiving. The contribution claim must be finalized after cost validation and stock-engine evaluation; hypothetical placeholders do not support it.")
s.label("Problem", 56, 222, size=25)
s.para("Local repair repeatedly generates code that may already be available.", 270, 222, 935, 28, 40)
s.label("Method", 56, 344, size=25)
s.para("Select ordinary decoding or a historical draft length using estimated time per token.", 270, 344, 935, 28, 40)
s.label("Evidence", 56, 470, size=25)
s.para("Reported gains: 1.65x model speed and 1.62x agent speed in the tested queues.", 270, 470, 935, 28, 40)
s.takeaway("Final checks: validate the cost model and complete standard-engine evaluation.")

s = Slide("Backup: Cost Model Assumptions", "Backup",
          "The current formula uses a single estimated acceptance probability p and a geometric-prefix approximation. G=1+sum(p^j,j1..k). R/Tstep=sum((j+1)*p^j*(1-p),j0..k-1), matching the assumed rejected-prefix replay but excluding some cache and policy overhead. It is a proxy, not an exact mathematical expectation of implementation latency. Current code acceptance update is .7 old+.3 accepted/proposed; lookup .8 old+.2 observed. k0 only used when no lower score passes5%margin. The feedback variant is distinct. Full acceptance cache-state differences and end-of-sequence truncation require explicit correctness checks.")
s.text("G(k) = 1 + sum[j=1..k] p^j", 56, 235, 30, mono=True, color=PURPLE)
s.text("R(k) = T_step * sum[j=0..k-1] (j+1) p^j (1-p)", 56, 308, 27, mono=True, color=PURPLE)
s.para("Assumption: one acceptance estimate p approximates the matching-prefix distribution.", 56, 404, 1130, 26, 38)
s.para("The recovery term models rejected-prefix replay in the current implementation.", 56, 501, 1130, 26, 38)
s.takeaway("Check independence, truncation, cache overhead and prediction error before treating C(k) as calibrated.")

s = Slide("Backup: Fixed-Length Sensitivity", "Reported follow-up", reported=True,
          notes="Reported follow-up fixed arms only: k2=1.216,k4=1.292,k8=1.368,k16=1.450. The follow-up reuses40 requests from the third public cohort in a separate timing block; it is not a new independent task set. Method identity and paired controller intervals in the external summary have not been fully reconciled, so no paired dynamic superiority claim is drawn on this slide. k16 was selected on validation before evaluation timing, per the report.")
s.chart("fixed_lengths")
s.label("Fixed arms only", 854, 235, size=24)
s.para("Longer fixed drafts helped within this tested range.", 854, 282, 350, 25, 37)
s.para("Separate follow-up block. Reuses an existing task cohort.", 854, 425, 350, 24, 36)
s.takeaway("Do not merge this timing block with the main comparison or infer a universal best k.")

s = Slide("Backup: Fallback Feedback", "Reported separate cohort", reported=True,
          notes="Separate20-task public cohort, not the main30. Fixed1.45, originalcontroller1.20, feedbackcontroller1.60. Feedback uses the already-computed ordinary next token to compare the first retrieved candidate, requiring no extra forward. It estimates one-token match, not an unbiased multi-token acceptance rate. The original controller does not outperform fixed history in this cohort. The feedback variant is not silently substituted for the primary method.")
s.chart("feedback")
s.label("Separate 20 tasks", 854, 235, size=24)
s.para("Original control can stay ordinary after acceptance falls.", 854, 282, 350, 25, 37)
s.para("Feedback uses the existing ordinary next token to update the estimate.", 854, 418, 350, 24, 36)
s.takeaway("The original controller is not always better than fixed history. Keep the feedback variant separate.")

for title, ids in [("References (1/2)", [1, 2, 3, 4, 5]), ("References (2/2)", [6, 7, 8, 9])]:
    s = Slide(title, "References", "Primary literature and software references. Software revisions and access context must be fixed in the final experimental manifest.", refs=tuple(ids))
    y = 211
    for number in ids:
        entry, url = REFS[number]
        y = s.para(f"[{number}] {entry}", 56, y, 1155, 22, 28)
        s.text(url, 94, y + 1, 19, color=PURPLE)
        y += 48

assert len(PAGES) == 28
payload = []
for s in PAGES:
    s.text(f"{s.number:02d} / {len(PAGES)}", 1137, 688, 17, color="#6F6280", max_width=90)
    refs = "\n".join(f"[{r}] {REFS[r][0]}\n{REFS[r][1]}" for r in s.refs)
    internal = ""
    if s.reported:
        internal = ("\nInternal provenance: adopted report in docs/DECODING_RESULTS_Claude.md and subsequent user corrections. "
                    "Not independently reproduced locally. Not withdrawn local experimental data.")
    if s.placeholder:
        internal += "\nCRITICAL: illustrative numerical placeholders. Do not cite as measurements. Replace before a final evidence-based defense."
    notes = s.notes + "\n\nReferences\n" + refs + internal
    payload.append({"number": s.number, "title": s.title, "placeholder": s.placeholder,
                    "reported": s.reported, "elements": s.elements, "notes": notes})
    svg = ['<svg xmlns="http://www.w3.org/2000/svg" width="1280" height="720" viewBox="0 0 1280 720">']
    for item in s.elements:
        attrs = dict(item["attrs"])
        if item["type"] == "text":
            attrs["font-family"] = item["font"]
            values = " ".join(f'{k}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
            svg.append(f'<text {values}>{escape(item["text"])}</text>')
        elif item["type"] == "image":
            import base64
            attrs["href"] = "data:image/png;base64," + base64.b64encode(Path(item["path"]).read_bytes()).decode()
            values = " ".join(f'{k}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
            svg.append(f'<image {values}/>')
        else:
            values = " ".join(f'{k}="{escape(str(v), quote=True)}"' for k, v in attrs.items())
            svg.append(f'<{item["type"]} {values}/>')
    svg.append("</svg>")
    (ROOT / "svg_output" / f"{s.number:02d}.svg").write_text("\n".join(svg))

(ROOT / ".build/slide_payload.json").write_text(json.dumps(payload, indent=2))
(ROOT / "notes/slide_notes.md").write_text("\n\n".join(
    f"# {p['number']:02d} {p['title']}\n\n{p['notes']}" for p in payload))
(ROOT / "sources/placeholder_register.json").write_text(json.dumps([
    {"slide": 12, "topic": "Verification cost by k", "status": "Hypothetical normalized values",
     "replace_with": "Synchronized native-path measurements by k and context, with spread and configuration"},
    {"slide": 14, "topic": "Decision example", "status": "Arithmetic on hypothetical slide12 inputs",
     "replace_with": "Real calibration or retain as a clearly labeled pedagogical example"},
    {"slide": 21, "topic": "Full mlx-lm API comparison", "status": "Hypothetical normalized times, integration pending",
     "replace_with": "Versioned standard-API baseline, method integration, full output checks, repeated timings"},
], indent=2))
print(f"Prepared {len(PAGES)} slides. Numerical placeholders only on 12, 14 and 21.")
