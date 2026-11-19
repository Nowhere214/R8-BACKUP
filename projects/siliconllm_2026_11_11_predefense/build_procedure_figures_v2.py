"""Minimal technical diagrams, exported separately from the slide-style version."""

import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle

SKILL = Path("/Users/laiqixiao/.codex/skills/scientific-visualization/scripts")
sys.path.insert(0, str(SKILL))
from figure_export import save_publication_figure
from style_presets import apply_publication_style

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exports/procedure_figures_2026_11_16_v2"
OUT.mkdir(parents=True, exist_ok=True)
apply_publication_style("default")
plt.rcParams.update({"font.family": "Arial", "svg.fonttype": "none",
                     "pdf.fonttype": 42, "mathtext.fontset": "dejavusans"})
INK, GRAY, LIGHT = "#252525", "#6B6B6B", "#ECECEC"
PURPLE, TINT = "#5C408A", "#EEE9F5"
CHECKS = []


class Figure:
    def __init__(self, name, title):
        self.name, self.labels = name, []
        self.fig = plt.figure(figsize=(12, 5.6), dpi=100, facecolor="none")
        self.fig.set_layout_engine("none")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 1200), ylim=(560, 0))
        self.ax.axis("off")
        self.text(44, 42, title, 26, bold=True)

    def text(self, x, y, s, size=23, color=INK, bold=False, ha="left", mono=False):
        t = self.ax.text(x, y, s, ha=ha, va="center", fontsize=size * .72,
                         color=color, weight="bold" if bold else "normal",
                         fontfamily="DejaVu Sans Mono" if mono else "Arial",
                         linespacing=1.35)
        self.labels.append(t)

    def line(self, points, color=GRAY, dashed=False, lw=1):
        self.ax.plot(*zip(*points), color=color, linewidth=lw,
                     linestyle=(0, (4, 3)) if dashed else "-")

    def arrow(self, points, color=INK, dashed=False):
        if len(points) > 2:
            self.line(points[:-1], color, dashed)
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="->",
                                         mutation_scale=12, color=color,
                                         linewidth=1, linestyle=(0, (4, 3)) if dashed else "-",
                                         shrinkA=2, shrinkB=3))

    def rect(self, x, y, w, h, fill="none", edge=GRAY, hatch=None):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                                   linewidth=.85, hatch=hatch))

    def finish(self, pdf):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        boxes = [(t.get_text(), t.get_window_extent(renderer)) for t in self.labels]
        for text, b in boxes:
            if b.x0 < 10 or b.y0 < 10 or b.x1 > 1190 or b.y1 > 550:
                raise ValueError(f"Outside canvas: {text}")
        for i, (text, a) in enumerate(boxes):
            for other, b in boxes[i + 1:]:
                if a.overlaps(b):
                    raise ValueError(f"Overlapping text: {text} / {other}")
        saved = save_publication_figure(
            self.fig, OUT / self.name, formats=["png", "svg"], dpi=300,
            transparent=True, bbox_inches=None, pad_inches=0)
        saved += save_publication_figure(
            self.fig, OUT / self.name, formats=["pdf"], dpi=300,
            facecolor="white", bbox_inches=None, pad_inches=0)
        if len(saved) != 3:
            raise RuntimeError("Missing export")
        pdf.savefig(self.fig, facecolor="white", bbox_inches=None)
        CHECKS.append({"figure": self.name, "text_bounds": "passed",
                       "text_overlap": "none", "labels": len(self.labels)})
        plt.close(self.fig)


def ordinary(pdf):
    f = Figure("01_mlx_lm_procedure", "mlx-lm generation")
    for x, title in [(110, "Input tokens"), (400, "Prefill"), (730, "Decode"), (1050, "Output")]:
        f.text(x, 182, title, 27, ha="center")
    f.arrow([(195, 182), (333, 182)])
    f.arrow([(469, 182), (656, 182)])
    f.arrow([(805, 182), (980, 182)])
    f.text(890, 212, "EOS / limit", 18, GRAY, ha="center")
    f.arrow([(772, 158), (772, 112), (688, 112), (688, 158)])
    f.text(730, 90, "next token", 18, GRAY, ha="center")
    f.text(400, 225, "prompt prefix", 19, GRAY, ha="center")
    f.text(730, 225, "forward + argmax", 19, GRAY, ha="center")
    f.text(268, 350, "K", 22, ha="right", mono=True)
    f.text(268, 391, "V", 22, ha="right", mono=True)
    for y in (332, 373):
        f.rect(290, y, 340, 35, LIGHT)
        f.rect(630, y, 200, 35)
        for x in range(670, 830, 40):
            f.line([(x, y), (x, y + 35)], "#ADADAD", lw=.6)
    f.arrow([(400, 250), (400, 326)], GRAY)
    f.text(420, 289, "initialize", 18, GRAY)
    f.arrow([(744, 325), (744, 250)], GRAY)
    f.arrow([(792, 250), (792, 325)], GRAY)
    f.text(700, 291, "read", 18, GRAY, ha="right")
    f.text(812, 291, "append", 18, GRAY)
    f.text(460, 447, "prompt cache", 20, GRAY, ha="center")
    f.text(730, 447, "generated tokens", 20, GRAY, ha="center")
    f.finish(pdf)


def integration(pdf):
    f = Figure("02_history_integration_plan", "History decoding in mlx-lm")
    f.text(1155, 42, "planned integration", 18, GRAY, ha="right")
    f.text(180, 174, "Current prefix", 24, ha="center")
    f.text(490, 174, "History lookup", 24, ha="center")
    f.text(490, 217, "previous code", 18, GRAY, ha="center")
    f.rect(758, 143, 243, 62, TINT, PURPLE)
    f.text(880, 174, "Choose k", 24, PURPLE, ha="center")
    f.text(880, 223, "0, 2, 4, 8, 16", 18, GRAY, ha="center")
    f.arrow([(270, 174), (397, 174)])
    f.arrow([(585, 174), (752, 174)])
    f.arrow([(810, 205), (810, 283), (658, 283), (658, 322)])
    f.arrow([(950, 205), (950, 322)])
    f.text(700, 262, "k = 0", 19, GRAY)
    f.text(972, 279, "k > 0", 19, GRAY)
    f.text(658, 344, "Ordinary step", 24, ha="center")
    f.text(950, 344, "Verify draft", 24, ha="center")
    f.arrow([(658, 368), (658, 440), (748, 440)])
    f.arrow([(950, 368), (950, 440), (868, 440)])
    f.text(808, 440, "KV update", 23, ha="center")
    f.arrow([(808, 461), (808, 501), (180, 501), (180, 202)])
    f.text(180, 540, "continue until EOS or token limit", 18, GRAY)
    f.finish(pdf)


def recovery(pdf):
    f = Figure("03_mlx_lm_cache_recovery", "KV cache after a rejected draft")
    f.text(44, 89, "mlx-lm target cache", 18, GRAY)
    f.text(1155, 89, "draft: A B C D     accepted: A B     correction: Z", 19, ha="right", mono=True)
    start, cell, y0 = 404, 71, 175
    labels = ["Before verification", "After verification", "After trim", "Next forward"]
    for index, label in enumerate(labels):
        y = y0 + index * 78
        f.text(44, y + 18, label, 22)
        f.rect(285, y, 119, 36, LIGHT)
        f.text(344, y + 18, "prefix", 19, ha="center")
    for row, tokens in [(1, ["x", "A", "B", "C", "D"]),
                        (2, ["x", "A", "B"]), (3, ["x", "A", "B", "Z"])]:
        y = y0 + row * 78
        for index, token in enumerate(tokens):
            rejected = row == 1 and index >= 3
            f.rect(start + index * cell, y, cell, 36, LIGHT if rejected else "none",
                   hatch="///" if rejected else None)
            f.text(start + (index + .5) * cell, y + 18, token, 22, ha="center", mono=True)
    f.text(438, y0 + 18, "x", 22, ha="center", mono=True)
    f.text(500, y0 + 18, "pending input", 18, GRAY)
    f.text(820, y0 + 96, "trim 2 entries", 20, GRAY)
    f.text(820, y0 + 174, "Z is not cached yet", 20, GRAY)
    f.text(820, y0 + 252, "process Z", 20, GRAY)
    f.line([(617, 301), (617, 309), (759, 309), (759, 301)])
    f.text(688, 331, "discard", 17, GRAY, ha="center")
    f.text(44, 513, "K and V have the same valid token range. Prefix entries remain in place.", 19, GRAY)
    f.finish(pdf)


def cost(pdf):
    f = Figure("04_cost_score_and_recovery", "Cost score")
    f.text(44, 135, r"$C(k)=\frac{V(k)+L+R(k)}{G(k)}$", 43)
    f.text(620, 127, "time per round / output tokens", 23)
    f.text(620, 174, "Compare with one ordinary decode step.", 19, GRAY)
    for y, symbol, meaning in [(253, "V(k)", "verification, including KV-cache use"),
                               (299, "L", "history lookup"),
                               (345, "G(k)", "expected accepted tokens + one target token"),
                               (391, "R(k)", "extra replay cost, when replay is used")]:
        f.text(44, y, symbol, 23, mono=True)
        f.text(165, y, meaning, 23)
    f.line([(44, 440), (1155, 440)], "#B0B0B0")
    f.text(44, 478, "Current local path", 19, GRAY)
    f.text(300, 478, "restore + replay", 21)
    f.text(655, 478, "mlx-lm path", 19, GRAY)
    f.text(870, 478, "trim + continue", 21)
    f.text(44, 532, "R(k) depends on the recovery path. The score is an estimate, not an exact timing formula.", 18, GRAY)
    f.finish(pdf)


with PdfPages(OUT / "mlx_lm_procedure_figures_v2.pdf") as pdf:
    ordinary(pdf)
    integration(pdf)
    recovery(pdf)
    cost(pdf)

(OUT / "validation.json").write_text(json.dumps(CHECKS, indent=2))
print(f"Exported {len(CHECKS)} minimal diagrams to {OUT}")
