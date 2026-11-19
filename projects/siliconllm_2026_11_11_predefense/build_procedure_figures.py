"""Draw explanatory figures without changing the reviewed deck or results."""

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import FancyArrowPatch, Rectangle

SKILL = Path("/Users/laiqixiao/.codex/skills/scientific-visualization")
sys.path.insert(0, str(SKILL / "scripts"))
from figure_export import save_publication_figure
from style_presets import apply_publication_style

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "exports/procedure_figures_2026_11_16"
OUT.mkdir(parents=True, exist_ok=True)
apply_publication_style("default")
plt.rcParams.update({"font.family": "Arial", "font.size": 14,
                     "svg.fonttype": "none", "pdf.fonttype": 42})
BG, PURPLE, INK, MUTED = "#F4EFFA", "#3E1485", "#1F1633", "#655875"
BLUE, GREEN, ORANGE = "#0072B2", "#009E73", "#D55E00"
PALE_BLUE, PALE_GREEN, PALE_ORANGE = "#E7F2FA", "#E4F4EE", "#FFF0E3"
STROKE = "#CABDDA"
CHECKS = []


class Drawing:
    def __init__(self, title, subtitle, key, source):
        self.key = key
        self.fig = plt.figure(figsize=(12.8, 7.2), dpi=100, facecolor=BG)
        self.fig.set_layout_engine("none")
        self.ax = self.fig.add_axes([0, 0, 1, 1])
        self.ax.set(xlim=(0, 1280), ylim=(720, 0))
        self.ax.axis("off")
        self.text_bounds = []
        self.rect(0, 0, 1280, 118, PURPLE, PURPLE)
        self.text(52, 58, title, 25, "white", True)
        self.text(54, 151, subtitle, 12.5, MUTED)
        self.ax.plot([54, 1226], [658, 658], color=STROKE, lw=0.8)
        self.text(54, 681, source, 9.5, MUTED)

    def text(self, x, y, content, size=16, color=INK, bold=False,
             ha="left", va="center", max_width=None):
        artist = self.ax.text(x, y, content, fontsize=size, color=color,
                              weight="bold" if bold else "normal", ha=ha,
                              va=va, linespacing=1.35)
        self.text_bounds.append((artist, max_width))
        return artist

    def rect(self, x, y, w, h, fill="white", edge=STROKE, lw=1.3, dashed=False):
        self.ax.add_patch(Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge,
                                   linewidth=lw, linestyle="--" if dashed else "-"))

    def box(self, x, y, w, h, title, detail="", fill="white", edge=STROKE):
        self.rect(x, y, w, h, fill, edge)
        self.text(x + w / 2, y + (h / 2 if not detail else 29), title,
                  16, INK, True, ha="center", max_width=w - 14)
        if detail:
            self.text(x + w / 2, y + 66, detail, 12, MUTED,
                      ha="center", max_width=w - 14)

    def arrow(self, points, color=MUTED, dashed=False):
        for p, q in zip(points[:-2], points[1:-1]):
            self.ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=1.5,
                         linestyle="--" if dashed else "-")
        self.ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle="-|>",
                                         mutation_scale=15, linewidth=1.5,
                                         color=color, linestyle="--" if dashed else "-",
                                         shrinkA=2, shrinkB=3))

    def finish(self, pdf):
        self.fig.canvas.draw()
        renderer = self.fig.canvas.get_renderer()
        for artist, width in self.text_bounds:
            bbox = artist.get_window_extent(renderer)
            if bbox.x0 < 0 or bbox.y0 < 0 or bbox.x1 > 1280 or bbox.y1 > 720:
                raise ValueError(f"Text outside canvas: {artist.get_text()}")
            if width and bbox.width > width:
                raise ValueError(f"Text too wide ({bbox.width:.1f}>{width}): {artist.get_text()}")
        paths = save_publication_figure(
            self.fig, OUT / self.key, formats=["png", "pdf", "svg"], dpi=300,
            bbox_inches=None, pad_inches=0, facecolor=BG)
        if len(paths) != 3 or not all(p.exists() for p in paths):
            raise RuntimeError("Incomplete figure export")
        pdf.savefig(self.fig, bbox_inches=None, facecolor=BG)
        CHECKS.append({"figure": self.key, "text_bounds": "passed",
                       "text_items": len(self.text_bounds), "files": [str(p) for p in paths]})
        plt.close(self.fig)


def ordinary(pdf):
    d = Drawing("How mlx-lm Generates a Response", "ORDINARY GREEDY DECODING  /  Schematic overview", "01_mlx_lm_procedure",
                "Source: mlx-lm generate.py and models/cache.py. Simplified flow, not a timing result.")
    d.box(54, 210, 230, 106, "Prompt", "Task, old code, feedback")
    d.box(344, 210, 230, 106, "Tokenize", "Text becomes token IDs")
    d.box(634, 210, 260, 106, "Prefill", "Process the prompt prefix", PALE_BLUE, BLUE)
    d.box(954, 210, 272, 106, "Decode step", "Model forward + greedy choice", PALE_BLUE, BLUE)
    for x1, x2 in [(284, 344), (574, 634), (894, 954)]:
        d.arrow([(x1, 263), (x2, 263)])
    d.box(634, 410, 260, 102, "KV cache", "Store past keys and values", PALE_GREEN, GREEN)
    d.arrow([(764, 316), (764, 410)], GREEN)
    d.arrow([(894, 435), (925, 435), (925, 291), (954, 291)], GREEN)
    d.arrow([(1070, 316), (1070, 370), (851, 370), (851, 410)], GREEN)
    d.text(959, 350, "Update cache", 11, GREEN)
    d.box(954, 442, 272, 94, "Return response", "Detokenize output tokens")
    d.arrow([(1180, 316), (1180, 442)])
    d.text(1192, 383, "Stop", 12)
    d.arrow([(1226, 259), (1244, 259), (1244, 190), (1090, 190), (1090, 210)])
    d.text(1166, 177, "Continue", 12, MUTED, ha="center")
    d.text(54, 438, "The cache saves prefix work.", 20, PURPLE, True, max_width=530)
    d.text(54, 486, "Each new token still needs a model step.", 17, max_width=530)
    d.text(54, 592, "Prefill initializes the cache. Decoding extends it as generation continues.", 17)
    d.finish(pdf)


def integration(pdf):
    d = Drawing("Where Our Method Fits", "PLANNED MLX-LM INTEGRATION  /  Shared model, tokenizer and cache", "02_history_integration_plan",
                "Dashed path: proposed history-policy integration. Full stock-API implementation and timing remain pending.")
    d.box(54, 205, 256, 100, "Prefill", "Same prompt and model", PALE_BLUE, BLUE)
    d.box(366, 205, 300, 100, "Retrieve history", "Find a candidate continuation")
    d.box(722, 205, 300, 100, "Choose draft length", "k = 0, 2, 4, 8 or 16", PALE_ORANGE, ORANGE)
    d.arrow([(310, 255), (366, 255)], dashed=True)
    d.arrow([(666, 255), (722, 255)], dashed=True)
    d.box(670, 387, 260, 104, "Ordinary step", "One next-token prediction", PALE_BLUE, BLUE)
    d.box(970, 387, 256, 104, "Verify draft", "One batched target forward", PALE_BLUE, BLUE)
    d.arrow([(806, 305), (806, 387)], dashed=True)
    d.text(780, 346, "k = 0", 13, ORANGE, ha="right")
    d.arrow([(950, 305), (1098, 345), (1098, 387)], dashed=True)
    d.text(1112, 346, "k > 0", 13, ORANGE)
    d.box(355, 535, 566, 76, "Keep output and align the KV cache", "")
    d.arrow([(800, 491), (800, 535)], dashed=True)
    d.arrow([(1098, 491), (1098, 573), (921, 573)], dashed=True)
    d.arrow([(625, 535), (625, 353), (516, 353), (516, 305)], dashed=True)
    d.text(54, 405, "History proposes tokens.", 18, PURPLE, True, max_width=570)
    d.text(54, 447, "The target model checks them.", 18, max_width=570)
    d.text(54, 489, "Stop at EOS or the token limit.", 15, MUTED, max_width=570)
    d.finish(pdf)


def recovery(pdf):
    d = Drawing("mlx-lm Cache Recovery: A Small Example", "OFFICIAL SPECULATIVE PATH  /  Target cache only, with a trimmable KV cache", "03_mlx_lm_cache_recovery",
                "Source: speculative_generate_step(), _rewind_cache(), KVCache.trim(). Token letters are schematic.")
    d.text(54, 198, "Draft: A B C D     Target accepts A B and predicts Z instead of C.", 18, PURPLE, True)
    start, gap, tw = 416, 12, 83

    def row(y, label, tokens, valid, suffix=""):
        d.text(54, y + 24, label, 15, bold=True, max_width=350)
        d.rect(281, y, 122, 49, "white")
        d.text(342, y + 24, "Prefix", 14, ha="center")
        for i, token in enumerate(tokens):
            fill = PALE_GREEN if i < valid else PALE_ORANGE
            edge = GREEN if i < valid else ORANGE
            d.rect(start + i * (tw + gap), y, tw, 49, fill, edge)
            d.text(start + i * (tw + gap) + tw / 2, y + 24, token, 16, ha="center", bold=True)
            if i >= valid:
                d.ax.plot([start + i * (tw + gap) + 8, start + i * (tw + gap) + tw - 8],
                          [y + 42, y + 7], color=ORANGE, lw=1.2)
        if suffix:
            d.text(938, y + 24, suffix, 13, MUTED, max_width=285)

    row(250, "1  Before verification", [], 0, "x is the pending input token")
    d.rect(416, 250, 83, 49, PALE_BLUE, BLUE)
    d.text(457, 274, "x", 16, ha="center", bold=True)
    d.text(515, 274, "Not cached yet", 13, BLUE)
    row(340, "2  After verification", ["x", "A", "B", "C", "D"], 3)
    d.text(938, 364, "Trim 4 - 2 = 2 entries", 13, ORANGE)
    row(430, "3  After cache trim", ["x", "A", "B"], 3, "Z is the next input")
    row(520, "4  Next model step", ["x", "A", "B", "Z"], 4, "Process Z and continue")
    d.text(54, 615, "Keep the accepted prefix. Rewind the invalid suffix. No prefix replay in this path.", 17, PURPLE, True)
    d.finish(pdf)


def cost(pdf):
    d = Drawing("Why Use Time per Output Token?", "CURRENT POLICY SCORE  /  Recovery depends on the implementation", "04_cost_score_and_recovery",
                "Local policy: src/tiny_llm/adaptive_decode.py. This explains the score and does not validate its predictions.")
    d.text(640, 216, r"$C(k) = \frac{V(k) + L + R(k)}{G(k)}$", 32, PURPLE, ha="center")
    d.text(640, 287, "Estimated time for one round / estimated output advance", 18, ha="center")
    d.text(54, 355, "V(k)  Verification forward, including KV-cache use", 17, bold=True)
    d.text(54, 397, "L       History lookup time", 17)
    d.text(54, 439, "G(k)  Expected accepted prefix plus one target token", 17)
    d.text(54, 481, "R(k)  Extra recovery work in the current replay path", 17)
    d.rect(835, 331, 391, 177, "white", STROKE)
    d.text(855, 358, "Two recovery paths", 17, PURPLE, True)
    d.text(855, 400, "Local: restore + replay", 15)
    d.text(855, 439, "mlx-lm: trim cache + continue", 15)
    d.text(855, 482, "Replay cost is not universal.", 14, ORANGE, True)
    d.text(54, 556, "Choose reuse only when its score beats an ordinary step by the policy margin.", 17, PURPLE, True)
    d.text(54, 605, "G(k) uses an acceptance approximation. Cache-trim and policy overhead still need checking.", 15, MUTED)
    d.finish(pdf)


with PdfPages(OUT / "mlx_lm_procedure_figures.pdf") as pdf:
    ordinary(pdf)
    integration(pdf)
    recovery(pdf)
    cost(pdf)

(OUT / "validation.json").write_text(json.dumps(CHECKS, indent=2))
repo = ROOT.parents[1]
manifest = {
    "date": "2026-11-16",
    "status": "Explanatory diagrams only. No new timing data. Reviewed PPT unchanged.",
    "official_sources": [
        "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/generate.py",
        "https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/models/cache.py",
    ],
    "official_scope": "Source inspected on this date; not a claim about the unavailable external experiment version.",
    "local_sources": {
        name: hashlib.sha256((repo / name).read_bytes()).hexdigest()
        for name in ["src/tiny_llm/adaptive_decode.py", "src/tiny_llm/native_history.py"]
    },
    "worked_example": {"draft_tokens": 4, "accepted_tokens": 2,
                       "target_cache_entries_trimmed": 2, "pending_correction_token": "Z"},
}
(OUT / "sources.json").write_text(json.dumps(manifest, indent=2))
print(f"Exported {len(CHECKS)} figures to {OUT}")
