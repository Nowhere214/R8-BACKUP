"""Redraw the native-component comparison without the acceptance-rule arm."""

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sources/reported_results.json"
source_bytes = SOURCE.read_bytes()
row = json.loads(source_bytes)["native"]
kept = [(m, v) for m, v in zip(row["methods"], row["speedup"]) if m != "Acceptance rule"]
assert kept == [("Ordinary KV", 1.0), ("Fixed history", 1.34), ("Dynamic control", 1.6)]

sys.path.insert(0, "/Users/laiqixiao/.codex/skills/scientific-visualization/scripts")
from figure_export import save_publication_figure
from style_presets import apply_publication_style

apply_publication_style("default")
plt.rcParams.update({"font.family": "Arial", "font.size": 15, "axes.labelsize": 15,
                     "xtick.labelsize": 14, "ytick.labelsize": 15, "svg.fonttype": "none",
                     "pdf.fonttype": 42, "figure.constrained_layout.use": False})
fig, ax = plt.subplots(figsize=(8.2, 4.5))
fig.set_layout_engine("none")
methods, values = zip(*kept)
bars = ax.barh(methods, values, color=["#7A7A7A", "#0072B2", "#009E73"], height=0.53)
ax.invert_yaxis()
ax.set_xlim(0, 2)
ax.set_xticks([i / 4 for i in range(9)])
ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
ax.set_xlabel("Speedup over ordinary decoding")
ax.set_axisbelow(True)
ax.xaxis.grid(True, color="#DDDDDD", linewidth=0.6)
for name in ("top", "right", "left"):
    ax.spines[name].set_visible(False)
ax.tick_params(axis="y", length=0, pad=10)
for bar, value in zip(bars, values):
    ax.text(value + 0.048, bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}x", va="center", fontsize=16, fontweight="bold")
fig.subplots_adjust(left=0.30, right=0.96, bottom=0.20, top=0.97)
fig.canvas.draw()
renderer = fig.canvas.get_renderer()
for label in [*ax.texts, *ax.get_yticklabels(), *ax.get_xticklabels(), ax.xaxis.label]:
    bounds = label.get_window_extent(renderer)
    assert bounds.x0 >= 0 and bounds.y0 >= 0
    assert bounds.x1 <= fig.bbox.width and bounds.y1 <= fig.bbox.height, label.get_text()
out = ROOT / "exports/native_three_methods_2026_11_19"
paths = save_publication_figure(fig, out, formats=["png", "pdf", "svg"], dpi=300)
assert len(paths) == 3
assert SOURCE.read_bytes() == source_bytes
(out.with_suffix(".json")).write_text(json.dumps({
    "source": str(SOURCE), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    "methods": list(methods), "speedup": list(values), "omitted_from_figure": "Acceptance rule",
    "scope": "Native model/cache components in a shared synchronous loop; not full API results.",
    "original_data_and_figure_preserved": True,
    "uncertainty": "Not shown in supplied figure; no new intervals or measurements added."
}, indent=2) + "\n")
plt.close(fig)
