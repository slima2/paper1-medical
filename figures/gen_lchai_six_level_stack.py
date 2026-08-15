"""Figure 4 — LCHAI v2 system overview and six-level interpretability stack.

Derived from thesis_edits/gen_lchai_pipeline.py, with the upstream encoder
band added and the thesis-only cross-references (RQ numbers, artifact_pipeline)
removed so the figure stands alone in the manuscript.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "lchai_six_level_stack.png")

W, H = 16.0, 13.0
COL_W = 7.1
LX, RX = 0.4, 8.5
ROW_H, ROW_GAP = 0.68, 0.09
STEP = ROW_H + ROW_GAP
TOP = 5.62


def row_y(i):
    return TOP - (i + 1) * STEP


def box(ax, x, y, w, h, text, color, textcolor="white", fontsize=10.5,
        bold=False):
    ax.add_patch(mpatches.FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.06", facecolor=color,
        edgecolor="#334155", linewidth=0.8, zorder=3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, color=textcolor,
            weight="bold" if bold else "normal", zorder=4, linespacing=1.25)


def arrow(ax, x1, y1, x2, y2, color="#94a3b8", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(
        (x1, y1), (x2, y2), connectionstyle=f"arc3,rad={rad}",
        arrowstyle="->", color=color, lw=lw, zorder=2, mutation_scale=13))


fig, ax = plt.subplots(figsize=(W, H))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")

ax.text(W / 2, 12.5, "LCHAI v2 — inference pipeline and six-level "
                     "interpretability stack",
        ha="center", fontsize=16, fontweight="bold", color="#1e293b")

# ---- upstream band: a single forward pass ----------------------------------
band_y, band_h, bw = 10.95, 0.95, 3.6
band = [
    ("Whole-slide image\nH&E · TCGA-LUAD", "#1e40af"),
    ("Tiling\n224$\\times$224 px · tissue mask", "#1d4ed8"),
    ("CTransPath (frozen)\n512-d embedding per tile", "#2563eb"),
    ("FuzzyArcLoss V2\n6-d pattern membership", "#3b82f6"),
]
cx = []
for i, (txt, col) in enumerate(band):
    bx = 0.4 + i * (bw + 0.27)
    box(ax, bx, band_y, bw, band_h, txt, col, fontsize=10)
    cx.append(bx + bw / 2)
for i in range(3):
    arrow(ax, cx[i] + bw / 2, band_y + band_h / 2,
          cx[i + 1] - bw / 2, band_y + band_h / 2, "#93c5fd")

# ---- the two channels the decomposition is taken over ----------------------
ch_y, ch_h, chw = 9.40, 0.95, 5.4
box(ax, 1.1, ch_y, chw, ch_h,
    "Morphology channel — PI-ABMIL\ngated attention over 512-d embeddings",
    "#ea580c", fontsize=10, bold=True)
box(ax, 9.5, ch_y, chw, ch_h,
    "Composition channel — FC-MIL\n2-additive Choquet over 6-d patterns",
    "#0d9488", fontsize=10, bold=True)
arrow(ax, cx[2], band_y, 3.8, ch_y + ch_h, "#93c5fd", rad=0.18)
arrow(ax, cx[3], band_y, 12.2, ch_y + ch_h, "#93c5fd", rad=0.12)

# ---- prediction, then the population-level verdict -------------------------
box(ax, 4.4, 8.15, 7.2, 0.80,
    "Mutation probability $P(\\mathrm{mut})$ — six driver genes",
    "#475569", fontsize=10.5, bold=True)
arrow(ax, 3.8, ch_y, 6.6, 8.95, "#ea580c", rad=-0.15)
arrow(ax, 12.2, ch_y, 9.4, 8.95, "#0d9488", rad=0.15)

box(ax, 3.6, 6.95, 8.8, 0.80,
    "Population verdict — Conclusive if mean AUROC $\\geq$ 0.70",
    "#059669", fontsize=10.5, bold=True)
arrow(ax, 8.0, 8.15, 8.0, 7.75, "#64748b")

# ---- left column: the six per-slide read-outs ------------------------------
box(ax, LX, 5.85, COL_W, 0.58,
    "Per-slide interpretability stack", "#78350f", fontsize=10.5, bold=True)
# Level order follows the manuscript (Results, six-level stack): L1 is the
# MIL attention overlay and L2 the pattern overlay, not the reverse.
levels = [
    "L1  MIL attention — slide heat map",
    "L2  Pattern overlay — tile-level map",
    "L3  SHAP — embedding vs pattern dims",
    "L4  Choquet\u2013Shapley + interaction indices",
    "L5  Channel ablation — $\\Delta$AUROC per slide",
    "L6  Fuzzy linguistic labels",
]
tones = ["#b45309", "#9a4a08"]
for i, txt in enumerate(levels):
    box(ax, LX, row_y(i), COL_W, ROW_H, txt, tones[i % 2], fontsize=10)
arrow(ax, 6.0, 6.95, 3.9, 6.43, "#059669", rad=0.15)

# ---- right column: the dual anti-hallucination layer -----------------------
box(ax, RX, 5.85, COL_W, 0.58,
    "Dual anti-hallucination layer", "#5b21b6", fontsize=10.5, bold=True)
guards = [
    ("Fuzzy labels constrain\n$how$ the agent speaks", "#7c3aed"),
    ("NCIt + MONDO graph constrains\n$what$ it may assert", "#6d28d9"),
    ("DeepSearch constrains\n$which$ sources it cites", "#5b21b6"),
]
for i, (txt, col) in enumerate(guards):
    box(ax, RX, row_y(i), COL_W, ROW_H, txt, col, fontsize=9.5)
box(ax, RX, row_y(3), COL_W, ROW_H,
    "LLM explanation agent", "#4c1d95", fontsize=10.5, bold=True)
box(ax, RX, row_y(5), COL_W, ROW_H + STEP - 0.09,
    "Clinical output\nOntology-grounded case explanation\n"
    "and confidence-labelled mutation report",
    "#b91c1c", fontsize=9.5, bold=True)
arrow(ax, 10.0, 6.95, 12.05, 6.43, "#059669", rad=-0.15)
arrow(ax, RX + COL_W / 2, row_y(3), RX + COL_W / 2,
      row_y(5) + ROW_H + STEP - 0.09, "#7c3aed")
arrow(ax, LX + COL_W, row_y(5) + 0.45, RX, row_y(5) + 0.45, "#b45309")

ax.text(W / 2, 0.32,
        "Fuzzy logic enters at four points: training (FuzzyArcLoss V2), "
        "representation (soft pattern vectors),\n"
        "aggregation (Choquet integral) and interpretation "
        "(fuzzy linguistic labels).",
        ha="center", fontsize=10, color="#475569", linespacing=1.4,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#f8fafc",
                  edgecolor="#cbd5e1", linewidth=0.9))

fig.savefig(OUT, dpi=260, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
