"""Methods figure — the two channels the information decomposition is taken over.

Adapted from scratch_abmil/slides/gen_slide_figures.py::fig_architecture, with
the manuscript's notation and without the slide-deck title.

The point of the figure is that the two channels are disjoint until the
classifier head, which is what makes the decomposition of Eq. (pid) well posed:
neither channel can launder information through the other.
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "two_channel_architecture.png")

EMB_FILL, EMB_EDGE = "#dbeafe", "#1e3a8a"
PAT_FILL, PAT_EDGE = "#fef3c7", "#a16207"
HEAD_FILL = "#e2e8f0"


def box(ax, x, y, w, h, txt, fc, ec, fs=11, fw="normal"):
    ax.add_patch(plt.Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc,
                               edgecolor=ec, linewidth=1.4, zorder=3))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
            fontweight=fw, zorder=4, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color="#334155", lw=1.4):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle="->", color=color, lw=lw))


fig, ax = plt.subplots(figsize=(11, 6.2))
ax.set_xlim(0, 12)
ax.set_ylim(0, 7)
ax.axis("off")

ax.text(6.0, 6.72, "The two channels the decomposition is taken over",
        ha="center", fontsize=14, fontweight="bold", color="#1e293b")

ax.text(1.8, 6.15, "morphology channel", ha="center", fontsize=11,
        color=EMB_EDGE, fontweight="bold")
ax.text(1.8, 4.15, "composition channel", ha="center", fontsize=11,
        color=PAT_EDGE, fontweight="bold")

box(ax, 1.8, 5.4, 2.6, 0.9,
    "$\\mathbf{h}_i$\nCTransPath, 512-d", EMB_FILL, EMB_EDGE)
box(ax, 1.8, 3.4, 2.6, 0.9,
    "$\\mathbf{p}_i$\npattern probs, 6-d", PAT_FILL, PAT_EDGE)

box(ax, 6.0, 5.4, 3.2, 0.9,
    "PI-ABMIL gated attention\n$\\alpha_i = \\mathrm{softmax}(\\cdot)$",
    EMB_FILL, EMB_EDGE)
box(ax, 6.0, 3.4, 3.2, 0.9,
    "FC-MIL Choquet integral\n$\\int \\mathbf{p}\\, d\\mu$, 2-additive",
    PAT_FILL, PAT_EDGE)
arrow(ax, 3.15, 5.4, 4.35, 5.4, EMB_EDGE)
arrow(ax, 3.15, 3.4, 4.35, 3.4, PAT_EDGE)

box(ax, 10.3, 5.4, 1.9, 0.9,
    "$\\mathbf{z}_{\\mathrm{attn}}$\n256-d", EMB_FILL, EMB_EDGE, fw="bold")
box(ax, 10.3, 3.4, 1.9, 0.9,
    "$\\mathbf{c}_s$\n6-d", PAT_FILL, PAT_EDGE, fw="bold")
arrow(ax, 7.65, 5.4, 9.3, 5.4, EMB_EDGE)
arrow(ax, 7.65, 3.4, 9.3, 3.4, PAT_EDGE)

box(ax, 6.0, 1.95, 5.2, 0.9,
    "$[\\,\\mathbf{c}_s \\;\\Vert\\; \\mathbf{z}_{\\mathrm{attn}}\\,]"
    " \\;\\rightarrow\\; \\mathrm{Linear} \\;\\rightarrow\\;"
    " P(\\mathrm{mut})$",
    HEAD_FILL, "#334155", fs=12, fw="bold")
arrow(ax, 10.0, 4.95, 7.9, 2.40, "#64748b")
arrow(ax, 10.0, 2.95, 7.9, 2.40, "#64748b")

ax.text(6.0, 0.55,
        "Attention sees only the embeddings and the Choquet integral only the "
        "pattern probabilities.\nThe channels meet nowhere before the "
        "classifier head, so neither can launder information\nthrough the "
        "other and the decomposition of the two sources is well posed.",
        ha="center", fontsize=10.5, style="italic", color="#475569",
        linespacing=1.45)

fig.savefig(OUT, dpi=260, bbox_inches="tight", facecolor="white")
plt.close(fig)
print(f"Saved: {OUT}")
