"""Generate the three-regime taxonomy tree (Figure 2 of Paper 1).

  Root: "Where is the discriminative signal p(slide | gene mutant)?"
   |
   +-- Regime (i)   encoder saturation                  -> TP53, EGFR
   |
   +-- Regime (ii)  inter-pattern co-occurrence
   |     +-- (ii-a) additive complementarity            -> (not populated here)
   |     +-- (ii-b) non-additive synergy                -> KRAS
   |
   +-- Regime (iii) out-of-ontology
         +-- Label-space mismatch                       -> STK11
         +-- No morphological projection                -> KEAP1
         +-- Collapse-to-prior                          -> RBM10

Ported from the thesis generator, with four corrections so the figure
speaks the same language as the manuscript:

  1. Regimes are numbered (i)/(ii)/(iii) as in the Methods and in
     tab:pid_regimes, not lettered A/B/C, and the columns now run in
     numerical order left to right. The thesis lettering put
     out-of-ontology second and the cross-term third, which inverts
     the (ii)/(iii) order used everywhere in the text.
  2. Regime (ii) is split into its additive sub-form (ii-a) and its
     non-additive sub-form (ii-b). Only (ii-b) is populated in
     TCGA-LUAD; (ii-a) is drawn empty and labelled as such, because it
     is the sub-form the CPTAC refit and the glioma 1p/19q case read.
  3. Regime colours follow the palette declared in
     gen_domain_translation.py -- green saturation, orange
     co-occurrence, red out-of-ontology. The thesis figure painted
     out-of-ontology orange and the cross-term blue, so the same
     regime carried different colours in different figures.
  4. The footer no longer cites "Table 6.10" (thesis numbering).

Per-gene slide-level values (P, outcome, Delta_pat) are unchanged from
the thesis figure.

Run:  python gen_signal_tree.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 300,
    "text.usetex": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))

# Regime palette, shared with gen_domain_translation.py:
#   green = encoder saturation, orange = co-occurrence, red = out-of-ontology.
GREEN_BG, GREEN_EC = "#D5F5E3", "#27AE60"
ORANGE_BG, ORANGE_EC = "#FDEBD0", "#E67E22"
RED_BG, RED_EC = "#FADBD8", "#C0392B"
GREY_BG, GREY_EC = "#F4F6F7", "#7F8C8D"
DARK = "#2C3E50"
MUTED = "#95A5A6"

fig, ax = plt.subplots(figsize=(15.0, 9.8))
ax.set_xlim(0, 100)
ax.set_ylim(0, 100)
ax.axis("off")


def node_box(cx, cy, w, h, text_lines, fc, ec, fontsize=9,
             bold_first=False, ls="solid", text_color=None):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.35,rounding_size=1.2",
        facecolor=fc, edgecolor=ec, linewidth=1.4, linestyle=ls,
    ))
    n = max(len(text_lines), 1)
    line_h = (h - 1.4) / n
    top = cy + h / 2 - 0.9
    for i, line in enumerate(text_lines):
        y = top - line_h * (i + 0.5)
        weight = "bold" if (i == 0 and bold_first) else "normal"
        ax.text(cx, y, line, ha="center", va="center", fontsize=fontsize,
                color=text_color or DARK, fontweight=weight)


def edge_l(x0, y0, x1, y1, color, width=1.0, ls="solid"):
    ymid = (y0 + y1) / 2
    ax.plot([x0, x0], [y0, ymid], color=color, linewidth=width, linestyle=ls)
    ax.plot([x0, x1], [ymid, ymid], color=color, linewidth=width, linestyle=ls)
    ax.plot([x1, x1], [ymid, y1], color=color, linewidth=width, linestyle=ls)


def vline(cx, y_top, y_bot, color, width=1.0, ls="solid"):
    ax.plot([cx, cx], [y_top, y_bot], color=color, linewidth=width,
            linestyle=ls)


# --------------------------------------------------------------------
# Title
# --------------------------------------------------------------------
ax.text(50, 97.0,
        "What kind of problem is morphology  $\\rightarrow$  mutation "
        "prediction for each gene?",
        ha="center", va="center", fontsize=13.5, fontweight="bold",
        color=DARK)
ax.text(50, 94.0,
        "Three regimes, read as the atoms of the partial information "
        "decomposition of the pattern and embedding channels",
        ha="center", va="center", fontsize=9.5, color=DARK, style="italic")

# --------------------------------------------------------------------
# Layout
# --------------------------------------------------------------------
ROOT_Y, L1_Y, L2_Y, L3_Y, BAND_Y = 86.0, 73.0, 55.0, 31.0, 12.5

I_X = 11.0                      # regime (i)
II_X = 40.0                     # regime (ii)
III_X = 79.0                    # regime (iii)

I_TP53_X, I_EGFR_X = 6.0, 17.0
IIA_X, IIB_X = 32.0, 48.0       # (ii) sub-forms
III_M1_X, III_M2_X, III_M3_X = 65.0, 79.0, 93.0

# --------------------------------------------------------------------
# Level 0: root
# --------------------------------------------------------------------
node_box(50, ROOT_Y, 56, 5.6,
         ["Where is the discriminative signal "
          "$p(\\mathrm{slide}\\,|\\,\\mathrm{gene\\,mutant})$?",
          "(CTransPath embedding 512-d  +  6-pattern ANORAK simplex)"],
         fc=GREY_BG, ec=DARK, fontsize=9.3, bold_first=True)

# --------------------------------------------------------------------
# Level 1: the three regimes, in numerical order
# --------------------------------------------------------------------
node_box(I_X, L1_Y, 20, 8,
         ["Regime (i)",
          "encoder saturation",
          "(within the CTransPath embedding)"],
         fc=GREEN_BG, ec=GREEN_EC, fontsize=8.6, bold_first=True)
node_box(II_X, L1_Y, 32, 8,
         ["Regime (ii)",
          "inter-pattern co-occurrence",
          "(pattern channel carries what the encoder does not)"],
         fc=ORANGE_BG, ec=ORANGE_EC, fontsize=8.6, bold_first=True)
node_box(III_X, L1_Y, 40, 8,
         ["Regime (iii)",
          "out-of-ontology signal",
          "(no exploitable morphological footprint within the simplex)"],
         fc=RED_BG, ec=RED_EC, fontsize=8.6, bold_first=True)

edge_l(50, ROOT_Y - 2.8, I_X, L1_Y + 4, GREEN_EC, width=1.2)
edge_l(50, ROOT_Y - 2.8, II_X, L1_Y + 4, ORANGE_EC, width=1.2)
edge_l(50, ROOT_Y - 2.8, III_X, L1_Y + 4, RED_EC, width=1.2)

# --------------------------------------------------------------------
# Level 2: sub-forms of (ii) and sub-mechanisms of (iii)
# --------------------------------------------------------------------
node_box(IIA_X, L2_Y, 14, 9,
         ["(ii-a) additive",
          "complementarity",
          "concat fusion suffices"],
         fc="white", ec=MUTED, fontsize=8.2, bold_first=True, ls="dashed",
         text_color=MUTED)
node_box(IIB_X, L2_Y, 14, 9,
         ["(ii-b) non-additive",
          "synergy",
          "$\\mathrm{FC}>\\mathrm{PI}$, $I_{jk}>0$"],
         fc="white", ec=ORANGE_EC, fontsize=8.2, bold_first=True)

node_box(III_M1_X, L2_Y, 13, 9,
         ["Label-space", "mismatch", "(signal in TME)"],
         fc="white", ec=RED_EC, fontsize=8.2, bold_first=True)
node_box(III_M2_X, L2_Y, 13, 9,
         ["No morphological", "projection", "(biochemical regulator)"],
         fc="white", ec=RED_EC, fontsize=8.2, bold_first=True)
node_box(III_M3_X, L2_Y, 13, 9,
         ["Collapse-to-prior", "(low prevalence,", "$n=37$ mutated slides)"],
         fc="white", ec=RED_EC, fontsize=8.2, bold_first=True)

edge_l(II_X, L1_Y - 4, IIA_X, L2_Y + 4.5, MUTED, ls="dashed")
edge_l(II_X, L1_Y - 4, IIB_X, L2_Y + 4.5, ORANGE_EC)
edge_l(III_X, L1_Y - 4, III_M1_X, L2_Y + 4.5, RED_EC)
vline(III_M2_X, L1_Y - 4, L2_Y + 4.5, RED_EC)
edge_l(III_X, L1_Y - 4, III_M3_X, L2_Y + 4.5, RED_EC)

# (ii-a) carries no gene in this cohort; say so where its leaf would be.
node_box(IIA_X, L3_Y, 13, 8,
         ["not populated", "in TCGA-LUAD", "(see CPTAC, 1p/19q)"],
         fc="#FBFCFC", ec=MUTED, fontsize=8.0, ls="dashed", text_color=MUTED)
vline(IIA_X, L2_Y - 4.5, L3_Y + 4.0, MUTED, ls="dashed")


# --------------------------------------------------------------------
# Level 3: gene leaves
# --------------------------------------------------------------------
def gene_leaf(cx, gene, p_value, outcome, dpat, fc, ec, w=11, h=12):
    ax.add_patch(FancyBboxPatch(
        (cx - w / 2, L3_Y - h / 2), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.0",
        facecolor=fc, edgecolor=ec, linewidth=1.4,
    ))
    ax.text(cx, L3_Y + h / 2 - 2.4, gene, ha="center", va="center",
            fontsize=11.5, fontweight="bold", color=ec)
    ax.text(cx, L3_Y - 0.4, r"$P = %s$    %s" % (p_value, outcome),
            ha="center", va="center", fontsize=8.3, color=DARK)
    ax.text(cx, L3_Y - h / 2 + 2.4,
            r"$\Delta_{\mathrm{pat}} = %s$ pp" % dpat,
            ha="center", va="center", fontsize=7.8, color=DARK)


gene_leaf(I_TP53_X, "TP53", "0.951", "TP", "$-$16.1", GREEN_BG, GREEN_EC,
          w=10)
gene_leaf(I_EGFR_X, "EGFR", "0.076", "FN", "$-$0.5", GREEN_BG, GREEN_EC,
          w=10)
gene_leaf(IIB_X, "KRAS", "0.965", "TP", "$+$7.6", ORANGE_BG, ORANGE_EC)
gene_leaf(III_M1_X, "STK11", "0.046", "FN", "$-$5.5", RED_BG, RED_EC)
gene_leaf(III_M2_X, "KEAP1", "0.247", "FN", "$-$44.7", RED_BG, RED_EC)
gene_leaf(III_M3_X, "RBM10", "0.017", "FN", "$+$0.3", RED_BG, RED_EC)

edge_l(I_X, L1_Y - 4, I_TP53_X, L3_Y + 6.0, GREEN_EC)
edge_l(I_X, L1_Y - 4, I_EGFR_X, L3_Y + 6.0, GREEN_EC)
vline(IIB_X, L2_Y - 4.5, L3_Y + 6.0, ORANGE_EC)
vline(III_M1_X, L2_Y - 4.5, L3_Y + 6.0, RED_EC)
vline(III_M2_X, L2_Y - 4.5, L3_Y + 6.0, RED_EC)
vline(III_M3_X, L2_Y - 4.5, L3_Y + 6.0, RED_EC)

# --------------------------------------------------------------------
# Cohort verdict band
# --------------------------------------------------------------------
ax.text(50, BAND_Y + 6.5,
        r"Cohort verdict (Hosmer--Lemeshow gate, AUROC threshold $0.70$)",
        ha="center", va="center", fontsize=9, color=DARK, fontweight="bold")

ax.add_patch(FancyBboxPatch(
    (1.0, BAND_Y - 5), 30, 9,
    boxstyle="round,pad=0.35,rounding_size=1.4",
    facecolor=GREEN_BG, edgecolor=GREEN_EC, linewidth=1.0, alpha=0.55))
ax.text(16, BAND_Y + 1.6, r"Conclusive  (AUROC $\geq 0.70$)",
        ha="center", va="center", fontsize=9, color=GREEN_EC,
        fontweight="bold")
ax.text(16, BAND_Y - 1.8, r"TP53 ($0.718$)    EGFR ($0.701$)",
        ha="center", va="center", fontsize=8.5, color=DARK)

# Neutral grey: this band mixes regimes (ii-b) and (iii), so it must not
# borrow either regime colour.
ax.add_patch(FancyBboxPatch(
    (33.0, BAND_Y - 5), 66, 9,
    boxstyle="round,pad=0.35,rounding_size=1.4",
    facecolor=GREY_BG, edgecolor=GREY_EC, linewidth=1.0, alpha=0.85))
ax.text(66, BAND_Y + 1.6, r"Inconclusive  (AUROC $< 0.70$)",
        ha="center", va="center", fontsize=9, color=GREY_EC,
        fontweight="bold")
ax.text(66, BAND_Y - 1.8,
        r"KRAS ($0.609$)    STK11 ($0.695$)    "
        r"KEAP1 ($0.610$)    RBM10 ($0.661$)",
        ha="center", va="center", fontsize=8.5, color=DARK)

# --------------------------------------------------------------------
# Footer
# --------------------------------------------------------------------
ax.text(50, 1.4,
        r"$\Delta_{\mathrm{pat}} = P_{\mathrm{combined}} - "
        r"P_{\mathrm{emb-only}}$ (pp)      |      "
        r"per-slide values from the case-study slides",
        ha="center", va="center", fontsize=8, color=DARK)

out = os.path.join(HERE, "signal_tree.png")
plt.savefig(out, bbox_inches="tight", dpi=300, facecolor="white")
plt.close()
print("wrote", out)
