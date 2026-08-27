"""Generate the two 'domain-translation' conceptual figures for Paper 1.

Outputs (300 dpi PNG, written next to this script):
  * domain_translation_concept.png : source morphology -> MIL translator ->
    target mutations, with per-gene AUROC as translational faithfulness.
  * fidelity_spectrum.png          : faithfulness (representative AUROC) vs
    training-cohort size across analogous medical cross-domain mapping tasks,
    locating the LUAD map in the low-faithfulness / low-data corner.

Run:  python gen_domain_translation.py
"""
from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.linewidth": 0.8,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

HERE = os.path.dirname(os.path.abspath(__file__))

# Regime colours (shared with signal_tree taxonomy).
C_SAT = "#2ca02c"   # encoder saturation  (TP53, EGFR)
C_COO = "#ff7f0e"   # inter-pattern co-occurrence (KRAS)
C_OOO = "#d62728"   # out-of-ontology (STK11, KEAP1, RBM10)
C_SRC = "#4c72b0"   # source-domain accent
C_TRN = "#7f7f7f"   # translator accent
INK = "#1a1a1a"


def _rbox(ax, x, y, w, h, fc, ec, lw=1.4, rad=0.02, alpha=1.0, z=2):
    box = FancyBboxPatch(
        (x, y), w, h,
        boxstyle=f"round,pad=0.0,rounding_size={rad}",
        linewidth=lw, edgecolor=ec, facecolor=fc, alpha=alpha, zorder=z,
    )
    ax.add_patch(box)
    return box


# ---------------------------------------------------------------------------
# Figure A: domain-translation concept
# ---------------------------------------------------------------------------
def fig_concept():
    fig, ax = plt.subplots(figsize=(12.4, 5.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    # ---- SOURCE DOMAIN ----
    _rbox(ax, 2, 16, 27, 74, "#eef2fa", C_SRC, lw=1.8, rad=0.03)
    ax.text(15.5, 85, "SOURCE DOMAIN", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color=C_SRC)
    ax.text(15.5, 80, "Tissue morphology (H&E)", ha="center", va="center",
            fontsize=11, color=INK)
    ax.text(15.5, 74.5, "vocabulary: 6 ANORAK growth patterns",
            ha="center", va="center", fontsize=9.5, style="italic", color=INK)

    patterns = ["lepidic", "acinar", "papillary",
                "micropapillary", "solid", "cribriform"]
    for i, p in enumerate(patterns):
        yy = 66.5 - i * 7.2
        _rbox(ax, 6.5, yy - 2.5, 18, 5, "white", C_SRC, lw=1.0, rad=0.06)
        ax.text(15.5, yy, p, ha="center", va="center", fontsize=9.5, color=INK)

    ax.text(15.5, 20.5,
            "rendered by FuzzyArcLoss V2\nmacro-F1 92.3% · best cribriform F1",
            ha="center", va="center", fontsize=8.6, color=C_SRC,
            fontweight="bold")

    # ---- TRANSLATOR ----
    _rbox(ax, 39, 40, 20, 26, "#f2f2f2", C_TRN, lw=1.8, rad=0.04)
    ax.text(49, 60, "TRANSLATOR", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#4d4d4d")
    ax.text(49, 54.5, "MIL aggregator", ha="center", va="center",
            fontsize=10.5, color=INK)
    ax.text(49, 49.5, "PI-ABMIL / FC-MIL", ha="center", va="center",
            fontsize=9.5, style="italic", color=INK)
    ax.text(49, 44.5, "(2-additive Choquet\n= non-additive grammar)",
            ha="center", va="center", fontsize=8.4, color="#4d4d4d")

    # arrows source -> translator -> target
    ax.add_patch(FancyArrowPatch((29.4, 53), (38.6, 53),
                 arrowstyle="-|>", mutation_scale=22, lw=2.2, color=C_TRN))
    ax.add_patch(FancyArrowPatch((59.4, 53), (68.6, 53),
                 arrowstyle="-|>", mutation_scale=22, lw=2.2, color=C_TRN))
    ax.text(64, 58.8, "per-gene\nAUROC", ha="center", va="center",
            fontsize=7.2, color="#4d4d4d", fontweight="bold", linespacing=0.95)
    ax.text(64, 48.5, "= faithfulness", ha="center", va="center",
            fontsize=6.9, color="#4d4d4d", style="italic")

    # ---- TARGET DOMAIN ----
    _rbox(ax, 69, 16, 29, 74, "#fdf2f0", "#b0b0b0", lw=1.8, rad=0.03)
    ax.text(83.5, 85, "TARGET DOMAIN", ha="center", va="center",
            fontsize=12.5, fontweight="bold", color="#8a2f2f")
    ax.text(83.5, 80, "Somatic driver mutations", ha="center", va="center",
            fontsize=11, color=INK)

    genes = [
        ("TP53",  "AUROC 0.718", C_SAT, "translates well"),
        ("EGFR",  "AUROC 0.701", C_SAT, "translates well"),
        ("KRAS",  "AUROC 0.609", C_COO, "needs pattern grammar"),
        ("STK11", "AUROC 0.695", C_OOO, "out-of-ontology"),
        ("KEAP1", "AUROC 0.610", C_OOO, "out-of-ontology"),
        ("RBM10", "AUROC 0.661", C_OOO, "out-of-ontology"),
    ]
    for i, (g, a, c, note) in enumerate(genes):
        yy = 72.5 - i * 9.4
        _rbox(ax, 72, yy - 3.9, 23, 7.8, "white", c, lw=1.8, rad=0.05)
        ax.add_patch(plt.Rectangle((72, yy - 3.9), 1.6, 7.8, color=c, zorder=3))
        ax.text(76, yy + 1.5, g, ha="left", va="center", fontsize=10.5,
                fontweight="bold", color=INK)
        ax.text(76, yy - 1.9, note, ha="left", va="center", fontsize=7.8,
                style="italic", color=c)
        ax.text(94, yy + 1.5, a, ha="right", va="center", fontsize=9.0, color=INK)

    # regime legend
    leg = [("encoder saturation", C_SAT),
           ("inter-pattern co-occurrence", C_COO),
           ("out-of-ontology", C_OOO)]
    for i, (lab, c) in enumerate(leg):
        x0 = 8 + i * 30
        ax.add_patch(plt.Rectangle((x0, 6.5), 2.2, 2.2, color=c, zorder=3))
        ax.text(x0 + 3.2, 7.6, lab, ha="left", va="center", fontsize=8.6,
                color=INK)

    ax.text(50, 96,
            "Histology-to-mutation prediction as an imperfect domain translation",
            ha="center", va="center", fontsize=13.5, fontweight="bold",
            color=INK)

    out = os.path.join(HERE, "domain_translation_concept.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


# ---------------------------------------------------------------------------
# Figure B: fidelity spectrum (faithfulness vs data scale)
# ---------------------------------------------------------------------------
def fig_spectrum():
    fig, ax = plt.subplots(figsize=(11.2, 6.6))

    # (label, reported AUROC, training n, category, label offset)
    #   category: 'analogue' or one of the three LUAD regimes.
    # Every analogue is the AUROC as published, not accuracy: Chang et al. report
    # IDH 0.91 / 1p19q 0.88 / MGMT 0.81 AUC alongside the 94/92/83% accuracies, and
    # the accuracies are what an earlier draft of this figure plotted. MGMT is shown
    # at the externally validated 0.562 (Kim et al., models trained on the n=585
    # BraTS-2021 cohort), because the 0.81 development-cohort value is exactly the
    # non-replicating result the Discussion is about.
    pts = [
        ("Virtual staining\n(autofluorescence->H&E)", 0.99, 700, "analogue", (-12, 20)),
        ("ECG -> LV function\n(EF<=35%)",             0.93, 44959, "analogue", (-10, 16)),
        ("Radiogenomics\nglioma IDH",                 0.91, 259, "analogue", (-6, 14)),
        ("Pathomics -> MSI\n(GI cancer)",             0.81, 315, "analogue", (-20, 14)),
        ("Retinal -> MACE\n(cardiovascular)",         0.70, 284335, "analogue", (12, 0)),
        ("Radiogenomics glioma MGMT\n(external validation)",
                                                      0.562, 585, "analogue", (2, 62)),
        ("TP53",  0.718, 687, "sat", (14, 16)),
        ("EGFR",  0.701, 687, "sat", (14, -18)),
        ("STK11", 0.695, 687, "ooo", (2, 34)),
        ("RBM10", 0.661, 687, "ooo", (6, -26)),
        # KRAS (0.609) and KEAP1 (0.610) coincide; draw KEAP1 last so the triangle
        # is not buried under the diamond.
        ("KRAS",  0.609, 687, "coo", (-6, -34)),
        ("KEAP1", 0.610, 687, "ooo", (20, 30)),
    ]
    cat_color = {"analogue": "#6b6b6b", "sat": C_SAT, "coo": C_COO, "ooo": C_OOO}
    cat_marker = {"analogue": "o", "sat": "s", "coo": "D", "ooo": "^"}

    for lab, au, n, cat, off in pts:
        is_luad = cat != "analogue"
        ax.scatter(au, n, s=190 if is_luad else 150,
                   c=cat_color[cat], marker=cat_marker[cat],
                   edgecolors="black", linewidths=1.0,
                   alpha=0.95 if is_luad else 0.65, zorder=4)
        # leader line for the crowded LUAD points so labels can sit apart
        arrow = dict(arrowstyle="-", color=cat_color[cat], lw=0.8,
                     shrinkA=0, shrinkB=3, alpha=0.9) if is_luad else None
        ha = "right" if off[0] < 0 else "left"
        ax.annotate(lab, (au, n), textcoords="offset points",
                    xytext=off, fontsize=8.2,
                    color=cat_color[cat] if is_luad else "#3a3a3a",
                    fontweight="bold" if is_luad else "normal",
                    ha=ha, va="center", zorder=6, arrowprops=arrow)

    ax.set_yscale("log")
    ax.set_xlim(0.55, 1.02)
    ax.set_ylim(150, 700000)
    ax.set_xlabel("Translational faithfulness  (reported AUROC; virtual staining: "
                  "pathologist concordance)", fontsize=11)
    ax.set_ylabel("Training cohort size  (slides / patients, log scale)", fontsize=11)

    # conclusive threshold
    ax.axvline(0.70, ls="--", lw=1.3, color="#555555", zorder=1)
    ax.text(0.695, 190, "Conclusive\nthreshold (0.70)", fontsize=8.6,
            color="#555555", ha="right", va="bottom")

    # shade the low-faithfulness / low-data corner
    ax.axvspan(0.55, 0.72, ymin=0, ymax=0.42, color="#d62728", alpha=0.06, zorder=0)
    ax.text(0.558, 6000, "low-faithfulness,\nlow-data corner",
            fontsize=8.8, color=C_OOO, ha="left", va="center", style="italic")

    # legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#6b6b6b",
               markeredgecolor="black", markersize=9, label="analogous medical mapping"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_SAT,
               markeredgecolor="black", markersize=9, label="LUAD: encoder saturation"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=C_COO,
               markeredgecolor="black", markersize=9, label="LUAD: co-occurrence"),
        Line2D([0], [0], marker="^", color="w", markerfacecolor=C_OOO,
               markeredgecolor="black", markersize=9, label="LUAD: out-of-ontology"),
    ]
    ax.legend(handles=handles, loc="center left", bbox_to_anchor=(0.005, 0.60),
              fontsize=8.8, frameon=True, framealpha=0.95)

    ax.set_title("A fidelity spectrum of medical cross-domain mapping tasks",
                 fontsize=13, fontweight="bold")
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)

    out = os.path.join(HERE, "fidelity_spectrum.png")
    fig.savefig(out)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    fig_concept()
    fig_spectrum()
