"""Redraw Figure 1 (signal_tree) for the v4 run.

The Aug-30 figure cannot be submitted: it declares TP53 (0.718) and EGFR
(0.701) Conclusive, and under nested cross-validation on the corrected cohort
no target reaches 0.70. It also hangs KRAS off the synergy branch, which the
conditional test no longer supports, and gives RBM10 n = 37 where the
patient-level count is 30.

Every number here is transcribed from tables/tab_mutation_luad.tex,
patient-preserved panel, and every regime assignment from the prose of
03_results_CM.tex, which is quoted in REGIME_SOURCE below so the two cannot
drift apart silently.

Dropped on purpose: the per-slide P values and the Delta_pat column of the old
figure. Both come from the case-study slides under the superseded weights and
neither could be verified against a v4 output; FIGURAS_v4_SPEC.md flags the
KEAP1 value (-44.7 pp) specifically. They are replaced by the four condition
AUROCs, which are verified and which are what the regime reads actually rest
on.

    python scripts/make_signal_tree.py
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

# tab_mutation_luad.tex, panel a (patient-preserved): B2, B3, PI, FC
GENES = {
    "TP53":  dict(b2=.661, b3=.579, pi=.658, fc=.646, npos=220, best=".661 B2"),
    "EGFR":  dict(b2=.582, b3=.480, pi=.580, fc=.567, npos=67,  best=".582 B2"),
    "STK11": dict(b2=.591, b3=.499, pi=.578, fc=.608, npos=58,  best=".608 FC"),
    "KEAP1": dict(b2=.488, b3=.485, pi=.446, fc=.460, npos=79,  best=".488 B2"),
    "KRAS":  dict(b2=.517, b3=.486, pi=.516, fc=.506, npos=124, best=".517 B2"),
    "RBM10": dict(b2=.536, b3=.490, pi=.552, fc=.528, npos=30,  best=".552 PI"),
}

# 03_results_CM.tex, verbatim, so a reader can audit the mapping:
REGIME_SOURCE = """
(i)   "The embedding-only baseline B2 reaches 0.661 and 0.582 ... fusion moves
       the AUROC by <=0.02 ... the pattern channel alone falls 0.082 and 0.102
       below the embedding."  -> TP53, EGFR, directional (no equivalence).
(ii)  "We leave (ii-b) unpopulated."  -> empty; KRAS tested and not supported.
(iii) "STK11 ... is a label-space mismatch"; "KEAP1 is a stronger null than the
       regime allows ... invisible not merely to the vocabulary but to the
       encoder"; "RBM10 (best 0.552) is a collapse to prior at n+ = 30."
"""

GREEN, AMBER, RED, GREY = "#2e7d52", "#c98010", "#c0392b", "#9aa0a6"
FILL = {"green": "#e8f4ed", "amber": "#fdf3e2", "red": "#fbeceb", "grey": "#f2f3f4"}

fig, ax = plt.subplots(figsize=(13.6, 8.8), dpi=300)
ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")


def box(x0, x1, y0, y1, edge, fill, lw=1.4, dashed=False, r=1.6):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), x1 - x0, y1 - y0,
        boxstyle=f"round,pad=0,rounding_size={r}",
        linewidth=lw, edgecolor=edge, facecolor=fill,
        linestyle=(0, (4, 2)) if dashed else "solid", zorder=2))


def txt(x, y, s, size=8, weight="normal", color="#202124", ha="center", style="normal"):
    ax.text(x, y, s, ha=ha, va="center", fontsize=size, fontweight=weight,
            color=color, style=style, zorder=3, linespacing=1.45)


def elbow(x_from, y_from, x_to, y_to, color, dashed=False):
    mid = (y_from + y_to) / 2
    ls = (0, (4, 2)) if dashed else "solid"
    for a, b in (((x_from, y_from), (x_from, mid)),
                 ((x_from, mid), (x_to, mid)),
                 ((x_to, mid), (x_to, y_to))):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=1.2,
                linestyle=ls, zorder=1, solid_capstyle="round")


# ---------------------------------------------------------------- title
txt(50, 97.5, "What kind of problem is morphology $\\rightarrow$ mutation "
              "prediction for each gene?", size=13.5, weight="bold")
txt(50, 94.2, "Three regimes, read as the atoms of the partial information "
              "decomposition of the pattern and embedding channels",
    size=9, style="italic", color="#5f6368")
txt(50, 91.3, "Nested cross-validation, 444 patients, five seeds, "
              "patient-preserved folds. No regime assignment is adjudicated.",
    size=8, color="#80868b")

# ---------------------------------------------------------------- root
box(27, 73, 84.5, 89.5, "#3c4043", "#f1f3f4")
txt(50, 87.6, "Where is the discriminative signal $p$(slide | gene mutant)?",
    size=9.5, weight="bold")
txt(50, 85.6, "CTransPath embedding 512-d  +  6-pattern ANORAK simplex", size=8,
    color="#5f6368")

# ---------------------------------------------------------------- regimes
REG = [
    (2.5, 26.5, GREEN, "green", False, "Regime (i)   encoder saturation",
     "the embedding already carries what\nthe vocabulary names"),
    (30.5, 55.5, AMBER, "amber", True, "Regime (ii)   complementarity",
     "the pattern channel carries what\nthe embedding does not"),
    (59.5, 97.5, RED, "red", False, "Regime (iii)   out-of-ontology",
     "no exploitable footprint inside the simplex"),
]
for x0, x1, edge, fk, dash, head, sub in REG:
    box(x0, x1, 73.0, 81.5, edge, FILL[fk] if not dash else FILL["grey"], dashed=dash)
    txt((x0 + x1) / 2, 79.6, head, size=9, weight="bold",
        color=edge if not dash else GREY)
    txt((x0 + x1) / 2, 77.0, sub, size=7.4, color="#5f6368" if not dash else GREY)
    elbow((x0 + x1) / 2, 81.5, 50, 84.5, edge if not dash else GREY, dash)
    if dash:
        txt((x0 + x1) / 2, 74.5,
            "EMPTY  —  no sub-form populated", size=6.6, weight="bold",
            color=GREY)

# ---------------------------------------------------------------- sub-mechanisms
SUB = [
    (30.5, 42.0, GREY, "grey", True, "(ii-a) additive",
     "concat fusion suffices;\nthis is the cell glioma\n1p/19q occupies"),
    (44.0, 55.5, GREY, "grey", True, "(ii-b) non-additive",
     "synergy: FC > PI and\n$I_{jk}>0$ on a stratum"),
    (59.5, 71.5, RED, "red", False, "label-space\nmismatch",
     "encoder sees it,\nvocabulary cannot name it"),
    (73.0, 84.5, RED, "red", False, "null below\nthe regime",
     "invisible to the vocabulary\n$\\it{and}$ to the encoder"),
    (86.0, 97.5, RED, "red", False, "collapse\nto prior",
     "too few positives to\nestimate anything"),
]
for x0, x1, edge, fk, dash, head, sub in SUB:
    box(x0, x1, 60.0, 69.5, edge, FILL[fk], dashed=dash)
    txt((x0 + x1) / 2, 67.2, head, size=7.6, weight="bold", color=edge)
    txt((x0 + x1) / 2, 62.7, sub, size=6.6, color="#5f6368" if not dash else GREY)
    parent = 43 if x1 <= 56 else 78.5
    elbow((x0 + x1) / 2, 69.5, parent, 73.0, edge, dash)

# ---------------------------------------------------------------- genes
def gene_box(x0, x1, name, edge, fk, note, dashed=False, parent=None):
    g = GENES[name]
    box(x0, x1, 37.5, 56.5, edge, FILL[fk], dashed=dashed)
    cx = (x0 + x1) / 2
    txt(cx, 54.0, name, size=11, weight="bold",
        color=edge if not dashed else GREY, style="italic")
    txt(cx, 51.3, f"best {g['best']}", size=8, weight="bold",
        color="#202124" if not dashed else GREY)
    txt(cx, 49.3, f"$n_+$ = {g['npos']}", size=7, color="#5f6368")
    txt(cx, 45.9, f"B2 {g['b2']:.3f}   B3 {g['b3']:.3f}\n"
                  f"PI {g['pi']:.3f}   FC {g['fc']:.3f}",
        size=6.8, color="#5f6368")
    txt(cx, 40.6, note, size=6.4, style="italic",
        color=edge if not dashed else GREY)
    if parent is not None:
        elbow(cx, 56.5, parent, 60.0, edge, dashed)


gene_box(2.5, 13.5, "TP53", GREEN, "green", "directional,\nnot adjudicated")
gene_box(15.5, 26.5, "EGFR", GREEN, "green", "directional,\nnot adjudicated")
for cx in (8.0, 21.0):
    elbow(cx, 56.5, 14.5, 73.0, GREEN)

gene_box(44.0, 55.5, "KRAS", GREY, "grey",
         "conditional test on\nterciles: high$-$low\n"
         "$-0.024$ and $+0.025$",
         dashed=True, parent=49.75)
gene_box(59.5, 71.5, "STK11", RED, "red", "directional,\nnot adjudicated",
         parent=65.5)
gene_box(73.0, 84.5, "KEAP1", RED, "red", "all four conditions\n$\\leq$ 0.49, "
         "both protocols", parent=78.75)
gene_box(86.0, 97.5, "RBM10", RED, "red", "$n_+$ = 30 $<$ 32,\nthe estimation "
         "floor", parent=91.75)

# ---------------------------------------------------------------- verdict
txt(50, 33.0, "Cohort verdict   (Hosmer$-$Lemeshow gate, AUROC threshold 0.70)",
    size=9, weight="bold")

box(2.5, 30.0, 19.5, 30.0, GREY, "#fafafa", dashed=True)
txt(16.25, 27.6, "Conclusive   (AUROC $\\geq$ 0.70)", size=8.5, weight="bold",
    color=GREY)
txt(16.25, 24.4, "none", size=10, weight="bold", color=GREY, style="italic")
txt(16.25, 21.4, "the box was populated before the\ncohort was corrected",
    size=6.2, color=GREY, style="italic")

box(33.0, 97.5, 19.5, 30.0, "#3c4043", "#f1f3f4")
txt(65.25, 27.6, "Inconclusive   (AUROC $<$ 0.70)", size=8.5, weight="bold",
    color="#3c4043")
txt(65.25, 24.4, "TP53 (.661)    EGFR (.582)    STK11 (.608)    "
                 "RBM10 (.552)    KRAS (.517)    KEAP1 (.488)",
    size=8.2, color="#202124")
txt(65.25, 21.4, "all six targets, under both split protocols", size=6.8,
    color="#5f6368", style="italic")

# ---------------------------------------------------------------- footnote
txt(50, 14.0,
    "Values are pooled out-of-fold AUROC, mean over five seeds "
    "(tab_mutation_luad, patient-preserved panel). B2 = ABMIL on embeddings; "
    "B3 = ABMIL on patterns;\nPI = concat fusion; FC = 2-additive Choquet. "
    "The only contrasts that resolve are B3$-$B2 on TP53, EGFR and STK11; "
    "no fusion contrast resolves on any gene,\nand no equivalence is "
    "established at the $\\pm$0.02 margin the regime reads require, so every "
    "assignment above is directional.",
    size=6.9, color="#5f6368")

fig.savefig("figures/signal_tree.png", bbox_inches="tight", facecolor="white")
fig.savefig("figures/signal_tree.pdf", bbox_inches="tight", facecolor="white")
print("escrito figures/signal_tree.png y .pdf")
print(REGIME_SOURCE)
