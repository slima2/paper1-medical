"""Cross-domain pipeline figures: the SAME four-condition protocol, two modalities.

Both diagrams reuse the exact visual grammar of two_channel_architecture.py
(blue = embedding/encoder channel E, amber = source-ontology/pattern channel P)
so the analogy to the TCGA-LUAD architecture is immediate: an unconstrained
learned encoder E, an expert source vocabulary P, and the four conditions
B2 (encoder only), B3 (ontology only), PI (concat fusion) and FC (2-additive
fuzzy Choquet over P) -- identical to LUAD, only E and P change with modality.

Outputs (in this folder):
    pipeline_ecg_domainC.png     ECG -> sex / age  (PTB-XL, run)
    pipeline_glioma_domainB.png  MRI -> IDH / 1p19q / MGMT (VASARI, pre-registered)
"""
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))

EMB_FILL, EMB_EDGE = "#dbeafe", "#1e3a8a"   # channel E (embeddings / encoder)
PAT_FILL, PAT_EDGE = "#fef3c7", "#a16207"   # channel P (ontology / patterns)
MIX_FILL, MIX_EDGE = "#e2e8f0", "#334155"   # fusion / heads
IN_FILL, IN_EDGE = "#f1f5f9", "#475569"     # modality input
TGT_FILL, TGT_EDGE = "#dcfce7", "#15803d"   # targets / verdict
GREY = "#64748b"


def box(ax, x, y, w, h, txt, fc, ec, fs=10.5, fw="normal", tc="#0f172a"):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                facecolor=fc, edgecolor=ec, linewidth=1.4, zorder=3))
    ax.text(x, y, txt, ha="center", va="center", fontsize=fs,
            fontweight=fw, color=tc, zorder=4, linespacing=1.3)


def arrow(ax, x1, y1, x2, y2, color=GREY, lw=1.4, style="->"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=2,
                arrowprops=dict(arrowstyle=style, color=color, lw=lw,
                                connectionstyle="arc3,rad=0.0"))


def draw(cfg, out_name):
    fig, ax = plt.subplots(figsize=(14, 7.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # ---- title + shared-protocol banner --------------------------------
    ax.text(7.0, 7.72, cfg["title"], ha="center", fontsize=15,
            fontweight="bold", color="#1e293b")
    ax.text(7.0, 7.28,
            "Identical four-condition protocol to TCGA-LUAD — only the encoder "
            "$\\mathbf{E}$ and the source vocabulary $\\mathbf{P}$ change",
            ha="center", fontsize=10.5, style="italic", color="#475569")

    yE, yP = 5.85, 3.35

    # ---- modality input -------------------------------------------------
    box(ax, 1.35, (yE + yP) / 2, 2.1, 3.1, cfg["input"], IN_FILL, IN_EDGE,
        fs=10.5, fw="bold")

    # ---- channel renderers ---------------------------------------------
    ax.text(4.35, 6.72, "encoder channel $\\mathbf{E}$", ha="center",
            fontsize=10.5, color=EMB_EDGE, fontweight="bold")
    ax.text(4.35, 2.28, "source ontology channel $\\mathbf{P}$", ha="center",
            fontsize=10.5, color=PAT_EDGE, fontweight="bold")
    box(ax, 4.35, yE, 2.5, 1.05, cfg["encoder"], EMB_FILL, EMB_EDGE)
    box(ax, 4.35, yP, 2.5, 1.05, cfg["ontology"], PAT_FILL, PAT_EDGE)
    ax.text(4.35, yE - 0.98, cfg["enc_luad"], ha="center", fontsize=8.6,
            style="italic", color=EMB_EDGE)
    ax.text(4.35, yP + 0.98, cfg["ont_luad"], ha="center", fontsize=8.6,
            style="italic", color=PAT_EDGE)
    arrow(ax, 2.42, yE, 3.08, yE, EMB_EDGE)
    arrow(ax, 2.42, yP, 3.08, yP, PAT_EDGE)

    # ---- channel outputs (vectors) -------------------------------------
    box(ax, 6.95, yE, 1.55, 0.9, cfg["z"], EMB_FILL, EMB_EDGE, fw="bold")
    box(ax, 6.95, yP, 1.55, 0.9, cfg["p"], PAT_FILL, PAT_EDGE, fw="bold")
    arrow(ax, 5.62, yE, 6.16, yE, EMB_EDGE)
    arrow(ax, 5.62, yP, 6.16, yP, PAT_EDGE)

    # ---- four-condition panel ------------------------------------------
    px0, px1 = 8.35, 11.55
    ax.add_patch(FancyBboxPatch((px0, 2.15), px1 - px0, 4.55,
                                boxstyle="round,pad=0.02,rounding_size=0.08",
                                facecolor="#ffffff", edgecolor="#94a3b8",
                                linewidth=1.5, zorder=2))
    ax.text((px0 + px1) / 2, 6.35, "four conditions", ha="center",
            fontsize=10.5, fontweight="bold", color="#334155")
    rows = [
        ("B2", "encoder only", r"$[\,\mathbf{z}\,]$", EMB_FILL, EMB_EDGE),
        ("B3", "ontology only", r"$[\,\mathbf{P}\,]$", PAT_FILL, PAT_EDGE),
        ("PI", "concat fusion", r"$[\,\mathbf{z}\;\Vert\;\mathbf{P}\,]$", MIX_FILL, MIX_EDGE),
        ("FC", "fuzzy Choquet", r"$\mathbf{z}\;\oplus\;\int\!\mathbf{P}\,d\mu$", MIX_FILL, MIX_EDGE),
    ]
    ry = [5.75, 4.95, 4.15, 3.35]
    for (tag, name, formula, fc, ec), y in zip(rows, ry):
        box(ax, (px0 + px1) / 2, y, px1 - px0 - 0.4, 0.66,
            f"$\\mathbf{{{tag}}}$  {name}   {formula}", fc, ec, fs=9.8)
    # inputs feeding the panel
    arrow(ax, 7.73, yE, px0 - 0.02, 5.55, EMB_EDGE)
    arrow(ax, 7.73, yP, px0 - 0.02, 3.55, PAT_EDGE)
    ax.text((px0 + px1) / 2, 2.55, cfg["fc_note"], ha="center", fontsize=8.4,
            style="italic", color="#64748b")

    # ---- targets / verdict ---------------------------------------------
    ax.text(12.95, 6.72, cfg["target_header"], ha="center", fontsize=10,
            fontweight="bold", color=TGT_EDGE)
    n = len(cfg["targets"])
    top, bot = 6.15, 2.7
    step = (top - bot) / max(n - 1, 1) if n > 1 else 0
    for i, t in enumerate(cfg["targets"]):
        ty = top - i * step if n > 1 else (top + bot) / 2
        box(ax, 12.95, ty, 2.0, 0.95, t, TGT_FILL, TGT_EDGE, fs=9.2)
        arrow(ax, px1 + 0.02, 4.4, 11.95, ty, GREY, lw=1.0)

    # ---- footer --------------------------------------------------------
    ax.text(7.0, 0.9, cfg["footer"], ha="center", fontsize=9.6,
            style="italic", color="#475569", linespacing=1.4)

    out = os.path.join(HERE, out_name)
    fig.savefig(out, dpi=260, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out}")


ECG = {
    "title": "Domain C — electrocardiogram $\\rightarrow$ physiological state (PTB-XL)",
    "input": "12-lead ECG\nraw signal\n(100 Hz)",
    "encoder": "Encoder $\\mathbf{E}$\n1D-CNN over\nthe raw signal",
    "enc_luad": "(as CTransPath\nembeddings in LUAD)",
    "ontology": "SCP-ECG statements\n71 graded, supplied\nper record",
    "ont_luad": "(as ANORAK growth\npatterns in LUAD)",
    "z": "$\\mathbf{z}$\n128-d",
    "p": "$\\mathbf{P}$\n71-d",
    "fc_note": "$\\mathbf{z}$ feeds B2, PI, FC   ·   $\\mathbf{P}$ feeds B3, PI, FC",
    "target_header": "target $\\rightarrow$ verdict",
    "targets": [
        "Sex\nAUROC 0.90 · Conclusive\n(i) saturation",
        "Age $\\geq$ 65\nAUROC 0.84 · Conclusive\n(i) saturation",
    ],
    "footer": ("The four-condition protocol and the Conclusive verdict transfer "
               "intact to a new modality and ontology.\nFusion (PI, FC) matches "
               "the encoder to within a fold's SD while B3 trails — the "
               "encoder-saturated signature of regime (i)."),
}

GLIOMA = {
    "title": "Domain B — multiparametric MRI $\\rightarrow$ glioma genotype (VASARI, pre-registered)",
    "input": "Multiparametric MRI\nT1 · T1c · T2 · FLAIR\n(BraTS 2021)",
    "encoder": "Encoder $\\mathbf{E}$\n2.5D CNN + ABMIL\nover axial slices",
    "enc_luad": "(as CTransPath +\nABMIL tiles in LUAD)",
    "ontology": "VASARI features\n$\\sim$30 semantic MR\n(seed $\\rightarrow$ BraTS)",
    "ont_luad": "(as ANORAK growth\npatterns in LUAD)",
    "z": "$\\mathbf{z}$\n128-d",
    "p": "$\\mathbf{P}$\n$\\sim$30-d",
    "fc_note": "$\\mathbf{z}$ feeds B2, PI, FC   ·   $\\mathbf{P}$ feeds B3, PI, FC",
    "target_header": "target $\\rightarrow$ predicted regime",
    "targets": [
        "IDH\n(i) saturation\nAUROC $\\approx$ 0.88",
        "1p/19q\n(ii) synergy\nFC/PI $>$ B2",
        "MGMT\n(iii) out-of-ontology\nAUROC $\\approx$ 0.50–0.56",
    ],
    "footer": ("Pre-registered falsification test: the non-additive FC "
               "aggregator should beat the additive PI on 1p/19q only, not on "
               "IDH or MGMT.\nMGMT is the make-or-break claim — an empty "
               "VASARI-side atom (out-of-ontology), distinguished from a data "
               "floor by the B3-vs-B2 signature."),
}

if __name__ == "__main__":
    draw(ECG, "pipeline_ecg_domainC.png")
    draw(GLIOMA, "pipeline_glioma_domainB.png")
