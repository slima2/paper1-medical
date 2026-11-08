#!/usr/bin/env python3
"""Two defence slides turning the cohort exclusions into the control panel.

The cohort drops from 1,054 downloaded images to 458 analysed slides, and the
obvious way to present that is as attrition. It reads far better the other way:
each group removed for being the wrong kind of tissue is a group whose identity
the pipeline can be asked to recover, so the exclusions supply the positive
controls rather than costing sample size.

Produces a standalone deck to paste into the main presentation.

    python scripts/make_control_slides.py
"""

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN
from pptx.util import Emu, Inches, Pt

OUT = Path(__file__).resolve().parent.parent / "figures" / "slides_controls.pptx"

FOOTER = ("Pattern-Informed Fuzzy Deep Learning, Ph.D. Defense, "
          "Servio F. Lima Reina, University of Fribourg")

INK = RGBColor(0x1A, 0x1A, 0x2E)
MUTED = RGBColor(0x6B, 0x70, 0x80)
ACCENT = RGBColor(0xC2, 0x1B, 0x5B)      # the pink used for "this thesis"
GOOD = RGBColor(0x1B, 0x7F, 0x5A)
RULE = RGBColor(0xD8, 0xDC, 0xE3)
PANEL = RGBColor(0xF5, 0xF6, 0xF8)


def textbox(slide, x, y, w, h):
    tb = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = tf.margin_top = tf.margin_bottom = 0
    return tf


def para(tf, text, size, bold=False, color=INK, space_before=0, space_after=0,
         first=False, align=PP_ALIGN.LEFT):
    p = tf.paragraphs[0] if first else tf.add_paragraph()
    p.alignment = align
    p.space_before = Pt(space_before)
    p.space_after = Pt(space_after)
    run = p.add_run()
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = "Calibri"
    return p


def add_slide(prs, title, number):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    tf = textbox(slide, 0.6, 0.42, 12.1, 0.9)
    para(tf, title, 30, bold=True, first=True)

    line = slide.shapes.add_shape(1, Inches(0.6), Inches(1.28),
                                  Inches(12.1), Emu(9525))
    line.fill.solid()
    line.fill.fore_color.rgb = RULE
    line.line.fill.background()
    line.shadow.inherit = False

    ftf = textbox(slide, 0.6, 7.02, 12.1, 0.3)
    para(ftf, f"{FOOTER}  {number}", 9, color=MUTED, first=True)
    return slide


def panel(slide, x, y, w, h):
    box = slide.shapes.add_shape(1, Inches(x), Inches(y), Inches(w), Inches(h))
    box.fill.solid()
    box.fill.fore_color.rgb = PANEL
    box.line.color.rgb = RULE
    box.line.width = Pt(0.75)
    box.shadow.inherit = False
    return box


# ══════════════════════════════════════════════════════════════════════════════

def slide_one(prs):
    s = add_slide(prs, "The discarded slides are not attrition: they are the "
                       "controls", 1)

    para(textbox(s, 0.6, 1.45, 12.1, 0.4),
         "1,054 images downloaded, 458 analysed. What left the cohort, and "
         "why each exit is also an experiment.",
         14, color=MUTED, first=True)

    # ---- left: the cascade --------------------------------------------------
    panel(s, 0.6, 2.05, 5.5, 4.3)
    tf = textbox(s, 0.95, 2.35, 4.9, 3.8)
    para(tf, "THE CASCADE", 11, bold=True, color=MUTED, first=True)

    rows = [
        ("1,054", "H&E WSIs from the GDC LUAD project", INK, False),
        ("687", "pass quality control (Otsu, pen marker, blur)", INK, False),
        ("\u2212150", "patients the MAF assigns to squamous carcinoma",
         ACCENT, True),
        ("\u221217", "no mutation call for any of the six genes", MUTED, False),
        ("\u221262", "solid-tissue-normal sections, barcode 11A",
         ACCENT, True),
        ("458", "tumour slides, 444 patients, 33 sites", GOOD, True),
    ]
    for n, what, col, bold in rows:
        p = tf.add_paragraph()
        p.space_before = Pt(9)
        r = p.add_run()
        r.text = f"{n:>7}   "
        r.font.size = Pt(16)
        r.font.bold = True
        r.font.color.rgb = col
        r.font.name = "Consolas"
        r2 = p.add_run()
        r2.text = what
        r2.font.size = Pt(12.5)
        r2.font.bold = bold
        r2.font.color.rgb = INK if bold else MUTED
        r2.font.name = "Calibri"

    # ---- right: the reframe -------------------------------------------------
    panel(s, 6.5, 2.05, 6.2, 4.3)
    tf = textbox(s, 6.85, 2.35, 5.6, 3.8)
    para(tf, "WHAT EACH EXIT BECOMES", 11, bold=True, color=MUTED, first=True)

    para(tf, "62 normal sections  \u2192  tumour vs normal", 15, bold=True,
         color=ACCENT, space_before=14)
    para(tf, "They carry the patient's germline barcode, so they inherit a "
             "mutation label while containing no tumour. Kept in, a model "
             "scores any gene by detecting tumour alone.",
         12, color=MUTED, space_before=3)

    para(tf, "150 squamous slides  \u2192  LUAD vs LUSC", 15, bold=True,
         color=ACCENT, space_before=14)
    para(tf, "The six ANORAK patterns are adenocarcinoma growth patterns. "
             "Squamous tissue has none of them, so the vocabulary channel "
             "becomes noise by construction on those slides.",
         12, color=MUTED, space_before=3)

    para(tf, "Every group removed for being the wrong kind of tissue is a "
             "group whose identity the pipeline can be asked to recover.",
         13, bold=True, color=INK, space_before=18)


def slide_two(prs):
    s = add_slide(prs, "Four controls: the pipeline works before it is asked "
                       "to be clever", 2)

    para(textbox(s, 0.6, 1.45, 12.1, 0.4),
         "Two positives say the machinery sees tissue. Two negatives say it "
         "is not seeing something else.",
         14, color=MUTED, first=True)

    header = ["Control", "What it asks", "AUROC", "What it rules out"]
    widths = [2.75, 4.05, 1.5, 3.8]
    rows = [
        ("Tumour vs normal", "is there tumour on this slide at all",
         "0.976", "a broken pipeline", GOOD),
        ("LUAD vs LUSC", "does the encoder see subtype morphology",
         "pending", "an encoder blind to histology", MUTED),
        ("Shuffled labels", "can it score a label with no signal",
         "chance", "leakage through the folds", GOOD),
        ("Patient sex", "is a demographic axis doing the work",
         "chance", "a confound standing in for genotype", GOOD),
    ]

    x0, y0, rh = 0.6, 2.1, 0.86
    # header
    hb = s.shapes.add_shape(1, Inches(x0), Inches(y0), Inches(sum(widths)),
                            Inches(0.44))
    hb.fill.solid()
    hb.fill.fore_color.rgb = PANEL
    hb.line.fill.background()
    hb.shadow.inherit = False
    x = x0
    for label, w in zip(header, widths):
        tf = textbox(s, x + 0.14, y0 + 0.1, w - 0.2, 0.3)
        para(tf, label.upper(), 10.5, bold=True, color=MUTED, first=True)
        x += w

    for i, (name, asks, val, rules, col) in enumerate(rows):
        y = y0 + 0.44 + i * rh
        sep = s.shapes.add_shape(1, Inches(x0), Inches(y), Inches(sum(widths)),
                                 Emu(9525))
        sep.fill.solid()
        sep.fill.fore_color.rgb = RULE
        sep.line.fill.background()
        sep.shadow.inherit = False

        x = x0
        tf = textbox(s, x + 0.14, y + 0.17, widths[0] - 0.2, 0.6)
        para(tf, name, 14, bold=True, first=True)
        x += widths[0]

        tf = textbox(s, x + 0.14, y + 0.2, widths[1] - 0.2, 0.6)
        para(tf, asks, 12, color=MUTED, first=True)
        x += widths[1]

        tf = textbox(s, x + 0.14, y + 0.14, widths[2] - 0.2, 0.6)
        para(tf, val, 18, bold=True, color=col, first=True)
        x += widths[2]

        tf = textbox(s, x + 0.14, y + 0.2, widths[3] - 0.2, 0.6)
        para(tf, rules, 12, color=MUTED, first=True)

    y = y0 + 0.44 + len(rows) * rh + 0.22
    panel(s, 0.6, y, 12.1, 0.92)
    tf = textbox(s, 0.95, y + 0.16, 11.5, 0.7)
    para(tf, "So a mutation AUROC of 0.66 is a statement about the "
             "genotype\u2013phenotype map, not a broken pipeline.",
         15, bold=True, first=True)
    para(tf, "The controls are what license reading a weak result as a "
             "property of the biology. Without them, every null is "
             "indistinguishable from a bug.",
         12, color=MUTED, space_before=3)


def main():
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)
    slide_one(prs)
    slide_two(prs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(OUT)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
