#!/usr/bin/env python3
"""Compose the LCHAI v2 six-level explanation-stack figure from live-UI screenshots.

Panels are cropped from viewport captures of https://lchai.gptfy.biz (case
TCGA-55-7815, KRAS), using exact getBoundingClientRect() coordinates scaled by
the browser devicePixelRatio, then arranged into a single labelled composite.

Inputs (in this folder):
  lchai_ui_src_row_attention.png  -> verdict banner, L2 pattern overlay, L1 attention
  lchai_ui_src_row_shap.png       -> L3 SHAP, L5 ablation
  lchai_ui_src_row_choquet.png    -> L6 explanation, L4 Choquet
  lchai_ui_src_graph.png          -> NCIt+MONDO knowledge graph

Output:
  lchai_ui_six_levels.png
"""
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
DPR = 0.8375000357627869  # window.devicePixelRatio at capture time

SRC = {
    "A": HERE / "lchai_ui_src_row_attention.png",
    "B": HERE / "lchai_ui_src_row_shap.png",
    "C": HERE / "lchai_ui_src_row_choquet.png",
    "G": HERE / "lchai_ui_src_graph.png",
}


def font(size, bold=False):
    name = "arialbd.ttf" if bold else "arial.ttf"
    try:
        return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)
    except OSError:
        return ImageFont.load_default()


def css_crop(img, x0, y0, x1, y1, pad=6):
    """Crop using CSS-pixel coordinates scaled to device pixels."""
    w, h = img.size
    bx0 = max(0, int(round(x0 * DPR)) - pad)
    by0 = max(0, int(round(y0 * DPR)) - pad)
    bx1 = min(w, int(round(x1 * DPR)) + pad)
    by1 = min(h, int(round(y1 * DPR)) + pad)
    return img.crop((bx0, by0, bx1, by1))


def dev_crop(img, x0, y0, x1, y1):
    w, h = img.size
    return img.crop((max(0, x0), max(0, y0), min(w, x1), min(h, y1)))


def resize_w(img, target_w):
    if img.width == target_w:
        return img
    r = target_w / img.width
    return img.resize((target_w, max(1, int(round(img.height * r)))), Image.LANCZOS)


LEVEL_COLOR = {
    "L1": (37, 99, 235),
    "L2": (5, 150, 105),
    "L3": (217, 70, 239),
    "L4": (234, 88, 12),
    "L5": (124, 58, 237),
    "L6": (13, 148, 136),
    "KG": (71, 85, 105),
}


def badge(canvas, xy, level, label):
    d = ImageDraw.Draw(canvas)
    fb = font(22, bold=True)
    fl = font(20, bold=True)
    tag = level
    tw = d.textbbox((0, 0), tag, font=fb)[2]
    pad = 10
    bh = 34
    bw = tw + 2 * pad
    x, y = xy
    col = LEVEL_COLOR[level]
    d.rounded_rectangle([x, y, x + bw, y + bh], radius=8, fill=col)
    d.text((x + pad, y + 5), tag, font=fb, fill=(255, 255, 255))
    # descriptor pill
    lw = d.textbbox((0, 0), label, font=fl)[2]
    lx = x + bw + 8
    d.rounded_rectangle([lx, y, lx + lw + 2 * pad, y + bh], radius=8,
                        fill=(255, 255, 255), outline=col, width=2)
    d.text((lx + pad, y + 6), label, font=fl, fill=(31, 41, 55))


def framed(img):
    """Add a thin gray border around a panel crop."""
    bordered = Image.new("RGB", (img.width + 2, img.height + 2), (203, 213, 225))
    bordered.paste(img.convert("RGB"), (1, 1))
    return bordered


def main():
    A = Image.open(SRC["A"]).convert("RGB")
    B = Image.open(SRC["B"]).convert("RGB")
    C = Image.open(SRC["C"]).convert("RGB")
    G = Image.open(SRC["G"]).convert("RGB")

    # --- crops (CSS coords from getBoundingClientRect) ---
    header = css_crop(A, 12, 150, 1490, 340)          # gene row + verdict + warning
    L2 = css_crop(A, 24, 356, 744, 676)               # pattern overlay
    L1 = css_crop(A, 760, 356, 1480, 676)             # attention
    L3 = css_crop(B, 24, 359, 744, 672)               # SHAP
    L5 = css_crop(B, 760, 359, 1480, 672)             # ablation
    L6 = css_crop(C, 24, 123, 1479, 349)              # LLM explanation + refs
    L4 = css_crop(C, 24, 369, 1479, 676)              # Choquet
    KG = dev_crop(G, 8, 110, 1247, 726)               # knowledge graph (device px; drop right scrollbar)

    # --- layout ---
    MARGIN = 46
    GAP = 22
    FULL_W = 1440
    HALF_W = (FULL_W - GAP) // 2

    header_i = framed(resize_w(header, FULL_W))
    L1_i = framed(resize_w(L1, HALF_W))
    L2_i = framed(resize_w(L2, HALF_W))
    L3_i = framed(resize_w(L3, HALF_W))
    L5_i = framed(resize_w(L5, HALF_W))
    L4_i = framed(resize_w(L4, FULL_W))
    L6_i = framed(resize_w(L6, FULL_W))
    KG_i = framed(resize_w(KG, FULL_W))

    title_h = 92
    row1_h = max(L1_i.height, L2_i.height)
    row2_h = max(L3_i.height, L5_i.height)
    total_h = (MARGIN + title_h + header_i.height + GAP + row1_h + GAP + row2_h
               + GAP + L4_i.height + GAP + L6_i.height + GAP + KG_i.height + MARGIN)
    total_w = MARGIN * 2 + FULL_W

    canvas = Image.new("RGB", (total_w, total_h), (255, 255, 255))
    d = ImageDraw.Draw(canvas)

    # title
    d.text((MARGIN, MARGIN - 6), "The six-level explanation stack as the pathologist sees it",
           font=font(34, bold=True), fill=(15, 23, 42))
    d.text((MARGIN, MARGIN + 40),
           "LCHAI v2 live interface  |  KRAS, TCGA-55-7815  |  verdict Inconclusive (AUROC 0.609), yet P(mut)=96.5% on this slide",
           font=font(21), fill=(71, 85, 105))

    y = MARGIN + title_h
    x = MARGIN

    canvas.paste(header_i, (x, y))
    y += header_i.height + GAP

    # row 1: L1 | L2
    canvas.paste(L1_i, (x, y))
    canvas.paste(L2_i, (x + HALF_W + GAP, y))
    badge(canvas, (x + 8, y + 8), "L1", "Attention (where)")
    badge(canvas, (x + HALF_W + GAP + 8, y + 8), "L2", "Patterns (what)")
    y += row1_h + GAP

    # row 2: L3 | L5
    canvas.paste(L3_i, (x, y))
    canvas.paste(L5_i, (x + HALF_W + GAP, y))
    badge(canvas, (x + 8, y + 8), "L3", "SHAP channels")
    badge(canvas, (x + HALF_W + GAP + 8, y + 8), "L5", "Channel ablation")
    y += row2_h + GAP

    # L4 full
    canvas.paste(L4_i, (x, y))
    badge(canvas, (x + 8, y + 8), "L4", "Choquet-Shapley + interaction indices")
    y += L4_i.height + GAP

    # L6 full
    canvas.paste(L6_i, (x, y))
    badge(canvas, (x + 8, y + 8), "L6", "Fuzzy labels + LLM explanation (KG-grounded)")
    y += L6_i.height + GAP

    # KG full
    canvas.paste(KG_i, (x, y))
    badge(canvas, (x + 8, y + 8), "KG", "NCIt+MONDO knowledge graph")

    out = HERE / "lchai_ui_six_levels.png"
    canvas.save(out, "PNG")
    print("wrote", out, canvas.size)


if __name__ == "__main__":
    main()
