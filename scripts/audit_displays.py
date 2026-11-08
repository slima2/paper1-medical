"""List the display items of the Article body and of the Supplementary
Information, in order.

Communications Medicine allows up to ten display items in an Article and has
no Extended Data tier, so the budget has to be read off the source rather
than remembered.
"""

from __future__ import annotations

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent

BODY = ["02_introduction_CM.tex", "03_results_CM.tex",
        "04_discussion_CM.tex"]

BEGIN = re.compile(r"\\begin\{(figure|table)\}")
LABEL = re.compile(r"\\label\{([^}]*)\}")
INPUT = re.compile(r"\\input\{tables/([^}]*)\}")
GFX = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}")


def scan(rel: str, counter: int, tag: str) -> int:
    path = ROOT / "sections" / rel
    text = path.read_text(encoding="utf-8", errors="ignore")
    for m in BEGIN.finditer(text):
        seg = text[m.start():m.start() + 4000]
        lab = LABEL.search(seg)
        src = INPUT.search(seg) or GFX.search(seg)
        counter += 1
        kind = m.group(1)
        name = src.group(1) if src else "?"
        label = lab.group(1) if lab else "?"
        print(f"  {tag}{counter:<2d} {kind:6s} {name:30s} {label}")
    return counter


if __name__ == "__main__":
    print("=== Article body (limite Communications Medicine: 10) ===")
    n = 0
    for f in BODY:
        n = scan(f, n, "")
    print(f"  total: {n}\n")

    print("=== Supplementary Information (sin limite) ===")
    m = scan("09_supplementary_CM.tex", 0, "S")
    print(f"  total: {m}")
