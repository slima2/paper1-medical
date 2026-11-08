"""Mechanical extraction of every numeric claim in the Communications Medicine manuscript.

The point is to make cross-section comparison depend on the text rather than
on anyone's memory of it. We strip LaTeX comments (which contain stale drafting
notes and would produce false hits), normalise the numeric spellings the
manuscript uses interchangeably (4{,}400 vs 4400, 0.716 vs .716), and index
every occurrence by file and line with its surrounding words.

Usage:
    python scripts/audit_consistency.py            # group numbers by value
    python scripts/audit_consistency.py <regex>    # show context for a quantity
"""

from __future__ import annotations

import collections
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent

FILES = [
    "sections/01_abstract_CM.tex",
    "sections/02_introduction_CM.tex",
    "sections/03_results_CM.tex",
    "sections/04_discussion_CM.tex",
    "sections/05_methods_CM.tex",
    "sections/06_declarations_CM.tex",
    "sections/09_supplementary_CM.tex",
] + sorted(str(p.relative_to(ROOT)).replace("\\", "/")
           for p in (ROOT / "tables").glob("*.tex"))

SHORT = {
    "01_abstract_CM.tex": "ABS",
    "02_introduction_CM.tex": "INTRO",
    "03_results_CM.tex": "RESULTS",
    "04_discussion_CM.tex": "DISC",
    "05_methods_CM.tex": "METHODS",
    "06_declarations_CM.tex": "AVAIL",
    "09_supplementary_CM.tex": "SUPP",
}

# A number as the manuscript writes it: optional leading digits with LaTeX
# thousands separators, optional decimal part. Percent and math markup are
# handled by the normaliser rather than the pattern.
NUM = re.compile(r"\d[\d,]*(?:\{,\})?[\d,]*(?:\.\d+)?")


def strip_comments(text: str) -> str:
    """Remove LaTeX comments but keep escaped percent signs."""
    out = []
    for line in text.split("\n"):
        buf, i = [], 0
        while i < len(line):
            c = line[i]
            if c == "\\" and i + 1 < len(line):
                buf.append(line[i:i + 2])
                i += 2
                continue
            if c == "%":
                break
            buf.append(c)
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def normalise(tok: str) -> str:
    """4{,}400 and 4,400 and 4400 all become 4400."""
    return tok.replace("{,}", "").replace(",", "")


def clean_context(line: str) -> str:
    line = re.sub(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{[^}]*\}", "", line)
    line = re.sub(r"\\(cref|ref|label)\{[^}]*\}", "", line)
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def load() -> list[tuple[str, int, str]]:
    """Return (short_name, lineno, cleaned_line) for every non-empty line."""
    rows = []
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        name = pathlib.Path(rel).name
        short = SHORT.get(name, name.replace(".tex", ""))
        text = strip_comments(p.read_text(encoding="utf-8", errors="ignore"))
        for i, line in enumerate(text.split("\n"), 1):
            c = clean_context(line)
            if c:
                rows.append((short, i, c))
    return rows


def group_numbers(rows) -> None:
    idx = collections.defaultdict(list)
    for short, lineno, line in rows:
        for m in NUM.finditer(line):
            idx[normalise(m.group())].append((short, lineno, line))

    def sortkey(v: str) -> tuple:
        try:
            return (0, float(v))
        except ValueError:
            return (1, 0.0)

    print(f"{'valor':>10s}  {'veces':>5s}  secciones donde aparece")
    print("-" * 78)
    for val in sorted(idx, key=sortkey):
        hits = idx[val]
        secs = collections.Counter(h[0] for h in hits)
        # Numbers appearing in two or more DIFFERENT sections are the ones that
        # must agree; single-section numbers cannot contradict anything.
        if len(secs) < 2:
            continue
        order = ["ABS", "INTRO", "RESULTS", "DISC", "METHODS", "AVAIL",
                 "EXTDATA", "SUPP"]
        keyed = sorted(secs.items(),
                       key=lambda kv: (order.index(kv[0]) if kv[0] in order
                                       else 99, kv[0]))
        desc = ", ".join(f"{s}x{n}" if n > 1 else s for s, n in keyed)
        print(f"{val:>10s}  {len(hits):5d}  {desc}")


def show(pattern: str, rows) -> None:
    rx = re.compile(pattern, re.I)
    for short, lineno, line in rows:
        if rx.search(line):
            print(f"[{short:8s} L{lineno:>4d}] {line}")


if __name__ == "__main__":
    data = load()
    if len(sys.argv) > 1:
        show(sys.argv[1], data)
    else:
        group_numbers(data)
