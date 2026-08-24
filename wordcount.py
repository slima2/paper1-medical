"""Approximate Nature-style word count per section, checked against npj
Digital Medicine Article limits.

Counts main-text words only: comments, float environments (figures/tables and
their legends), and LaTeX markup are stripped. npj Digital Medicine sets no
main-text limit; the binding constraints are the 150-word abstract and the
60-reference guideline.
"""

import glob
import os
import re

FLOATS = re.compile(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", re.S)
ARGLESS = re.compile(r"\\(cite\w*|ref|cref|Cref|label|input|includegraphics|url|href)\s*(\[[^\]]*\])?\{[^}]*\}")


def words(path):
    s = open(path, encoding="utf-8").read()
    s = re.sub(r"(?m)^\s*%.*$", "", s)
    s = FLOATS.sub("", s)
    s = ARGLESS.sub(" ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$&~\\]", " ", s)
    return len([w for w in s.split() if any(c.isalnum() for c in w)])


def abstract_words(path):
    """Count only the abstract body; the keywords list sits outside it."""
    s = open(path, encoding="utf-8").read()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    if not m:
        return 0
    body = re.sub(r"(?m)^\s*%.*$", "", m.group(1))
    body = ARGLESS.sub(" ", body)
    body = re.sub(r"\\[a-zA-Z]+\*?", " ", body)
    body = re.sub(r"[{}$&~\\]", " ", body)
    return len([w for w in body.split() if any(c.isalnum() for c in w)])


def floats_and_cites():
    figs = tables = 0
    cites = set()
    for f in glob.glob("sections/*.tex"):
        s = open(f, encoding="utf-8").read()
        figs += len(re.findall(r"\\begin\{figure\*?\}", s))
        tables += len(re.findall(r"\\begin\{table\*?\}", s))
        for m in re.findall(r"\\cite\w*\{([^}]*)\}", s):
            cites.update(x.strip() for x in m.split(","))
    return figs, tables, cites


main_files = ["01_abstract", "02_introduction", "03_results", "04_discussion"]
total_main = 0
for f in sorted(glob.glob("sections/*.tex")):
    n = words(f)
    base = os.path.basename(f).replace(".tex", "")
    tag = "MAIN" if base in main_files else "    "
    if base in main_files and base != "01_abstract":
        total_main += n
    print(f"{tag} {base:26s} {n:6d}")

figs, tables, cites = floats_and_cites()
print()
print(f"Main text (intro+results+discussion): {total_main}   npj: no limit")
print(f"Abstract body (excl. keywords):       {abstract_words('sections/01_abstract.tex')}   limit 150")
print(f"Display items: {figs} figures + {tables} tables = {figs + tables}   npj: no limit")
print(f"Unique references cited:              {len(cites)}   guideline 60, not strictly enforced")
