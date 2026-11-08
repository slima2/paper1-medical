"""Word, display-item and reference count for the *Scientific Reports*
build (main_SR.tex), checked against the journal's Article guidance
(https://www.nature.com/srep/author-instructions/submission-guidelines):

  title            <= 20 words;
  main text        <= 4,500 words, Methods not counted (abstract, references
                   and figure legends excluded here as well);
  abstract         <= 200 words, unreferenced;
  display items    up to 8 figures and tables combined;
  references       ~60 (not strictly enforced);
  footnotes        none.

With --anon it also sweeps for identifying strings, for the double-anonymised
variant of the build (\anontrue in main_SR.tex).

Run from the directory containing sections/ and tables/.
"""

import glob
import os
import re
import sys

LIMITS = {"title": 20, "main": 4500, "abstract_max": 200, "displays": 8,
          "refs": 60}

FLOATS = re.compile(r"\\begin\{(figure|table)\*?\}.*?\\end\{\1\*?\}", re.S)
ARGLESS = re.compile(
    r"\\(cite\w*|ref|cref|Cref|label|input|includegraphics|url|href)"
    r"\s*(\[[^\]]*\])?\{[^}]*\}"
)


def _strip(s):
    s = ARGLESS.sub(" ", s)
    s = re.sub(r"\\[a-zA-Z]+\*?", " ", s)
    s = re.sub(r"[{}$&~\\]", " ", s)
    return len([w for w in s.split() if any(c.isalnum() for c in w)])


def _nocomment(s):
    return re.sub(r"(?m)^\s*%.*$", "", s)


def words(path):
    s = _nocomment(open(path, encoding="utf-8").read())
    return _strip(FLOATS.sub("", s))


def abstract_words(path):
    s = open(path, encoding="utf-8").read()
    m = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", s, re.S)
    return _strip(_nocomment(m.group(1))) if m else 0


def title_words():
    s = open("main_SR.tex", encoding="utf-8").read()
    m = re.search(r"\\newcommand\{\\SRpdftitle\}\{([^}]*)\}", s)
    return len(m.group(1).split()) if m else 0


def verdict(actual, limit):
    return "OK" if actual <= limit else f"OVER by {actual - limit}"


MAIN = ["02_introduction_SR", "03_results_SR", "04_discussion_SR"]
SR_BUILD = MAIN + ["01_abstract_SR", "05_methods_SR", "06_declarations_SR",
                   "07_backmatter_SR", "09_supplementary_SR"]

total = 0
print(f"{'section':<26}{'words':>8}")
for f in sorted(glob.glob("sections/*.tex")):
    base = os.path.basename(f).replace(".tex", "")
    if base not in SR_BUILD:
        continue
    n = words(f)
    tag = "MAIN" if base in MAIN else "    "
    if base in MAIN:
        total += n
    print(f"{tag} {base:<21}{n:>8}")

figs = tables = 0
cites = set()
footnotes = 0
for base in SR_BUILD:
    f = os.path.join("sections", base + ".tex")
    if not os.path.exists(f):
        continue
    s = _nocomment(open(f, encoding="utf-8").read())
    footnotes += len(re.findall(r"\\footnote\{", s))
    if base == "09_supplementary_SR":
        continue
    figs += len(re.findall(r"\\begin\{figure\*?\}", s))
    tables += len(re.findall(r"\\begin\{table\*?\}", s))


def collect_cites(text, into):
    for m in re.findall(r"\\cite\w*\{([^}]*)\}", text):
        into.update(x.strip() for x in m.split(","))


inputs = set()
for base in SR_BUILD:
    f = os.path.join("sections", base + ".tex")
    if not os.path.exists(f):
        continue
    s = _nocomment(open(f, encoding="utf-8").read())
    if base != "09_supplementary_SR":
        collect_cites(s, cites)
    inputs.update(re.findall(r"\\input\{([^}]*)\}", s))

for rel in sorted(inputs):
    p = rel if rel.endswith(".tex") else rel + ".tex"
    if os.path.exists(p):
        footnotes += len(re.findall(r"\\footnote\{",
                                    _nocomment(open(p, encoding="utf-8").read())))
        collect_cites(_nocomment(open(p, encoding="utf-8").read()), cites)

print()
t = title_words()
print(f"Title words:                          {t:>6}   "
      f"limit {LIMITS['title']}   {verdict(t, LIMITS['title'])}")
print(f"Main text (intro+results+discussion): {total:>6}   "
      f"limit {LIMITS['main']}   {verdict(total, LIMITS['main'])}")
n = abstract_words("sections/01_abstract_SR.tex")
print(f"Abstract body:                        {n:>6}   "
      f"limit {LIMITS['abstract_max']}   {verdict(n, LIMITS['abstract_max'])}")
d = figs + tables
print(f"Display items: {figs} figures + {tables} tables = {d:>6}   "
      f"limit {LIMITS['displays']}   {verdict(d, LIMITS['displays'])}")

sup = "sections/09_supplementary_SR.tex"
if os.path.exists(sup):
    s = open(sup, encoding="utf-8").read()
    n_sf = len(re.findall(r"(?m)^\\begin\{figure\*?\}", s))
    n_st = len(re.findall(r"(?m)^\\begin\{table\*?\}", s))
    print(f"Supplementary items: {n_sf} figures + {n_st} tables")

print(f"Unique references cited (article):    {len(cites):>6}   "
      f"guide {LIMITS['refs']}   {verdict(len(cites), LIMITS['refs'])}")
print(f"Footnotes:                            {footnotes:>6}   "
      f"limit 0   {verdict(footnotes, 0)}")

if "--anon" in sys.argv:
    IDENT = re.compile(
        r"GPTfy|gptfy|slima|HSLU|Lucerne|Guayaquil|SOLCA|CISOL|Lima Reina|"
        r"Lima Idrobo|L[oó]pez Garc[ií]a|Ter[aá]n|Portmann|S\.F\.L\.R|github\.com/",
        re.I)
    hits = []
    for base in SR_BUILD:
        f = os.path.join("sections", base + ".tex")
        if not os.path.exists(f):
            continue
        for i, line in enumerate(open(f, encoding="utf-8"), 1):
            if not line.lstrip().startswith("%") and IDENT.search(line):
                hits.append(f"{f}:{i}: {line.strip()[:90]}")
    print()
    print(f"Identifying strings in non-comment lines (check that each sits "
          f"inside an \\ifanon ... \\else branch): {len(hits)}")
    for h in hits:
        print("  " + h)
