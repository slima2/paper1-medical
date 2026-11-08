"""Count where each reference is cited, to find the least load-bearing ones.

Only the files main_CM.tex actually inputs are scanned, and LaTeX comments are
skipped, so a citation that survives only inside a commented-out draft does not
count as used.
"""

from __future__ import annotations

import collections
import pathlib
import re

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
    "01_abstract_CM.tex": "ABS", "02_introduction_CM.tex": "INTRO",
    "03_results_CM.tex": "RESULTS", "04_discussion_CM.tex": "DISC",
    "05_methods_CM.tex": "METHODS", "06_declarations_CM.tex": "AVAIL",
}

CITE = re.compile(r"\\cite[a-zA-Z]*\*?(?:\[[^\]]*\])*\{([^}]*)\}")
WEIGHT = {"INTRO": 3, "RESULTS": 3, "DISC": 3, "ABS": 3,
          "METHODS": 2, "EXTDATA": 1, "SUPP": 1, "AVAIL": 1}


def strip_comments(text: str) -> str:
    out = []
    for line in text.split("\n"):
        buf, i = [], 0
        while i < len(line):
            if line[i] == "\\" and i + 1 < len(line):
                buf.append(line[i:i + 2])
                i += 2
                continue
            if line[i] == "%":
                break
            buf.append(line[i])
            i += 1
        out.append("".join(buf))
    return "\n".join(out)


def main() -> None:
    use: dict[str, list[str]] = collections.defaultdict(list)
    for rel in FILES:
        p = ROOT / rel
        if not p.exists():
            continue
        name = pathlib.Path(rel).name
        short = SHORT.get(name, name.replace(".tex", ""))
        for line in strip_comments(
                p.read_text(encoding="utf-8", errors="ignore")).split("\n"):
            for m in CITE.finditer(line):
                for key in m.group(1).split(","):
                    key = key.strip()
                    if key:
                        use[key].append(short)

    singles = {k: v for k, v in use.items() if len(v) == 1}
    print(f"Referencias citadas: {len(use)}   "
          f"citadas una sola vez: {len(singles)}\n")
    print("Candidatas a recorte, ordenadas por coste (menor primero).")
    print("El coste sube si la cita vive en el cuerpo del articulo.\n")
    ranked = sorted(singles.items(), key=lambda kv: WEIGHT.get(kv[1][0], 1))
    for key, where in ranked:
        cost = {1: "bajo ", 2: "medio", 3: "ALTO "}[WEIGHT.get(where[0], 1)]
        print(f"  [{cost}] {where[0]:8s}  {key}")


if __name__ == "__main__":
    main()
