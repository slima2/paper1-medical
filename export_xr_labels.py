"""Snapshot each document's labels into a .tex file the other document can
read when its .aux is not there.

main_SR.tex and supplementary_SR.tex cite each other through xr-hyper, which
reads the other document's .aux. Overleaf silently drops .aux (and .bbl)
files from uploaded ZIPs, so on Overleaf every cross-document \\cref printed
as ``??''. preamble_SR.tex therefore falls back to xr_<job>.tex, written
here from <job>.aux after a full build: every \\newlabel line, with the URL
field set to <job>.pdf exactly as xr-hyper would set it, so the links point
at the other PDF rather than at a missing anchor.

Run from paper1_medical/ after both documents have compiled (build_SR.ps1
does this automatically):  python export_xr_labels.py
"""

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
JOBS = ["main_SR", "supplementary_SR"]


def groups(s: str) -> list[str]:
    """Split a balanced brace string '{a}{b}{c}' into its top-level groups."""
    out, depth, start = [], 0, None
    for i, ch in enumerate(s):
        if ch == "{":
            if depth == 0:
                start = i + 1
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                out.append(s[start:i])
    return out


def newlabel_lines(aux: Path, url: str) -> list[str]:
    lines = []
    for line in aux.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.startswith(r"\newlabel"):
            continue
        name, body = groups(line[len(r"\newlabel"):])[:2]
        fields = groups(body)
        # xr-hyper keeps the first four fields and writes the URL as fifth.
        fields = (fields + ["", "", "", ""])[:4] + [url]
        lines.append(r"\newlabel{%s}{%s}" % (name, "".join("{%s}" % f
                                                           for f in fields)))
    return lines


def main() -> int:
    for job in JOBS:
        aux = ROOT / f"{job}.aux"
        if not aux.exists():
            print(f"{aux.name} missing; build first", file=sys.stderr)
            return 1
        lines = newlabel_lines(aux, f"{job}.pdf")
        out = ROOT / f"xr_{job}.tex"
        header = [
            f"% Labels of {job}.tex, exported from {job}.aux by",
            "% export_xr_labels.py. Read by preamble_SR.tex only when the .aux",
            "% itself is absent (Overleaf drops .aux files from uploaded ZIPs).",
            "% Do not edit: regenerate with .\\build_SR.ps1.",
        ]
        out.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
        print(f"wrote {out.name}: {len(lines)} labels")
    return 0


if __name__ == "__main__":
    sys.exit(main())
