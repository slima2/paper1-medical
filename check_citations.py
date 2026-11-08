"""Audit every \cite key reachable from main_CM.tex against shared/references.bib.

Reports, in order:
  1. keys cited but absent from the .bib            (render as [?])
  2. keys cited whose .bib entry looks incomplete
  3. .bib entries never cited in the CM build      (dead weight)
  4. keys cited only from files not in the CM build
  5. duplicate .bib keys
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).parent
BIB = ROOT.parent / "shared" / "references.bib"
MAIN = ROOT / "main_CM.tex"

INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
CITE_RE = re.compile(r"\\(?:cite|citet|citep|citeauthor|citeyear)\s*(?:\[[^\]]*\])*\s*\{([^}]+)\}")
BIBKEY_RE = re.compile(r"^\s*@(\w+)\s*\{\s*([^,\s]+)\s*,", re.M)


def strip_comments(text: str) -> str:
    out = []
    for line in text.splitlines():
        idx, esc = None, False
        for i, ch in enumerate(line):
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
            elif ch == "%":
                idx = i
                break
        out.append(line if idx is None else line[:idx])
    return "\n".join(out)


def resolve(name: str) -> Path | None:
    for cand in (ROOT / name, ROOT / (name + ".tex")):
        if cand.is_file():
            return cand
    return None


def collect(path: Path, seen: set[Path]) -> list[tuple[Path, str]]:
    """Return (file, citekey) pairs reachable from path, following \input."""
    if path in seen or not path.is_file():
        return []
    seen.add(path)
    body = strip_comments(path.read_text(encoding="utf-8", errors="ignore"))
    found = [(path, k.strip()) for m in CITE_RE.finditer(body) for k in m.group(1).split(",")]
    for m in INPUT_RE.finditer(body):
        child = resolve(m.group(1))
        if child:
            found += collect(child, seen)
    return found


bib_raw = BIB.read_text(encoding="utf-8", errors="ignore")
bib_entries: dict[str, str] = {}
duplicates: list[str] = []
matches = list(BIBKEY_RE.finditer(bib_raw))
for i, m in enumerate(matches):
    key = m.group(2)
    end = matches[i + 1].start() if i + 1 < len(matches) else len(bib_raw)
    if key in bib_entries:
        duplicates.append(key)
    bib_entries[key] = bib_raw[m.start():end]

used = collect(MAIN, set())
used_keys = {k for _, k in used}

# Every .tex in sections/ and tables/, so we can tell CM-only from npj-only usage.
all_pairs: list[tuple[Path, str]] = []
for f in sorted(ROOT.rglob("*.tex")):
    body = strip_comments(f.read_text(encoding="utf-8", errors="ignore"))
    all_pairs += [(f, k.strip()) for m in CITE_RE.finditer(body) for k in m.group(1).split(",")]
all_keys = {k for _, k in all_pairs}

where: dict[str, set[str]] = {}
for f, k in used:
    where.setdefault(k, set()).add(f.name)

print(f"bib entries      : {len(bib_entries)}")
print(f"cited in CM      : {len(used_keys)}")
print(f"cited anywhere   : {len(all_keys)}")

missing = sorted(used_keys - bib_entries.keys())
print(f"\n[1] cited in CM  BUT MISSING FROM .bib  ({len(missing)})")
for k in missing:
    print(f"    {k:32s} <- {', '.join(sorted(where[k]))}")

REQUIRED = {"article": ("author", "title", "journal", "year"),
            "inproceedings": ("author", "title", "booktitle", "year"),
            "incollection": ("author", "title", "booktitle", "year"),
            "book": ("author", "title", "year"),
            "misc": ("title",),
            "techreport": ("author", "title", "year"),
            "phdthesis": ("author", "title", "year")}
print("\n[2] INCOMPLETE .bib ENTRIES CITED IN CM")
bad = 0
for k in sorted(used_keys & bib_entries.keys()):
    entry = bib_entries[k]
    etype = BIBKEY_RE.search(entry).group(1).lower()
    fields = {f.lower() for f in re.findall(r"[\s,{]([A-Za-z]+)\s*=", entry)}
    lacking = [r for r in REQUIRED.get(etype, ()) if r not in fields]
    if lacking:
        bad += 1
        print(f"    {k:32s} ({etype}) missing: {', '.join(lacking)}")
    if re.search(r"under review|in press|submitted|forthcoming", entry, re.I):
        print(f"    {k:32s} !! entry text mentions under review / submitted")
if not bad:
    print("    none")

unused = sorted(bib_entries.keys() - all_keys)
print(f"\n[3] .bib ENTRIES CITED NOWHERE AT ALL  ({len(unused)})")
print("    " + ("\n    ".join(unused) if unused else "none"))

npj_only = sorted(all_keys - used_keys - bib_entries.keys() | (all_keys - used_keys))
print(f"\n[4] CITED ONLY OUTSIDE THE CM build  ({len(npj_only)})")
for k in npj_only:
    files = sorted({f.name for f, kk in all_pairs if kk == k})
    flag = "" if k in bib_entries else "   [also missing from .bib]"
    print(f"    {k:32s} <- {', '.join(files)}{flag}")

print(f"\n[5] DUPLICATE .bib KEYS  ({len(duplicates)})")
print("    " + ("\n    ".join(sorted(set(duplicates))) if duplicates else "none"))

sys.exit(1 if missing else 0)

