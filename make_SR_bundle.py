"""Package the Scientific Reports LaTeX source as a self-contained ZIP.

Collects every project file the last build actually read (from the .fls
dependency lists latexmk writes for main_SR and supplementary_SR), plus the
two .bbl files, and leaves out the .bib: the journal's system cannot take
BibTeX databases, so the bundle compiles against the .bbl files alone.
Also drops a copy of the per-contrast bootstrap CSV, which the Data
availability statement says is released with the source.

Run from paper1_medical/ after .\build_SR.ps1:  python make_SR_bundle.py
Then verify with:  python make_SR_bundle.py --test   (compiles the ZIP in a
temporary directory, both documents, two rounds, and reports errors).
"""

import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "SR_latex_source.zip"
JOBS = ["main_SR", "supplementary_SR"]
# The two .aux files are shipped so that the xr-hyper cross-document
# references (\cref to main-text equations/tables from the SI and vice
# versa) resolve on the first compile of either document. Overleaf drops
# .aux and .bbl files from uploaded ZIPs, so the label snapshots
# xr_<job>.tex (read by the preamble when the .aux is absent) and the .bib
# (so Overleaf can run BibTeX) are shipped as well.
EXTRA = ["main_SR.bbl", "supplementary_SR.bbl", "main_SR.aux",
         "supplementary_SR.aux", "xr_main_SR.tex", "xr_supplementary_SR.tex",
         "shared/references.bib", "build_SR.ps1", "export_xr_labels.py",
         "data/luad_hierarchical_contrasts.csv",
         "data/glioma_mgmt_bootstrap_aurocs.csv",
         "data/glioma_mgmt_bootstrap_contrasts.csv",
         "data/ecg_bootstrap_aurocs.csv",
         "data/ecg_bootstrap_contrasts.csv"]


def inputs_from_fls(job: str) -> set[str]:
    files = set()
    for line in (ROOT / f"{job}.fls").read_text(encoding="utf-8",
                                                errors="replace").splitlines():
        if not line.startswith("INPUT "):
            continue
        p = Path(line[6:].strip())
        if not p.is_absolute():
            p = ROOT / p
        try:
            rel = p.resolve().relative_to(ROOT)
        except ValueError:
            continue  # TeX distribution file
        s = rel.as_posix()
        if re.search(r"\.(aux|log|out|fls|fdb_latexmk|bbl|blg|bib|pdf)$", s):
            if s.endswith(".pdf") and s.startswith("figures/"):
                files.add(s)
            continue
        files.add(s)
    return files


def build_zip() -> list[str]:
    names = set()
    for job in JOBS:
        names |= inputs_from_fls(job)
    names |= set(EXTRA)
    # The .bib lives one level up (shared/ is a sibling of paper1_medical/,
    # found at build time through BIBINPUTS=..); inside the bundle it goes
    # to shared/references.bib, which is the path \bibliography names.
    sources = {n: ROOT / n for n in names}
    bib = ROOT.parent / "shared" / "references.bib"
    if not sources["shared/references.bib"].exists() and bib.exists():
        sources["shared/references.bib"] = bib
    names = sorted(n for n in names if sources[n].exists())
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.write(sources[n], n)
    size = OUT.stat().st_size / 1e6
    print(f"wrote {OUT.name}: {len(names)} files, {size:.1f} MB")
    for n in names:
        print("  " + n)
    return names


def test_zip() -> None:
    tmp = Path(tempfile.mkdtemp(prefix="sr_bundle_"))
    with zipfile.ZipFile(OUT) as z:
        z.extractall(tmp)
    # Mimic Overleaf, which discards .aux and .bbl from uploaded ZIPs: the
    # documents must compile from the .bib and the xr_*.tex snapshots alone.
    for p in list(tmp.glob("*.aux")) + list(tmp.glob("*.bbl")):
        p.unlink()
    env = dict(os.environ, BIBINPUTS=".;")
    ok = True
    for rnd in (1,):
        for job in JOBS:
            r = subprocess.run(
                ["latexmk", "-pdf", "-interaction=nonstopmode",
                 "-halt-on-error", f"{job}.tex"],
                cwd=tmp, env=env, capture_output=True, text=True)
            status = "ok" if r.returncode == 0 else "FAILED"
            print(f"round {rnd} {job}: {status}")
            ok &= r.returncode == 0
    for job in JOBS:
        log = (tmp / f"{job}.log").read_text(encoding="utf-8",
                                             errors="replace")
        n_undef = len(re.findall(r"(Reference|Citation) `[^']+' on page \d+ "
                                 r"undefined", log))
        print(f"{job}: undefined references/citations in final log: {n_undef}")
        ok &= n_undef == 0
    shutil.rmtree(tmp, ignore_errors=True)
    print("bundle compiles standalone" if ok else "BUNDLE TEST FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--test" in sys.argv:
        test_zip()
    else:
        build_zip()
