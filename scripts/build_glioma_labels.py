#!/usr/bin/env python
"""Assemble the glioma genotype targets IDH / 1p19q / MGMT for domain B.

Writes $DATA_ROOT/glioma/derived/glioma_labels.parquet with
    index = subject_id  (the BraTS subject folder name, e.g. BraTS2021_00495)
    columns = idh, onep19q, mgmt   (1/0, NaN where unknown)
consumed by run_glioma_domainB.py.

Where each label comes from (they do NOT share a source):
  * MGMT   : BraTS 2021 (RSNA-MICCAI) train_labels.csv -> MGMT_value, available
             for the whole classification cohort. This is the target that makes
             the domain worth doing.
  * IDH,   : molecular calls exist only for the TCGA subset of BraTS. They are
    1p19q    joined through the BraTS-ID <-> TCIA-ID crosswalk XLSX (from the
             TCIA BraTS 2021 page) to a TCGA genotype table.

The TCGA genotype table is expected at
    $DATA_ROOT/glioma/tcga_genotype.csv
with columns  tcga_id, idh, onep19q  (idh: 1=mutant/0=wildtype;
onep19q: 1=codeleted/0=intact). Build it once from cBioPortal studies
lgg_tcga and gbm_tcga (Download tab, or the REST API): IDH from IDH1/IDH2
mutation status, 1p/19q from the codeletion clinical attribute. A best-effort
cBioPortal fetch is provided via --fetch-cbioportal (needs internet).

Usage (on the DGX, after `source env.sh`):
    python build_glioma_labels.py
    python build_glioma_labels.py --fetch-cbioportal    # try to build tcga_genotype.csv
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUBJECT_RE = re.compile(r"BraTS2021_\d{5}", re.IGNORECASE)
TCGA_RE = re.compile(r"TCGA-[0-9A-Z]{2}-[0-9A-Z]{4}", re.IGNORECASE)


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "glioma"


def brats_subjects(glioma_root: Path) -> list[str]:
    """All BraTS subject folder names present locally (the imaging cohort)."""
    seen: set[str] = set()
    for p in glioma_root.rglob("BraTS2021_*"):
        if p.is_dir() and any(p.glob("*_flair.nii*")):
            seen.add(p.name)
    return sorted(seen)


# --------------------------------------------------------------------------
# MGMT from the Kaggle train_labels.csv
# --------------------------------------------------------------------------

def load_mgmt(glioma_root: Path) -> pd.Series:
    path = glioma_root / "brats2021" / "train_labels.csv"
    if not path.exists():
        print(f"[warn] {path} not found; MGMT will be all-NaN "
              "(run download_glioma.sh --brats)")
        return pd.Series(dtype="float64", name="mgmt")
    df = pd.read_csv(path)
    id_col = "BraTS21ID" if "BraTS21ID" in df.columns else df.columns[0]
    val_col = "MGMT_value" if "MGMT_value" in df.columns else df.columns[1]
    # BraTS21ID is a zero-padded 5-digit int -> BraTS2021_00000 folder name.
    subj = df[id_col].apply(lambda v: f"BraTS2021_{int(v):05d}")
    out = pd.Series(df[val_col].astype(float).values, index=subj.values, name="mgmt")
    print(f"[ok] MGMT: {out.notna().sum()} labelled subjects")
    return out


# --------------------------------------------------------------------------
# Crosswalk BraTS-ID <-> TCGA-ID
# --------------------------------------------------------------------------

def load_crosswalk(glioma_root: Path) -> pd.DataFrame:
    """Return DataFrame [subject_id, tcga_id] from the TCIA crosswalk XLSX/CSV."""
    cands = list(glioma_root.glob("**/*crosswalk*.xls*")) + \
            list(glioma_root.glob("**/*mapping*.xls*")) + \
            list(glioma_root.glob("**/*BraTS*TCIA*.xls*")) + \
            list(glioma_root.glob("**/*crosswalk*.csv"))
    if not cands:
        print("[warn] no BraTS<->TCIA crosswalk found; IDH/1p19q will be all-NaN.\n"
              "       download the crosswalk XLSX from the TCIA BraTS 2021 page into "
              f"{glioma_root} (see download_glioma.sh --brats).")
        return pd.DataFrame(columns=["subject_id", "tcga_id"])
    path = cands[0]
    df = pd.read_excel(path) if path.suffix.startswith(".xls") else pd.read_csv(path)
    # find the two columns holding BraTS ids and TCGA ids, whatever they are called
    def col_matching(rx):
        for c in df.columns:
            if df[c].astype(str).str.contains(rx, regex=True).mean() > 0.3:
                return c
        return None
    bcol, tcol = col_matching(SUBJECT_RE), col_matching(TCGA_RE)
    if bcol is None or tcol is None:
        print(f"[warn] could not identify columns in {path.name}; IDH/1p19q all-NaN")
        return pd.DataFrame(columns=["subject_id", "tcga_id"])
    out = pd.DataFrame({
        "subject_id": df[bcol].astype(str).str.extract(f"({SUBJECT_RE.pattern})",
                                                        expand=False),
        "tcga_id": df[tcol].astype(str).str.extract(f"({TCGA_RE.pattern})",
                                                    expand=False).str.upper(),
    }).dropna()
    print(f"[ok] crosswalk: {len(out)} BraTS<->TCGA pairs from {path.name}")
    return out


# --------------------------------------------------------------------------
# TCGA IDH / 1p19q
# --------------------------------------------------------------------------

def load_tcga_genotype(glioma_root: Path) -> pd.DataFrame:
    path = glioma_root / "tcga_genotype.csv"
    if not path.exists():
        print(f"[warn] {path} not found; IDH/1p19q will be all-NaN.\n"
              "       build it from cBioPortal (lgg_tcga, gbm_tcga) with columns "
              "tcga_id,idh,onep19q, or run --fetch-cbioportal.")
        return pd.DataFrame(columns=["tcga_id", "idh", "onep19q"])
    df = pd.read_csv(path)
    df["tcga_id"] = df["tcga_id"].astype(str).str.upper().str.extract(
        f"({TCGA_RE.pattern})", expand=False)
    print(f"[ok] TCGA genotype: {len(df)} rows from {path.name}")
    return df[["tcga_id", "idh", "onep19q"]].dropna(subset=["tcga_id"])


def fetch_cbioportal(glioma_root: Path) -> None:
    """Best-effort build of tcga_genotype.csv from cBioPortal REST API."""
    try:
        import requests
    except Exception:
        sys.exit("pip install requests to use --fetch-cbioportal")
    base = "https://www.cbioportal.org/api"
    rows: dict[str, dict] = {}
    for study in ("lgg_tcga", "gbm_tcga"):
        prof = f"{study}_mutations"
        sample_list = f"{study}_all"
        # IDH1/IDH2 mutation status
        try:
            r = requests.post(
                f"{base}/molecular-profiles/{prof}/mutations/fetch",
                params={"projection": "SUMMARY"},
                json={"sampleListId": sample_list,
                      "entrezGeneIds": [3417, 3418]},  # IDH1, IDH2
                timeout=60,
            )
            r.raise_for_status()
            muts = r.json()
            for m in muts:
                pid = m.get("patientId") or m.get("sampleId", "")[:12]
                rows.setdefault(pid, {"tcga_id": pid, "idh": 0, "onep19q": np.nan})
                rows[pid]["idh"] = 1
        except Exception as exc:
            print(f"[warn] cBioPortal IDH fetch failed for {study}: {exc}")
        # 1p/19q codeletion clinical attribute
        for attr in ("1P_19Q_CODELETION", "CODELETION_1P19Q", "X1P_19Q_CODELETION"):
            try:
                r = requests.post(
                    f"{base}/studies/{study}/clinical-data/fetch",
                    params={"clinicalDataType": "PATIENT", "projection": "SUMMARY"},
                    json={"attributeIds": [attr], "ids": []},
                    timeout=60,
                )
                if r.status_code != 200 or not r.json():
                    continue
                for c in r.json():
                    pid = c.get("patientId")
                    val = str(c.get("value", "")).strip().lower()
                    code = 1 if val in ("codeleted", "yes", "1", "co-deletion") else 0
                    rows.setdefault(pid, {"tcga_id": pid, "idh": np.nan, "onep19q": np.nan})
                    rows[pid]["onep19q"] = code
                break
            except Exception:
                continue
    if not rows:
        sys.exit("cBioPortal fetch produced nothing; build tcga_genotype.csv manually")
    out = pd.DataFrame(list(rows.values()))
    dest = glioma_root / "tcga_genotype.csv"
    out.to_csv(dest, index=False)
    print(f"[ok] wrote {dest} ({len(out)} patients). Review before using.")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--fetch-cbioportal", action="store_true",
                    help="build tcga_genotype.csv from cBioPortal, then exit")
    args = ap.parse_args()

    glioma_root = resolve_root(args.data_root)
    if args.fetch_cbioportal:
        fetch_cbioportal(glioma_root)
        return

    subjects = brats_subjects(glioma_root)
    if not subjects:
        sys.exit(f"no BraTS2021_* subject folders under {glioma_root}; "
                 "run download_glioma.sh --brats")
    print(f"{len(subjects)} BraTS subjects")

    labels = pd.DataFrame(index=pd.Index(subjects, name="subject_id"),
                          columns=["idh", "onep19q", "mgmt"], dtype="float64")

    mgmt = load_mgmt(glioma_root)
    labels["mgmt"] = mgmt.reindex(labels.index)

    xwalk = load_crosswalk(glioma_root)
    geno = load_tcga_genotype(glioma_root)
    if len(xwalk) and len(geno):
        merged = xwalk.merge(geno, on="tcga_id", how="inner").set_index("subject_id")
        labels.loc[merged.index.intersection(labels.index), "idh"] = \
            merged["idh"].reindex(labels.index).dropna()
        labels.loc[merged.index.intersection(labels.index), "onep19q"] = \
            merged["onep19q"].reindex(labels.index).dropna()

    derived = glioma_root / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    out = derived / "glioma_labels.parquet"
    labels.to_parquet(out)
    labels.to_csv(out.with_suffix(".csv"))

    print("\nlabel coverage (n positive / n labelled):")
    for c in ("idh", "onep19q", "mgmt"):
        s = labels[c]
        print(f"  {c:8s} n_+={int(np.nansum(s.values)):4d}  "
              f"labelled={int(s.notna().sum()):4d}  floor~37 "
              f"{'OK' if np.nansum(s.values) >= 37 else 'UNDERPOWERED'}")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
