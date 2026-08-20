#!/usr/bin/env python
"""Assemble the VASARI source-vocabulary channel for domain B (glioma).

VASARI is the ANORAK equivalent: a consensus-defined controlled terminology of
~30 semantic MR features scored by neuroradiologists on the TCGA glioma
collections. It is the annotated seed set that a feature classifier is trained
on and then propagated to the much larger BraTS 2021 cohort -- exactly the
Zenodo-ANORAK -> TCGA-LUAD move in the manuscript.

TCIA serves the score CSVs as collection supporting documents rather than
through a stable REST path, so this script does best-effort discovery and
otherwise tells you precisely what to place where. The feature key PDF is the
authority for what each column means; column names are taken from the CSVs
themselves rather than hardcoded, because the coding scheme has been revised.

Outputs, under $DATA_ROOT/glioma/vasari/:
    vasari_features.parquet    subject x feature, one row per annotated case
    vasari_features.csv        same, for inspection
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import pandas as pd

TCIA_PAGES = {
    "TCGA-GBM": "https://wiki.cancerimagingarchive.net/display/Public/VASARI+Research+Project",
    "TCGA-LGG": "https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24282890",
}

# TCIA subject identifiers, e.g. TCGA-06-0125 or TCGA-DU-7304.
SUBJECT_RE = re.compile(r"TCGA-[0-9A-Z]{2}-[0-9A-Z]{4}", re.IGNORECASE)


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "glioma" / "vasari"


def find_subject_column(df: pd.DataFrame) -> str | None:
    """Locate the column holding TCIA subject IDs, whatever it is called."""
    for col in df.columns:
        values = df[col].astype(str).head(50)
        if values.str.contains(SUBJECT_RE, regex=True).mean() > 0.5:
            return col
    return None


def load_score_tables(raw_dir: Path) -> dict[str, pd.DataFrame]:
    tables: dict[str, pd.DataFrame] = {}
    for path in sorted(raw_dir.glob("*.csv")) + sorted(raw_dir.glob("*.xlsx")):
        try:
            df = pd.read_excel(path) if path.suffix == ".xlsx" else pd.read_csv(path)
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"[warn] could not read {path.name}: {exc}")
            continue

        subject_col = find_subject_column(df)
        if subject_col is None:
            print(f"[skip] {path.name}: no TCGA subject column found")
            continue

        df = df.rename(columns={subject_col: "subject_id"})
        df["subject_id"] = df["subject_id"].astype(str).str.strip().str.upper()
        df["source_file"] = path.name
        tables[path.name] = df
        print(f"[ok] {path.name}: {len(df)} rows x {df.shape[1] - 2} features")
    return tables


def instructions(raw_dir: Path) -> None:
    print("\nNo VASARI score tables found in", raw_dir)
    print("Download them by hand (they are small) from:")
    for collection, url in TCIA_PAGES.items():
        print(f"  {collection:10s} {url}")
    print(
        "\nLook for the supporting-document links named along the lines of\n"
        "  'VASARI annotations' / 'VASARI information (CSV)' / "
        "'TCGA-GBM VASARI Round 1'\n"
        f"and drop the files into {raw_dir}. Then re-run this script.\n"
        "\nExpected coverage per the collection descriptions: 178 complete "
        "TCGA-LGG cases and 75 final TCGA-GBM cases."
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    vasari_root = resolve_root(args.data_root)
    raw_dir = vasari_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    tables = load_score_tables(raw_dir)
    if not tables:
        instructions(raw_dir)
        sys.exit(1)

    merged = pd.concat(tables.values(), ignore_index=True)

    # A subject scored in more than one round appears more than once; keep the
    # last occurrence, which is the revised score.
    duplicates = merged["subject_id"].duplicated().sum()
    if duplicates:
        print(f"[info] {duplicates} duplicate subject rows, keeping the last")
        merged = merged.drop_duplicates(subset="subject_id", keep="last")

    merged = merged.set_index("subject_id").sort_index()

    feature_cols = [c for c in merged.columns if c != "source_file"]
    coverage = merged[feature_cols].notna().mean()
    sparse = coverage[coverage < 0.5]
    if len(sparse):
        print(f"[warn] {len(sparse)} features scored on under half the cohort:")
        print("       " + ", ".join(sparse.index[:10]))

    merged.to_parquet(vasari_root / "vasari_features.parquet")
    merged.to_csv(vasari_root / "vasari_features.csv")

    print(f"\n{len(merged)} annotated subjects x {len(feature_cols)} features")
    print(f"wrote {vasari_root / 'vasari_features.parquet'}")
    key = raw_dir / "VASARI_MR_featurekey.pdf"
    print(
        f"\nFeature semantics: {key}"
        if key.exists()
        else "\nFeature key PDF missing; run: bash download_glioma.sh --vasari"
    )
    print(
        "Next: join to BraTS subjects through the BraTS-ID <-> TCIA-ID crosswalk\n"
        "XLSX on the RSNA-ASNR-MICCAI-BRATS-2021 TCIA page, then train the\n"
        "feature classifier on this annotated seed and propagate to the full cohort."
    )


if __name__ == "__main__":
    main()
