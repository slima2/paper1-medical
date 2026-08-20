#!/usr/bin/env bash
# Domain A' -- CPTAC-LUAD cohort transfer: download the matched H&E WSIs.
#
# CPTAC-LUAD on the NCI Imaging Data Commons carries BOTH radiology (CT) and
# digital pathology. We want only the pathology slides (DICOM VL Whole Slide
# Microscopy, Modality == SM); this script filters them out and skips the CT.
# No credentials are needed; idc-index streams from public AWS/GCS buckets.
#
#   --dry-run   size the pathology selection without downloading
#   (default)   download the SM series into $DATA_ROOT/luad_cptac/wsi
#
# Idempotent: idc-index skips series already present.

set -euo pipefail

: "${DATA_ROOT:?source env.sh first (or export DATA_ROOT=...)}"
ROOT="${DATA_ROOT}/luad_cptac"
WDIR="${ROOT}/wsi"
mkdir -p "${WDIR}"

DRY=0
[[ "${1:-}" == "--dry-run" ]] && DRY=1

python -c "import idc_index" 2>/dev/null || { echo "installing idc-index"; pip install --quiet --upgrade idc-index; }

DRY="${DRY}" DEST="${WDIR}" ROOT="${ROOT}" python - <<'PY'
import os
import pandas as pd
from idc_index import IDCClient

dest = os.environ["DEST"]
root = os.environ["ROOT"]
dry = os.environ["DRY"] == "1"

c = IDCClient.client()
df = c.index

# Column names have been stable but guard anyway.
col_collection = "collection_id" if "collection_id" in df.columns else "collection_name"
if "Modality" not in df.columns:
    raise SystemExit(f"unexpected idc-index schema: {list(df.columns)[:20]}")

sel = df[(df[col_collection].str.lower() == "cptac_luad") & (df["Modality"] == "SM")]
if sel.empty:
    # Fall back: some releases label the collection 'cptac-luad'.
    sel = df[(df[col_collection].str.lower().str.replace('-', '_') == "cptac_luad")
             & (df["Modality"] == "SM")]

# Restrict to the sequenced cohort we have labels for. The pathology collection
# on IDC is broader (imaged cases without matched WGS/WES) and includes
# normal-adjacent slides; downloading all of it wastes hundreds of GB. If the
# labels are present, keep only those patients.
labels_path = os.path.join(root, "labels.parquet")
if "PatientID" in sel.columns and os.path.exists(labels_path):
    cohort = set(pd.read_parquet(labels_path).reset_index()["patient_id"].astype(str))
    before = sel["PatientID"].nunique()
    sel = sel[sel["PatientID"].astype(str).isin(cohort)]
    print(f"filtered to labelled cohort: {sel['PatientID'].nunique()}/{before} patients")
else:
    print("[warn] labels.parquet not found; NOT filtering to the cohort "
          "(run fetch_cptac_labels.py first to avoid a ~470 GB pull)")

n_series = sel["SeriesInstanceUID"].nunique()
n_patients = sel["PatientID"].nunique() if "PatientID" in sel.columns else "?"
size_gb = None
for cand in ("series_size_MB", "series_size_mb"):
    if cand in sel.columns:
        size_gb = sel[cand].sum() / 1024.0
        break
print(f"CPTAC-LUAD pathology (SM): {n_series} series, {n_patients} patients"
      + (f", ~{size_gb:.1f} GB" if size_gb else ""))

if dry:
    print("dry-run: not downloading")
else:
    c.download_from_selection(
        seriesInstanceUID=sel["SeriesInstanceUID"].dropna().unique().tolist(),
        downloadDir=dest,
    )
    print(f"downloaded SM series to {dest}")
PY

echo "done."
echo "NOTE: IDC serves WSIs as DICOM VL Whole Slide Microscopy, not .svs."
echo "  OpenSlide >= 4.0 reads DICOM WSI directly; otherwise convert with"
echo "  'wsidicom' or 'dicom2svs' before feeding the lchai tiling pipeline."
