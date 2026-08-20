#!/usr/bin/env bash
# Domain B — glioma MRI -> genotype (IDH / 1p19q / MGMT).
#
#   --vasari   VASARI feature key + score CSVs        (small; the source ontology)
#   --idc      TCGA-GBM / TCGA-LGG / UPENN-GBM DICOM  (~150 GB via IDC)
#   --brats    BraTS 2021 training+validation         (~137 GB via Kaggle)
#   --synapse  BraTS 2021 sequestered test set        (needs SYNAPSE_AUTH_TOKEN)
#   --all      everything above
#
# Idempotent and resumable.

set -euo pipefail

: "${DATA_ROOT:?source env.sh first}"
ROOT="${DATA_ROOT}/glioma"
mkdir -p "${ROOT}"

log() { printf '[glioma %(%H:%M:%S)T] %s\n' -1 "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

need_python_pkg() {
  local pkg="$1" import_as="$2"
  python -c "import ${import_as}" 2>/dev/null || {
    log "installing ${pkg}"
    pip install --quiet --upgrade "${pkg}"
  }
}

do_vasari=0 do_idc=0 do_brats=0 do_synapse=0
[[ $# -eq 0 ]] && { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
for arg in "$@"; do
  case "${arg}" in
    --vasari)  do_vasari=1 ;;
    --idc)     do_idc=1 ;;
    --brats)   do_brats=1 ;;
    --synapse) do_synapse=1 ;;
    --all)     do_vasari=1; do_idc=1; do_brats=1; do_synapse=1 ;;
    *) echo "unknown option: ${arg}" >&2; exit 2 ;;
  esac
done

# ---------------------------------------------------------------------------
# VASARI — the controlled vocabulary. This is the ANORAK equivalent and the
# single most important artefact for the framework: without it there is no
# source-side channel to decompose.
# ---------------------------------------------------------------------------
if (( do_vasari )); then
  VDIR="${ROOT}/vasari/raw"
  mkdir -p "${VDIR}"

  # The feature key has a stable URL on the TCIA CDN.
  if [[ ! -f "${VDIR}/VASARI_MR_featurekey.pdf" ]]; then
    log "fetching VASARI feature key"
    wget -q --show-progress -O "${VDIR}/VASARI_MR_featurekey.pdf" \
      "https://www.cancerimagingarchive.net/wp-content/uploads/VASARI_MR_featurekey.pdf"
  fi

  # The per-subject score CSVs are served as collection supporting documents
  # and do not have a stable REST path. fetch_vasari.py attempts discovery and
  # falls back to printing instructions.
  log "score CSVs: run 'python fetch_vasari.py' next"
  log "  if discovery fails, download manually into ${VDIR}:"
  log "  https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=24282890  (TCGA-LGG, 178 complete)"
  log "  https://wiki.cancerimagingarchive.net/display/Public/VASARI+Research+Project (TCGA-GBM, 75 final)"
fi

# ---------------------------------------------------------------------------
# DICOM series via the NCI Imaging Data Commons. idc-index replaces the
# retiring NBIA REST API and needs no credentials; downloads run through a
# bundled s5cmd against public AWS/GCS buckets.
# ---------------------------------------------------------------------------
if (( do_idc )); then
  need_python_pkg "idc-index" "idc_index"
  IDIR="${ROOT}/dicom"
  mkdir -p "${IDIR}"
  for coll in tcga_gbm tcga_lgg upenn_gbm; do
    if [[ -d "${IDIR}/${coll}" && -n "$(ls -A "${IDIR}/${coll}" 2>/dev/null)" ]]; then
      log "${coll} already present, skipping"
      continue
    fi
    log "sizing ${coll}"
    python - "${coll}" "${IDIR}" <<'PY'
import sys
from idc_index import IDCClient
coll, dest = sys.argv[1], sys.argv[2]
client = IDCClient.client()
client.download_from_selection(collection_id=coll, downloadDir=dest, dry_run=True)
PY
    log "downloading ${coll} -> ${IDIR}"
    idc download "${coll}" --download-dir "${IDIR}"
  done
fi

# ---------------------------------------------------------------------------
# BraTS 2021. The Kaggle mirror is the practical route: it carries
# train_labels.csv with the binary MGMT status, which is the target that makes
# this domain worth doing. Accept the competition rules once in a browser.
# ---------------------------------------------------------------------------
if (( do_brats )); then
  need_python_pkg "kaggle" "kaggle"
  BDIR="${ROOT}/brats2021"
  mkdir -p "${BDIR}"
  if [[ ! -f "${BDIR}/train_labels.csv" ]]; then
    log "downloading BraTS 2021 from Kaggle (~137 GB)"
    kaggle competitions download \
      -c rsna-miccai-brain-tumor-radiogenomic-classification \
      -p "${BDIR}"
    log "unzipping"
    ( cd "${BDIR}" && unzip -q -n ./*.zip )
  else
    log "BraTS 2021 already present, skipping"
  fi
  log "MGMT labels: ${BDIR}/train_labels.csv"
  log "TCIA alternative (142 GB, needs IBM Aspera Connect):"
  log "  https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/"
  log "  that page also hosts the BraTS-ID <-> TCIA-ID crosswalk XLSX, which you"
  log "  need to join BraTS subjects to their VASARI scores."
fi

# ---------------------------------------------------------------------------
# Sequestered test set. Only needed if you want the held-out challenge split.
# ---------------------------------------------------------------------------
if (( do_synapse )); then
  : "${SYNAPSE_AUTH_TOKEN:?set SYNAPSE_AUTH_TOKEN in env.sh}"
  need_python_pkg "synapseclient" "synapseclient"
  SDIR="${ROOT}/brats2021_test"
  mkdir -p "${SDIR}"
  log "syncing syn25829067 -> ${SDIR}"
  python - "${SDIR}" <<'PY'
import os, sys
import synapseclient
from synapseclient import Synapse
syn = Synapse()
syn.login(authToken=os.environ["SYNAPSE_AUTH_TOKEN"])
entity = syn.get("syn25829067", downloadLocation=sys.argv[1])
print("root entity:", entity.name)
PY
fi

log "done. tree:"
du -sh "${ROOT}"/* 2>/dev/null || true
