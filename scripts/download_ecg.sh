#!/usr/bin/env bash
# Domain C — ECG -> laboratory state / structural heart disease.
#
#   --core        PTB-XL v1.0.3 + PTB-XL+ v1.0.1   (open,  ~4 GB)
#   --mimic       MIMIC-IV-ECG v1.0                (open, 33.8 GB)
#   --restricted  EchoNext v1.1.1 + Symile-MIMIC   (signed DUA / credentialed)
#   --all         everything above
#
# Idempotent and resumable. Re-running skips completed files.

set -euo pipefail

: "${DATA_ROOT:?source env.sh first}"
DL_JOBS="${DL_JOBS:-16}"
ROOT="${DATA_ROOT}/ecg"
mkdir -p "${ROOT}"

log() { printf '[ecg %(%H:%M:%S)T] %s\n' -1 "$*"; }
have() { command -v "$1" >/dev/null 2>&1; }

# PhysioNet open datasets are mirrored on S3 under the physionet-open bucket,
# which is far faster than the HTTPS mirror and supports parallel sync.
sync_open() {
  local slug="$1" dest="$2"
  mkdir -p "${dest}"
  if have aws; then
    log "s3 sync ${slug} -> ${dest}"
    aws s3 sync --no-sign-request --only-show-errors \
      "s3://physionet-open/${slug}/" "${dest}/"
  else
    log "aws-cli not found, falling back to wget for ${slug}"
    wget -q --show-progress -r -N -c -np -nH --cut-dirs=3 \
      -P "${dest}" "https://physionet.org/files/${slug}/"
  fi
}

# Restricted datasets have no anonymous S3 mirror; HTTPS with credentials only.
sync_restricted() {
  local slug="$1" dest="$2"
  : "${PHYSIONET_USER:?set PHYSIONET_USER in env.sh}"
  mkdir -p "${dest}"
  log "wget ${slug} -> ${dest} (as ${PHYSIONET_USER})"
  wget -r -N -c -np -nH --cut-dirs=3 \
    --user "${PHYSIONET_USER}" --ask-password \
    -P "${dest}" "https://physionet.org/files/${slug}/"
}

verify() {
  local dir="$1"
  if [[ -f "${dir}/SHA256SUMS.txt" ]]; then
    log "verifying ${dir} (this reads every byte; skip with --no-verify)"
    ( cd "${dir}" && sha256sum -c --quiet SHA256SUMS.txt ) \
      && log "checksums OK: ${dir}" \
      || { log "CHECKSUM FAILURE in ${dir}"; return 1; }
  else
    log "no SHA256SUMS.txt in ${dir}, skipping verification"
  fi
}

do_core=0 do_mimic=0 do_restricted=0 do_verify=1
[[ $# -eq 0 ]] && { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0; }
for arg in "$@"; do
  case "${arg}" in
    --core)       do_core=1 ;;
    --mimic)      do_mimic=1 ;;
    --restricted) do_restricted=1 ;;
    --all)        do_core=1; do_mimic=1; do_restricted=1 ;;
    --no-verify)  do_verify=0 ;;
    *) echo "unknown option: ${arg}" >&2; exit 2 ;;
  esac
done

if (( do_core )); then
  # PTB-XL: 21,799 records, 71 SCP-ECG statements with 0-100 likelihoods.
  # This is the source vocabulary channel — the ANORAK equivalent.
  sync_open "ptb-xl/1.0.3" "${ROOT}/ptb-xl"
  # PTB-XL+: SNOMED-CT mapping with upward ontology propagation, plus the
  # 12SL automatic statements. This is the knowledge-graph layer, prebuilt.
  sync_open "ptb-xl-plus/1.0.1" "${ROOT}/ptb-xl-plus"
  (( do_verify )) && { verify "${ROOT}/ptb-xl" || true; }
  log "core ready. Key files:"
  log "  ${ROOT}/ptb-xl/ptbxl_database.csv      (scp_codes: statement -> likelihood)"
  log "  ${ROOT}/ptb-xl/scp_statements.csv      (71 statements + class hierarchy)"
  log "  ${ROOT}/ptb-xl-plus/ptbxl_statements.csv, SNOMED_description.csv"
fi

if (( do_mimic )); then
  # Open since v1.0 (free-text notes were removed, so credentialing was dropped).
  sync_open "mimic-iv-ecg/1.0" "${ROOT}/mimic-iv-ecg"
  (( do_verify )) && { verify "${ROOT}/mimic-iv-ecg" || true; }
fi

if (( do_restricted )); then
  # EchoNext: 100k ECGs with 12 echocardiography-confirmed SHD labels.
  sync_restricted "echonext/1.1.1" "${ROOT}/echonext"
  # Symile-MIMIC: 11,622 admissions pairing ECG + CXR + 50 blood labs.
  sync_restricted "symile-mimic/1.0.0" "${ROOT}/symile-mimic"
fi

log "done. tree:"
du -sh "${ROOT}"/* 2>/dev/null || true
