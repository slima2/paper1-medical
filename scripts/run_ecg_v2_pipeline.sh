#!/usr/bin/env bash
# ECG domain-C v2 pipeline (MIMIC-IV-ECG lab targets, IPW) -- unattended.
# Run on the DGX. Edit VENV / DATA_ROOT / TARGETS below if your paths differ.
#
#   bash run_ecg_v2_pipeline.sh
# or unattended:
#   LOG="ecg_v2_$(date +%Y%m%d_%H%M%S).log"
#   nohup bash run_ecg_v2_pipeline.sh > "$LOG" 2>&1 & echo "PID $! log $(pwd)/$LOG"

set -uo pipefail
log(){ echo "[ecg-v2] $(date '+%F %T') $*"; }

# ---- environment (edit to taste) ----------------------------------------
VENV="${VENV:-/home/rapids/cptac_venv}"
export DATA_ROOT="${DATA_ROOT:-/home/rapids/notebooks/slima/data}"
TARGETS="${TARGETS:-ntprobnp troponin_i}"
WINDOW_H="${WINDOW_H:-24}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# activate venv
if [ -f "$VENV/bin/activate" ]; then source "$VENV/bin/activate"; else
  log "FATAL venv not found at $VENV (set VENV=...)"; exit 1; fi
cd "$SCRIPTS"
log "python=$(which python)  DATA_ROOT=$DATA_ROOT  targets=[$TARGETS]"

D="$DATA_ROOT/ecg/derived"

# ---- prerequisites ------------------------------------------------------
RL="$DATA_ROOT/ecg/mimic4/waveforms/record_list.csv"
LE="$DATA_ROOT/ecg/mimic4/hosp/labevents.csv.gz"
DLI="$DATA_ROOT/ecg/mimic4/hosp/d_labitems.csv.gz"
for f in "$RL" "$LE" "$DLI"; do
  [ -e "$f" ] || { log "FATAL missing $f -- download MIMIC-IV-ECG + hosp (credentialed PhysioNet)"; exit 1; }
done
if [ ! -e "$D/scp_membership.parquet" ]; then
  log "scp_membership.parquet missing -> building PTB-XL vocabulary"
  python build_ptbxl_vocabulary.py || { log "FATAL build_ptbxl_vocabulary (need PTB-XL under \$DATA_ROOT/ecg/ptb-xl)"; exit 1; }
fi

# ---- 1. labels (pair ECG<->lab, binarise, observed mask) ----------------
log "1/4 build_ecg_lab_labels (window ${WINDOW_H}h)"
python build_ecg_lab_labels.py --targets $TARGETS --window-h "$WINDOW_H" \
  || { log "FATAL build_ecg_lab_labels"; exit 1; }

# ---- 2. propagate SCP statements PTB-XL -> MIMIC ------------------------
log "2/4 propagate_scp_to_mimic (smoke then full)"
python propagate_scp_to_mimic.py --smoke || { log "FATAL propagate smoke"; exit 1; }
python propagate_scp_to_mimic.py         || { log "FATAL propagate full"; exit 1; }

# ---- 3. IPW weights -----------------------------------------------------
log "3/4 fit_lab_propensity"
python fit_lab_propensity.py --targets $TARGETS || { log "FATAL fit_lab_propensity"; exit 1; }

# ---- 4. the run ---------------------------------------------------------
log "4/4 run_ecg_domainC_v2 (smoke then full)"
python run_ecg_domainC_v2.py --smoke --targets $TARGETS || { log "FATAL run smoke"; exit 1; }
python run_ecg_domainC_v2.py --targets $TARGETS         || { log "FATAL run full"; exit 1; }

log "DONE. results in $D/domainC_v2/  (results_v2.json, regimes_v2.json)"
log "guardians: $D/propagation_quality.csv  $D/propensity_quality.csv"
