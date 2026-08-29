#!/usr/bin/env bash
# CPTAC regime probe, TP53 only -- unattended, DGX.
#
# WHY TP53 ONLY
#   TP53 (n_+=55) is the single gene in CPTAC that clears the n_+=37 estimation
#   floor, so it is the only one that can settle anything. Its B2 (encoder-only,
#   refit inside CPTAC) is also the number that decides whether the whole
#   enterprise is viable: the Discussion asserts that a within-CPTAC refit is
#   confounded by an off-distribution encoder, but the evidence for that (the
#   frozen predictor at 0.48) cannot separate a failed MIL head from a failed
#   representation. B2 refit here separates them.
#
#     B2 recovers toward ~0.70  -> the representation is fine, the frozen
#                                  collapse was head-level, the confound does
#                                  not exist, and the full four-condition read
#                                  is licensed.
#     B2 stays at ~0.50         -> either the representation is degenerate on
#                                  CPTAC or n=106 cannot fit even a light head.
#                                  Ambiguous, but measured rather than asserted.
#
#   Asymmetric on purpose: a positive is decisive, a null is not.
#
# USAGE
#   bash run_cptac_tp53_probe.sh
# unattended:
#   LOG="cptac_tp53_$(date +%Y%m%d_%H%M%S).log"
#   nohup bash run_cptac_tp53_probe.sh > "$LOG" 2>&1 & echo "PID $! log $(pwd)/$LOG"
#
# PLUMBING CHECK FIRST (a few minutes instead of hours):
#   SMOKE=1 bash run_cptac_tp53_probe.sh
#   Extracts only $SMOKE_SLIDES slides and runs the trainer on them. This proves
#   the hook, the feature dump and the trainer all work end to end. It is NOT
#   scientifically meaningful -- TP53 needs every patient.
#
# STAIN-NORMALISED ARM (the second arm the run plan asks for):
#   STAIN=1 bash run_cptac_tp53_probe.sh
#
# OVERRIDES
#   VENV=... DATA_ROOT=... LCHAI=... SMOKE=1 SMOKE_SLIDES=12 STAIN=1
#   INFER_DIR=...    dir holding run_cptac_inference.py, if it is not alongside
#                    this script (the extractor imports it)
#   SEARCH_ROOT=...  where to hunt for it automatically (default $HOME)
#   LCHAI_APP=...    the inference app dir, if it is not one of the layouts
#   LCHAI_MODELS=... the models dir, if it is not $LCHAI/models

set -uo pipefail
# Extraction runs for hours under nohup, where python block-buffers stdout and
# the per-slide progress only lands in the log in multi-kilobyte bursts.
export PYTHONUNBUFFERED=1
log(){ echo "[cptac-tp53] $(date '+%F %T') $*"; }
die(){ log "FATAL $*"; exit 1; }

# ---- environment --------------------------------------------------------
VENV="${VENV:-/home/rapids/cptac_venv}"
export DATA_ROOT="${DATA_ROOT:-/home/rapids/notebooks/slima/data/translatability}"
LCHAI="${LCHAI:-/home/rapids/notebooks/slima/paper1_nature}"
SMOKE="${SMOKE:-0}"
SMOKE_SLIDES="${SMOKE_SLIDES:-12}"
STAIN="${STAIN:-0}"
MATCHMAG="${MATCHMAG:-0}"
TARGET_MPP="${TARGET_MPP:-0.25}"
SCRIPTS="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

[ -f "$VENV/bin/activate" ] || die "venv not found at $VENV (set VENV=...)"
source "$VENV/bin/activate"

# extract_cptac_features.py imports run_cptac_inference.py (it reuses the slide
# enumeration, DICOM handling and stain normalisation already validated on this
# cohort). The two do not have to live in the same directory, but the importing
# one has to be on PYTHONPATH. Resolve it here rather than demanding a layout.
cd "$SCRIPTS" || die "cannot cd to $SCRIPTS"
if [ ! -f run_cptac_inference.py ]; then
  CAND="${INFER_DIR:-}"
  if [ -z "$CAND" ]; then
    # Look in the usual places before giving up: a sibling checkout, then a
    # bounded search under the user's notebook tree.
    for d in "$SCRIPTS/../scripts" "$SCRIPTS/.." "$HOME/paper1-medical/scripts"; do
      [ -f "$d/run_cptac_inference.py" ] && { CAND="$(cd "$d" && pwd)"; break; }
    done
  fi
  if [ -z "$CAND" ]; then
    CAND="$(dirname "$(find "${SEARCH_ROOT:-$HOME}" -name run_cptac_inference.py \
             -not -path '*/.git/*' -print -quit 2>/dev/null)" 2>/dev/null)"
  fi
  [ -n "$CAND" ] && [ -f "$CAND/run_cptac_inference.py" ] || die \
"run_cptac_inference.py not found.
  Searched: $SCRIPTS, \$INFER_DIR, sibling dirs, and under \${SEARCH_ROOT:-\$HOME}.
  Fix with either:
    INFER_DIR=/path/to/dir/holding/run_cptac_inference.py bash $(basename "$0")
  or put the whole scripts/ dir together:
    git clone https://github.com/slima2/paper1-medical.git && cd paper1-medical/scripts"
  export PYTHONPATH="$CAND${PYTHONPATH:+:$PYTHONPATH}"
  log "run_cptac_inference.py resolved at $CAND (added to PYTHONPATH)"
fi

for f in extract_cptac_features.py run_cptac_domainA_v2.py; do
  [ -f "$SCRIPTS/$f" ] || die \
"$f is missing from $SCRIPTS.
  This pipeline needs the three CPTAC scripts together. Get them with:
    git clone https://github.com/slima2/paper1-medical.git && cd paper1-medical/scripts"
done

log "python   = $(which python)"
log "DATA_ROOT= $DATA_ROOT"
log "LCHAI    = $LCHAI"
log "app      = ${LCHAI_APP:-<autodetect>}"
log "mode     = $([ "$SMOKE" = 1 ] && echo "SMOKE ($SMOKE_SLIDES slides, plumbing only)" || echo full)$([ "$STAIN" = 1 ] && echo " + reinhard stain" || echo "")$([ "$MATCHMAG" = 1 ] && echo " + field-of-view matched to mpp $TARGET_MPP" || echo "")"

# ---- prerequisites ------------------------------------------------------
# The app directory is named differently depending on how LCHAI was deployed:
# apps/inference-service in the upstream checkout, lchai_app in the flattened
# copy on the DGX. Accept either rather than pinning one layout.
APP="${LCHAI_APP:-}"
if [ -z "$APP" ]; then
  for d in "$LCHAI/apps/inference-service" "$LCHAI/lchai_app" "$LCHAI/inference-service"; do
    [ -d "$d" ] && { APP="$d"; break; }
  done
fi
MODELS="${LCHAI_MODELS:-$LCHAI/models}"
WSI="$DATA_ROOT/luad_cptac/wsi"
[ -n "$APP" ] && [ -d "$APP" ] || die \
"no LCHAI app dir under $LCHAI (tried apps/inference-service, lchai_app, inference-service).
  Set LCHAI=/path/to/lchai/root or LCHAI_APP=/path/to/the/app/dir directly."
[ -d "$MODELS" ] || die "missing LCHAI models dir: $MODELS (set LCHAI_MODELS=...)"
[ -d "$WSI" ]    || die "missing WSI dir: $WSI (is DATA_ROOT right?)"
for f in "$MODELS/backbones/ctranspath.pth" \
         "$MODELS/backbones/best_fuzzyarcloss_v2.pth" \
         "$MODELS/checkpoints"; do
  [ -e "$f" ] || die "missing checkpoint path: $f"
done

if [ ! -e "$DATA_ROOT/luad_cptac/labels.parquet" ]; then
  log "labels.parquet missing -> fetching from cBioPortal"
  python fetch_cptac_labels.py || die "fetch_cptac_labels"
fi

# ---- stain arm ----------------------------------------------------------
STAIN_ARGS=()
FEATURES="$DATA_ROOT/luad_cptac/features"
if [ "$STAIN" = 1 ]; then
  REF="$SCRIPTS/tcga_luad_reference.png"
  [ -f "$REF" ] || die "stain reference not found: $REF (run make_stain_reference.py)"
  STAIN_ARGS=(--stain-normalize --stain-method reinhard --stain-target "$REF")
  FEATURES="$DATA_ROOT/luad_cptac/features_stain_reinhard"
fi

# ---- field-of-view arm ---------------------------------------------------
# TCGA-LUAD is 83% 40x and CPTAC-LUAD 99% 20x, and both pipelines tile at
# level 0 with a fixed 224px window, so a CPTAC tile carries four times the
# tissue area of the tiles the model was fitted on. This arm rescales each tile
# to the same physical field before the encoder sees it.
MAG_ARGS=()
if [ "$MATCHMAG" = 1 ]; then
  MAG_ARGS=(--match-mag --target-mpp "$TARGET_MPP")
  FEATURES="${FEATURES}_mag${TARGET_MPP}"
fi

COMMON=(--lchai-app-dir "$APP" --models-dir "$MODELS" --wsi-dir "$WSI" --device cuda)

# ---- 1. resolve the tile-pass contract on one slide ---------------------
# The return type of _real_tile_inference is private to your LCHAI build, so
# the extractor locates the two channels by shape. This prints what the hook
# actually received; if it cannot resolve them it dies here rather than after
# hours of extraction.
log "1/3 inspect: resolving the E/P channels on one slide"
python extract_cptac_features.py --inspect --limit 1 "${COMMON[@]}" "${STAIN_ARGS[@]}" "${MAG_ARGS[@]}" \
  || die "channel resolution failed -- read the structure dump above and pin the fields with --emb-key/--pat-key"

# ---- 2. extract the two channels for the cohort -------------------------
# Resumable: relaunching continues from the manifest.
# Roughly half the 498 series belong to patients cBioPortal has no genotype
# for. Those slides cannot enter a bag, so tiling them is several hours of GPU
# spent on rows the trainer will drop.
EXTRA=(--only-labeled)
if [ "$SMOKE" = 1 ]; then EXTRA+=(--limit "$SMOKE_SLIDES"); fi
log "2/3 extracting features -> $FEATURES"
python extract_cptac_features.py "${COMMON[@]}" "${STAIN_ARGS[@]}" "${MAG_ARGS[@]}" "${EXTRA[@]}" \
  || die "feature extraction"

# ---- 3. the four conditions, TP53 only ----------------------------------
log "3/3 four-condition run on TP53"
python run_cptac_domainA_v2.py --features "$FEATURES" --genes TP53 --smoke \
  || die "trainer smoke"
if [ "$SMOKE" != 1 ]; then
  python run_cptac_domainA_v2.py --features "$FEATURES" --genes TP53 \
    || die "trainer full"
fi

OUT="$DATA_ROOT/luad_cptac/derived/domainA_v2"
log "DONE. results in $OUT"
log "  regimes.json           <- B2 vs TCGA 0.718 is the number to read first"
log "  contrast_bootstrap.csv <- p_sign_agrees_with_tcga per contrast"
log "  choquet_interactions.csv"
if [ "$SMOKE" = 1 ]; then
  log "NOTE: SMOKE run -- $SMOKE_SLIDES slides only. Plumbing verified, results meaningless."
  log "Re-run without SMOKE=1 for the real probe."
fi
