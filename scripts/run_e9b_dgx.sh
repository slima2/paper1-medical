#!/usr/bin/env bash
# E9b: five seeds for glioma MGMT and ECG (PTB-XL), then the seed-and-patient
# hierarchical bootstrap, so the cross-domain arms carry the same uncertainty
# treatment as LUAD (five seeds, 2,000 draws).
#
# Run on the DGX from paper1_nature/ after uploading
#   patch_seed_crossdomain.py  bootstrap_crossdomain_contrasts.py  run_e9b_dgx.sh
#     nohup bash run_e9b_dgx.sh > run_e9b.log 2>&1 &
# Idempotent: seeds whose OOF directory is complete are skipped, so it can be
# relaunched after an interruption. Expected wall time: MGMT ~5 min/seed,
# ECG ~50 min/seed on one A100 -> about 4 h in total.
set -uo pipefail
log(){ echo "[e9b] $(date '+%F %T') $*"; }

NB=/home/rapids/notebooks/slima/paper1_nature
B=/home/rapids/notebooks/slima/data/glioma/derived/domainB          # glioma/env.sh DATA_ROOT
C=/home/rapids/notebooks/slima/data/translatability/ecg/derived/domainC   # env.sh DATA_ROOT
SEEDS="43 44 45 46"      # seed42 is the E9 run, moved into the seed layout below
N_BOOT=2000

cd "$NB"
log "patching runners for --seed"
python patch_seed_crossdomain.py || { log "FATAL patch"; exit 1; }

# 0. move the E9 (seed 42) artefacts into the per-seed layout, once
for D in "$B" "$C"; do
  if ls "$D"/oof/oof_*_fold*.csv >/dev/null 2>&1; then
    mkdir -p "$D/oof/seed42" && mv "$D"/oof/oof_*_fold*.csv "$D/oof/seed42/"
    log "moved E9 OOF -> $D/oof/seed42 ($(ls "$D/oof/seed42" | wc -l) files)"
  fi
  for f in results regimes; do
    [ -e "$D/$f.json" ] && [ ! -e "$D/${f}_seed42.json" ] && cp -a "$D/$f.json" "$D/${f}_seed42.json"
  done
  [ -e "$D/choquet_interactions.csv" ] && [ ! -e "$D/choquet_interactions_seed42.csv" ] \
    && cp -a "$D/choquet_interactions.csv" "$D/choquet_interactions_seed42.csv"
done

# 1. MGMT, four more seeds (20 OOF files per complete seed: 4 conditions x 5 folds)
cd "$NB/glioma"; source env.sh 2>/dev/null
for s in $SEEDS; do
  if [ -d "$B/oof/seed$s" ] && [ "$(ls "$B/oof/seed$s" 2>/dev/null | wc -l)" -ge 20 ]; then
    log "MGMT seed $s already complete, skip"; continue
  fi
  log "MGMT seed $s"
  python -u run_glioma_domainB.py --targets mgmt --seed "$s" > "run_domainB_seed$s.log" 2>&1 \
    || { log "FATAL MGMT seed $s (see glioma/run_domainB_seed$s.log)"; exit 1; }
done

# 2. ECG, four more seeds (40 OOF files per complete seed: 4 conditions x 5 fold groups x 2 targets)
cd "$NB"; source env.sh 2>/dev/null
for s in $SEEDS; do
  if [ -d "$C/oof/seed$s" ] && [ "$(ls "$C/oof/seed$s" 2>/dev/null | wc -l)" -ge 40 ]; then
    log "ECG seed $s already complete, skip"; continue
  fi
  log "ECG seed $s"
  python -u run_ecg_domainC.py --seed "$s" > "run_domainC_seed$s.log" 2>&1 \
    || { log "FATAL ECG seed $s (see run_domainC_seed$s.log)"; exit 1; }
done

# 3. hierarchical bootstrap, both arms
log "bootstrap MGMT"
python bootstrap_crossdomain_contrasts.py --oof-dir "$B/oof" --n-boot $N_BOOT | tee "$B/bootstrap_e9b.log"
log "bootstrap ECG"
python bootstrap_crossdomain_contrasts.py --oof-dir "$C/oof" --n-boot $N_BOOT | tee "$C/bootstrap_e9b.log"

log "DONE. Download: $B/bootstrap_{contrasts,aurocs}.csv $B/bootstrap_contrasts.md"
log "               $C/bootstrap_{contrasts,aurocs}.csv $C/bootstrap_contrasts.md"
log "               plus results_seed*.json of both arms (fold means per seed)"
