# CPTAC regime re-read — run plan

Test whether the **three-regime assignment** reproduces on a second cohort, by
retraining the four-condition protocol inside CPTAC-LUAD rather than applying
frozen TCGA weights to it.

---

## 1. Why this run exists

The paper currently makes two claims that sit close together and must not be
conflated:

| Claim | Experiment that establishes it | Status |
|---|---|---|
| The trained **predictor** does not survive the move to CPTAC | `run_cptac_inference.py` + `eval_cptac_transfer.py` | Done (`tab:cptac_transfer`) |
| A **regime assignment** is/is not intrinsic to a gene | *this run* | Not done |

`tab_cptac_transfer` has one AUROC column per gene because the frozen run only
ever produced one number per gene. A regime is read from the sign of contrasts
between B2/B3/PI/FC, so re-reading it requires all four conditions **fitted on
CPTAC**. Until that exists, the Introduction and Discussion can only mark the
question open — which, as of the current draft, they now correctly do.

This run closes it, to the extent the cohort's power allows.

---

## 2. The power problem, stated before running

CPTAC-LUAD: 106 genotyped patients with matched H&E, 498 slides. Positive counts
from `prevalence.csv` (written by `fetch_cptac_labels.py`) against the
`n_+ = 37` estimation floor carried over from TCGA:

| Gene | n₊ | vs floor 37 | Adjudicable? |
|---|---|---|---|
| TP53 | 55 | above | **Yes** |
| EGFR | 36 | at the edge | Borderline |
| KRAS | 32 | below | No |
| STK11 | 20 | below | No |
| KEAP1 | 12 | below | No |
| RBM10 | 5 | below | Not even fitted |

**Only TP53 can settle anything.** The run is still worth doing — a single
well-powered concordance test is exactly what the Discussion says a
single-cohort study cannot supply — but the design has to prevent four
underpowered genes from producing confident-looking labels. Three guards are
built into `run_cptac_domainA_v2.py`:

1. **Pooled out-of-fold AUROC is the point estimate, not the mean of fold
   AUROCs.** With ~21 test patients per fold, KEAP1 puts ~2 positives in a fold,
   where a fold-level AUROC is noise. Per-fold values are still written out.
2. **The headline statistic is a paired bootstrap over patients** on the pooled
   OOF predictions: resample, recompute the four AUROCs, recompute the
   contrasts, and report `p_sign_agrees_with_tcga` per contrast and
   `p_regime_matches_tcga` overall. A regime that reproduces in 52% of
   resamples is a coin flip and prints as one.
3. **Anything below the floor is flagged `powered: false`** in `regimes.json`
   with an explicit caveat string, regardless of how clean the point estimate
   looks. `RBM10` (n₊=5) is not fitted at all.

---

## 3. Prerequisites

Already on the DGX from the transfer run:

- `$DATA_ROOT/luad_cptac/wsi/` — the 498 WSIs (or DICOM series)
- `$DATA_ROOT/luad_cptac/labels.parquet` — `fetch_cptac_labels.py`
- LCHAI checkpoints: `models/backbones/ctranspath.pth`,
  `models/backbones/best_fuzzyarcloss_v2.pth`, `models/checkpoints/`

New and required: the two channels per tile, which the inference run discarded.

---

## 4. Steps

### 4.1 `extract_cptac_features.py` — dump the E and P channels

Runs the identical LCHAI tile pass (same tiling, backbones, optional stain
normalisation) and intercepts the tile-level output instead of the gene scores.
The LCHAI repo is not modified; the hook wraps `_real_tile_inference` exactly as
the stain normaliser already does.

Per slide it writes `emb [T,512]` (CTransPath), `pat [T,6]` (ANORAK) and
`coords [T,2]`.

**Run `--inspect` on one slide first.** The internal return type of the tile
pass is a private detail of your LCHAI version. The script finds the two
channels by shape (trailing dim 512 → embeddings, 6 → patterns), which survives
refactors, but `--inspect` prints what the hook actually received so you can
confirm — or pin fields with `--emb-key` / `--pat-key` if the probe is
ambiguous.

### 4.2 `run_cptac_domainA_v2.py` — the four conditions

- Patient bags: a patient's tiles pooled across their slides, capped at
  `--max-tiles 600` by seeded uniform subsample so an 8-slide patient does not
  dominate a 1-slide patient.
- `E`: frozen CTransPath → 512→128 projection → ABMIL. CTransPath stays frozen,
  so the trainable encoder is small enough for 106 patients.
- `P`: the patient's mean growth-pattern composition over retained tiles — the
  `c_s` of `tab:levels_of_analysis`.
- Conditions B2 / B3 / PI / FC, 5-fold patient-disjoint stratified.
- The Choquet operator is a verbatim port from `run_glioma_domainB.py`, so
  `I_jk` here is directly comparable to the glioma and ECG runs.

---

## 5. Fixed decisions (pre-register before running)

1. **Regime assignment is directional** — sign of the condition contrasts and of
   `I_jk`, never absolute AUROC. Identical rule and thresholds to
   `run_glioma_domainB.py`, so the CPTAC read and the TCGA read come from the
   same function.
2. **The floor is `n_+ = 37`**, carried over unchanged from TCGA. It is not
   re-tuned to make more CPTAC genes look adjudicable.
3. **TP53 is the pre-declared primary test.** Everything else is reported as
   exploratory and labelled underpowered. This is declared now so a favourable
   result on, say, KEAP1 cannot be promoted after the fact.
4. **The architecture deviates from the TCGA run in one documented way**: the
   projection width (512→128) and the frozen backbone are chosen for 106
   patients. Any difference in regime read is therefore confounded with capacity
   and must be reported as such, not attributed to the cohort alone.
5. Both arms are run: raw and Reinhard-normalised features (`--features
   .../features_stain_reinhard`). The stain arm exists because the transfer run
   already showed a colour cast is *not* the cause of the collapse; if the regime
   read differs between arms, that is informative on its own.

---

## 6. Runbook (unattended, `nohup`)

```bash
source ~/cptac_venv/bin/activate
source .../env.sh
cd "$DATA_ROOT/luad_cptac"

LCHAI=/home/rapids/notebooks/slima/lchai/oncology-xai

# 1. confirm the tile-pass hook resolves both channels (one slide, no writes)
python extract_cptac_features.py --inspect --limit 1 \
    --lchai-app-dir "$LCHAI/apps/inference-service" \
    --models-dir "$LCHAI/models" --wsi-dir "$DATA_ROOT/luad_cptac/wsi"

# 2. full extraction (resumable; re-launch continues where it stopped)
LOG="cptac_feat_$(date +%Y%m%d_%H%M%S).log"
nohup python extract_cptac_features.py \
    --lchai-app-dir "$LCHAI/apps/inference-service" \
    --models-dir "$LCHAI/models" --wsi-dir "$DATA_ROOT/luad_cptac/wsi" \
    --device cuda > "$LOG" 2>&1 &
echo "PID $! ; log $(pwd)/$LOG"

# 3. smoke the trainer, then the real run
python run_cptac_domainA_v2.py --smoke --genes TP53
LOG="cptac_regime_$(date +%Y%m%d_%H%M%S).log"
nohup python run_cptac_domainA_v2.py > "$LOG" 2>&1 &
echo "PID $! ; log $(pwd)/$LOG"
```

---

## 7. How each outcome lands in the paper

**TP53 re-reads as regime (i) with a high `p_regime_matches_tcga`.** The
assignment is reproducible across cohorts for the one case we can test. The
Discussion's "an assignment is a property of the triple (ontology, encoder,
cohort) rather than of the gene" can be narrowed to "…except where we tested it,
where it held". The Introduction's open-question clause gets an exception.
Strongest available result.

**TP53 re-reads differently, or `p_regime_matches_tcga` ≈ 0.5.** The assignment
is cohort-relative. The current hedge stands unchanged and is now *measured*
rather than asserted — also publishable, and arguably more interesting, because
it sharpens the difference between framework portability (which glioma and ECG
establish) and assignment portability (which this refutes).

**Everything else comes back unresolvable.** That is the estimation-floor
machinery working: the same instrument that separated KEAP1 from RBM10 in TCGA
declines to adjudicate where it should. Reported as a row of `powered: false`,
not as six regime labels.

### Secondary output, independent of the regime question

The FC condition yields the Choquet interaction matrix on a **second cohort**,
per fold. The paper currently claims the sign of an interaction is more stable
than the AUROC gap it produces, and evidences it with a single case-study slide
(`I(lepidic×solid) = +0.032` on TCGA-55-7815). `choquet_interactions.csv` gives
the per-fold distribution on CPTAC, which is the direct test of that claim — and
it is worth reporting even if every gene comes back underpowered for the regime
read, because it is the load-bearing assumption behind the whole
"directional, not magnitude-based" argument in Methods.

---

## 8. Known limitations to state with the result

- The bootstrap resamples patients over fixed out-of-fold predictions. It
  quantifies sampling noise in the contrasts, **not** training-time variance
  (weight init, fold assignment). Retraining 1000× is not affordable; the
  per-fold spread is the available proxy and is reported alongside.
- 106 patients is small for fitting even a light MIL head. A null result is
  therefore weak evidence against regime portability, whereas a positive
  concordance on TP53 is strong evidence for it. Asymmetric, and to be written
  up as such.
- The pattern channel here is the production ANORAK head, the same one used on
  TCGA. If ANORAK itself degrades on CPTAC stain/scanner characteristics, a
  weak B3 is a channel artefact rather than an empty atom. Compare the raw and
  Reinhard arms before reading anything into a low B3.

---

## 9. Measured: the cohorts differ in scanner magnification

Found while the first extraction was running, and it bears on how the frozen
transfer result can be read. Both pipelines tile at level 0 through a fixed
224px window — CPTAC via LCHAI's `roi_inference` (`level: int = 0`), TCGA via
`SLIMA_PARALELL_inferencing_..._ctranspath_v2.py`
(`read_region((x, y), 0, ...)`, annotated *"Tile size at level 0 — MUST be 224
to match CTransPath training"*). Neither selects a level from the slide's
microns-per-pixel, so a tile spans `224 × mpp` microns, and that span follows
the scanner.

Read from the slide headers (`openslide.mpp-x`):

| cohort | n | 40x (mpp<0.35) | 20x (0.35–0.75) | unknown |
|---|---|---|---|---|
| TCGA-LUAD, the 687 slides with embeddings | 687 | 525 (76.4%) | 155 (22.6%) | 7 (1.0%) |
| CPTAC-LUAD series | 498 | 7 (1.4%) | 491 (98.6%) | 0 |

A 40x tile carries ~56 µm of tissue, a 20x tile ~111 µm: four times the area.
The proportions are close to inverted between fitting and transfer.

Two things follow, and they pull in opposite directions. Magnification is a
confound on reading the frozen-transfer collapse as untranslatability, since
stain normalisation corrects colour and leaves field of view untouched. But
TCGA is itself mixed, and its 22.6% of 20x slides means the encoder was fitted
on both fields rather than on 40x alone — so scale cannot be assumed to explain
the collapse either. Hence the `MATCHMAG=1` arm: it centre-crops each CPTAC tile
to the pixel count spanning `--target-mpp` microns and upsamples it back, which
equalises the field of view and turns the question into a measurement. It is a
test of the scale hypothesis, not a reconstruction of a 40x acquisition — the
crop samples a quarter of each tile's tissue, and the upsampling interpolates
detail the scanner never recorded.

The comment in the TCGA tiler is worth recording separately: pinning 224px at
level 0 was intended to match CTransPath's 20x pretraining, and achieves it only
on the 20x slides. On the 76.4% at 40x the tiles are half the intended linear
scale. This does not disturb the published TCGA AUROCs, which were
cross-validated over exactly that mixture, but it belongs in limitations.
