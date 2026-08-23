# VASARI / Glioma (Domain B) — Run Plan

Pre-registered instantiation of the translatability framework in a third
modality: **preoperative multiparametric MRI → glioma genotype**. This is the
glioma counterpart of the TCGA-LUAD paper and of `run_ecg_domainC.py`. The
framework, the four-condition protocol, the 2-additive fuzzy Choquet operator,
the Conclusive/Inconclusive verdict and the model-side regime signature are
**identical across domains** — only the source channel `P` and the encoder
channel `E` change.

Reference: `tables/tab_crossdomain_protocol.tex` (column B) is the
pre-registration; nothing below overrides it.

---

## 0. What is already locked (do not change)

From `tab:crossdomain_protocol`, column B:

| Element | Fixed value |
|---|---|
| Source modality | Preoperative multiparametric MRI (T1, T1c, T2, FLAIR) |
| Targets | **IDH, 1p/19q, MGMT** |
| Source ontology | **VASARI**, ~30 semantic MR features |
| Vocabulary renderer | VASARI-feature classifier on the TCIA-annotated seed (**178 LGG + 75 GBM**) |
| Propagated to | **~1,251 BraTS 2021 subjects** |
| Target lexicon (KG) | NCIt + MONDO + **RadLex** |
| Regime (i) saturation | **IDH** (AUROC ≈ 0.88) |
| Regime (ii) synergy | **1p/19q** (feature combinations) |
| Regime (iii) out-of-ontology | **MGMT** (≈ 0.50–0.56 across architectures) |
| Falsification test | Non-additive aggregator wins on **1p/19q but not IDH or MGMT** |
| Access | TCIA/IDC + BraTS 2021, public |

**MGMT is the make-or-break claim** (`04_discussion.tex`, ~L467): it must come
out as an **empty VASARI-side atom** (out-of-ontology), *not* a data floor. The
run distinguishes the two exactly as LUAD separated KEAP1 (empty) from RBM10
(floor): via the model-side signatures `B3-vs-B2` and the Choquet interactions,
not a single AUROC.

---

## 1. Fixed design decisions (v1)

These are the choices made in `run_glioma_domainB.py`; documented here so the
DGX run is reproducible and the rationale is on record.

- **Encoder `E` = 2.5D CNN + ABMIL.** Each subject is a **bag of axial
  slices**; a 2D CNN (`SliceEncoder`) embeds each 4-sequence slice and gated
  ABMIL pools the bag. Chosen over full-3D for **maximal methodological
  consistency with LUAD** (tile-bag + ABMIL), lower compute, and a free L1
  "where" read-out (slice-level attention). Full-3D is a later ablation, not v1.
- **Sequences = 4 co-registered BraTS channels** (T1, T1c, T2, FLAIR) as input
  channels `C = 4`.
- **Slice selection = K=24 axial slices** centred on the tumour centroid
  (from the BraTS segmentation mask). Fixed K → fixed `[K, C, H, W]` tensors,
  so batching is trivial. `SLICE_HW = 128` in-plane crop.
- **Source channel `P` = ~30 VASARI features in [0,1]**, ordinal expert scores
  min-max scaled per feature, propagated to every BraTS subject.
- **Four conditions** (identical to LUAD/ECG): `B2` encoder-only (ceiling),
  `B3` VASARI-only, `PI` concat fusion, `FC` encoder + 2-additive Choquet over
  the VASARI features.
- **Splits = 5-fold patient-stratified**, built once and frozen in the manifest.
- **Verdict threshold = AUROC ≥ 0.70** under the best condition.
- **Seed = 42.**

---

## 1b. v1 scope: TCGA-LGG-centric (the path we are running)

The full BraTS-centric design above (propagation to ~1,251 subjects, MGMT) is
retained as the target v2. **v1 is a lighter, cleaner instantiation** that
reuses the same four-condition machinery but avoids the two hardest data joins
(the BraTS↔TCIA crosswalk and cBioPortal), because everything is already keyed
by TCGA id:

- **Cohort = the ~187 TCGA-LGG subjects** that carry **real VASARI scores**
  (`TCGA_vasari_INFO.csv`, 188 rows) intersected with molecular calls.
- **Source channel `P` = the real VASARI expert scores** (26 features), min-max
  rescaled to [0,1] in `run_glioma_domainB.py`. **No propagation** — these are
  ground-truth annotations, so `propagate_vasari.py` is skipped in v1.
- **Targets = IDH and 1p/19q**, read straight from `TCGA_clinical_INFO.csv`
  (`IDH1`, `1p_19q_co_del_status`). Measured power on the 187:
  **IDH n₊=136/186, 1p/19q n₊=49/184** — both clear the ~37 floor.
  (Bonus binary target available: **sex** from `gender`.)
- **MGMT is deferred to v2** (absent from this clinical source; needs BraTS).
  `run_glioma_domainB.py` reports it *underpowered/unavailable → Inconclusive*
  rather than failing.
- **Encoder input = BraTS-TCGA-LGG NIfTI** (Bakas 2017): skull-stripped,
  co-registered T1/T1-Gd/T2/FLAIR + expert segmentation, **named by TCGA id**
  → direct join to VASARI and labels, no crosswalk. Public training arm = **65
  LGG subjects**; the test arm (43) is helpdesk-gated. So the imaging∩VASARI
  cohort is ~65 in v1: IDH stays well-powered; **1p/19q may fall below the
  floor on 65** — if so the framework correctly returns *Inconclusive*, and the
  full-power 1p/19q read moves to v2 (add BraTS-TCGA-GBM + test arm).

This v1 hits **2 of the 3 pre-registered regimes** (IDH = i, 1p/19q = ii) with
zero crosswalk risk; v2 adds MGMT (iii) and full 1p/19q power.

---

## 2. Data contract (what the run expects on disk)

All under `$DATA_ROOT/glioma/`. The run is fully defined; only these artefacts
are stubbed and must be produced first.

| Artefact | Path | Producer | Used in |
|---|---|---|---|
| VASARI real scores | `vasari/vasari_features.parquet` | `fetch_vasari.py` | v1 + v2 |
| Clinical calls (IDH/1p19q/sex/age) | `vasari/TCGA_clinical_INFO.csv` | download (step 1a) | v1 |
| VASARI propagated | `vasari/vasari_propagated.parquet` | `propagate_vasari.py` | v2 only |
| Genotype labels | `derived/glioma_labels.parquet` | `build_glioma_labels.py` | v1 + v2 |
| Subject manifest | `derived/manifest.parquet` | `prepare_glioma_slices.py` | v1 + v2 |
| Per-subject slices | `derived/slices/<subject>.npy` `[K,C,H,W]` f32 | `prepare_glioma_slices.py` | v1 + v2 |

In **v1** the source channel `P` is read directly from `vasari_features.parquet`
(min-max rescaled in the run); `vasari_propagated.parquet` is a **v2** artefact.

All five producer scripts now exist under `scripts/`. Two artefacts still
need manual acquisition before they run (see the runbook, §3): the
BraTS↔TCIA **crosswalk XLSX** and the **`tcga_genotype.csv`** (IDH/1p19q),
which are the only pieces that cannot be scripted end-to-end.

Column contracts (enforced by the loaders in `run_glioma_domainB.py`):
- `manifest.parquet`: `subject_id`, `strat_fold` (1..5), `slices_path`.
- `vasari_propagated.parquet`: index `subject_id`, ~30 feature columns in [0,1].
- `glioma_labels.parquet`: index `subject_id`, columns `idh`, `onep19q`,
  `mgmt` (1/0, NaN allowed → dropped per target).

---

## 2b. DGX runbook — v2 / BraTS-centric (copy-paste)

> This is the full BraTS-centric path (propagation + MGMT). For what we are
> actually running now, use **§2c (v1, TCGA-LGG-centric)** below.

```bash
# 0. environment
cd scripts
source env.sh                 # sets DATA_ROOT (+ any tokens)
pip install torch nibabel wfdb pandas pyarrow scikit-learn requests idc-index kaggle

# 1. data (idempotent; accept Kaggle rules once in a browser)
bash download_glioma.sh --vasari      # VASARI feature key + score CSVs
bash download_glioma.sh --brats       # BraTS 2021 imaging + train_labels.csv (MGMT)
#   also drop by hand into $DATA_ROOT/glioma/ :
#     - the BraTS<->TCIA crosswalk XLSX  (TCIA BraTS 2021 page)
#     - tcga_genotype.csv  with columns  tcga_id,idh,onep19q
#       (from cBioPortal lgg_tcga + gbm_tcga; or try:
#        python build_glioma_labels.py --fetch-cbioportal )

# 2. source channel P (seed table -> classifier -> full-cohort propagation)
python fetch_vasari.py                 # -> vasari/vasari_features.parquet

# 3. encoder input + folds  (writes derived/slices/*.npy and manifest.parquet)
python prepare_glioma_slices.py        # add --limit 20 for a quick smoke first

# 4. targets  (-> derived/glioma_labels.parquet ; prints n_+ power check)
python build_glioma_labels.py

# 5. propagate VASARI to all subjects (needs manifest + labels for folds)
python propagate_vasari.py             # -> vasari/vasari_propagated.parquet
#   inspect vasari/propagation_quality.csv; optionally --min-auroc 0.6

# 6. the run
python run_glioma_domainB.py --smoke   # 1 fold, 3 epochs, sanity
python run_glioma_domainB.py           # full: idh onep19q mgmt, 5 folds
#   -> derived/domainB/{results.json,regimes.json,choquet_interactions.csv}
```

Dependency order is strict: `prepare_glioma_slices.py` and
`build_glioma_labels.py` must both run before `propagate_vasari.py` (which
needs the manifest and uses MGMT for stratification), and all three before
`run_glioma_domainB.py`. Re-runs are incremental (slices are cached).

**Power gate before believing MGMT:** step 4 prints `n_+` per target.
If MGMT clears the floor (`n_+ ≳ 37`) a near-chance result is a real
out-of-ontology finding; if it does not, it is a power statement, not a
regime verdict — do not report the make-or-break MGMT claim until the gate
is passed.

## 2c. DGX runbook — v1 / TCGA-LGG-centric (copy-paste, the one we run)

Real VASARI + clinical calls, keyed by TCGA id. **No propagation, no crosswalk,
no cBioPortal.** `build_glioma_labels.py` and `run_glioma_domainB.py`
auto-detect this mode when the clinical CSV and real VASARI table are present.

```bash
cd scripts && source env.sh          # DATA_ROOT set; venv active (numpy 1.26.4)

# 1a. VASARI scores + feature key + clinical  (already downloaded in step 1a)
#   $DATA_ROOT/glioma/vasari/raw/TCGA_vasari_INFO.csv
#   $DATA_ROOT/glioma/vasari/raw/VASARI_MR_featurekey.pdf
#   $DATA_ROOT/glioma/vasari/TCGA_clinical_INFO.csv     (IDH/1p19q/sex/age)
python fetch_vasari.py               # -> vasari/vasari_features.parquet (188 x 26)

# 1b. imaging: BraTS-TCGA-LGG *processed NIfTI* (65 subj, folders named TCGA-XX-XXXX
#     with *_t1 *_t1Gd *_t2 *_flair + *ManuallyCorrected*/*GlistrBoost* seg).
#     Source = TCIA Aspera Faspex, Training set (zip, 536 MB):
#       https://faspex.cancerimagingarchive.net/aspera/faspex/external_deliveries/89?passcode=48c6c69da7d44989c20f6cf366b7103e19565cfc
#     Get the zip via the Faspex "Download" button (HTTP fallback) or aspera-cli,
#     then unzip into $DATA_ROOT/glioma/brats_tcga_lgg/ :
mkdir -p "$DATA_ROOT/glioma/brats_tcga_lgg"
# unzip <downloaded.zip> -d "$DATA_ROOT/glioma/brats_tcga_lgg"

# 2. labels from the clinical CSV  (auto: clinical-csv mode)
python build_glioma_labels.py        # -> derived/glioma_labels.parquet
#   prints: idh n_+=~136, onep19q n_+=~49 over the VASARI cohort; mgmt = NaN

# 3. encoder input + folds  (auto-detects TCGA-* NIfTI, t1Gd, ManuallyCorrected)
python prepare_glioma_slices.py --limit 8    # smoke first
python prepare_glioma_slices.py              # all -> derived/manifest.parquet
#   NB: manifest∩labels∩VASARI is what actually runs (~65 with public imaging)

# 4. the run  (real VASARI auto-used; MGMT auto-skipped as underpowered)
python run_glioma_domainB.py --smoke --targets idh onep19q
python run_glioma_domainB.py --targets idh onep19q sex
#   -> derived/domainB/{results.json,regimes.json,choquet_interactions.csv}
```

`propagate_vasari.py` is **not used in v1** (VASARI here is ground truth, not
propagated). If 1p/19q comes back Inconclusive at n≈65, that is the correct
framework output; the full-power 1p/19q read is a v2 item (add BraTS-TCGA-GBM
and the helpdesk-gated LGG test arm).

---

## 2d. DGX runbook — MGMT v2 (the make-or-break regime-(iii) test)

This is the run that removes the "MGMT deferred" caveat. MGMT is not in the
TCGA clinical CSV, so v1 could not test it; the MGMT status lives in the
**BraTS 2021 / RSNA-MICCAI** classification cohort (~585 subjects, ~53%
methylated → both classes clear the floor by a wide margin). The four-condition
protocol is unchanged; only the data joins differ.

**Why this needs propagation (unlike v1).** Real VASARI scores exist only for
the ~188 TCGA seed subjects. The MGMT cohort is BraTS2021_XXXXX, which has *no*
real VASARI, so the B3 (VASARI-only) and FC (Choquet) arms need VASARI
**propagated** onto those subjects — that is exactly what tells an *empty
VASARI-side atom* (out-of-ontology, the prediction) apart from a *data floor*.
Propagation now maps the seed by **identity** to the BraTS-TCGA-LGG imaging you
already have, so **no BraTS↔TCIA crosswalk is required**.

### What to acquire (two artefacts)

1. **NIfTI BraTS 2021 imaging with segmentation** — the 4 co-registered
   sequences (`*_t1 *_t1ce *_t2 *_flair`) plus `*_seg`, folders named
   `BraTS2021_XXXXX`. **Not** the RSNA-MICCAI *DICOM* competition set (it has no
   segmentation, which `prepare_glioma_slices.py` needs for the tumour
   centroid). The Kaggle NIfTI mirror is the practical route
   (e.g. `dschettler8845/brats2021-task1`; verify the folder layout after
   unzip). ~15 GB.
2. **MGMT labels** — `train_labels.csv` (`BraTS21ID`, `MGMT_value`) from the
   RSNA-MICCAI competition; a tiny CSV.

```bash
cd scripts && source env.sh          # DATA_ROOT set; venv active (numpy 1.26.4)
#   (kaggle CLI needs ~/.kaggle/kaggle.json; accept each page's terms once in a
#    browser the first time.)

# 1. imaging: NIfTI BraTS 2021 Task 1 (with *_seg). Kaggle DATASET (not the
#    competition): dschettler8845/brats-2021-task1. Ships BraTS2021_Training_Data.tar
#    with BraTS2021_XXXXX/{_flair,_t1,_t1ce,_t2,_seg}.nii.gz (matches the loader).
mkdir -p "$DATA_ROOT/glioma/brats2021_nifti"
kaggle datasets download -d dschettler8845/brats-2021-task1 \
  -p "$DATA_ROOT/glioma/brats2021_nifti"
cd "$DATA_ROOT/glioma/brats2021_nifti"
unzip -n brats-2021-task1.zip                 # -> BraTS2021_Training_Data.tar (+ val)
tar -xf BraTS2021_Training_Data.tar           # -> BraTS2021_XXXXX/ folders (~1,251)
cd -   # back to scripts
#   ~15 GB zip, ~40 GB extracted; only the ~585 MGMT-labelled subjects are used
#   by the run, but prepare processes every folder it finds.

# 2. MGMT labels (tiny CSV) into the path load_mgmt() expects:
mkdir -p "$DATA_ROOT/glioma/brats2021"
kaggle competitions download -c rsna-miccai-brain-tumor-radiogenomic-classification \
  -f train_labels.csv -p "$DATA_ROOT/glioma/brats2021"
( cd "$DATA_ROOT/glioma/brats2021" && unzip -n train_labels.csv.zip 2>/dev/null || true )
#   -> $DATA_ROOT/glioma/brats2021/train_labels.csv  (585 rows: BraTS21ID, MGMT_value)

# 3. slices for BOTH cohorts in one manifest: the TCGA seed (v1 imaging,
#    named by TCGA id → propagation trainer) and the BraTS2021_* target cohort.
#    prepare auto-detects both patterns; folds stratify on MGMT when present.
python prepare_glioma_slices.py --limit 8        # smoke
python prepare_glioma_slices.py                  # all (~1,251 subj; heavy I/O)

# 4. labels indexed by BraTS2021_* with MGMT populated (force crosswalk mode so
#    it does NOT fall back to the clinical CSV, which lacks MGMT). IDH/1p19q stay
#    NaN here — fine, the MGMT run only needs mgmt.
python build_glioma_labels.py --source crosswalk
#   prints  mgmt n_+≈307  labelled≈585  OK

# 5. propagate VASARI onto every manifest subject (seed = TCGA real scores,
#    identity-mapped to their imaging; predicts for the BraTS2021_* cohort).
#    Inspect vasari/propagation_quality.csv BEFORE trusting the B3 arm.
python propagate_vasari.py --smoke               # sanity
python propagate_vasari.py                        # -> vasari/vasari_propagated.parquet

# 6. the run: MGMT only (full power)
python run_glioma_domainB.py --smoke --targets mgmt
python run_glioma_domainB.py --targets mgmt       # 5 folds, 4 conditions
#   -> derived/domainB/{results.json,regimes.json,choquet_interactions.csv}
```

### How to read the outcome (regime iii vs data floor vs propagation artefact)

MGMT clears the power floor here (n_+ ≈ 307, n_− ≈ 278 ≫ 37), so a near-chance
result is a **finding**, not a power statement. Read the signatures, not one
AUROC:

| Observed | Verdict |
|---|---|
| B2 ≈ B3 ≈ PI ≈ FC ≈ 0.50–0.56, and `propagation_quality.csv` shows VASARI **is** recoverable from MRI (so B3-at-chance is not a broken channel) | **regime (iii), empty VASARI-side atom** — the pre-registered untranslatability, now *reported as a finding*. This is the result that removes "deferred". |
| B3 > B2 (VASARI beats the encoder) at chance-level B2 | **data floor / pattern-side unique**, a different sub-state — report as such, not as out-of-ontology |
| B3 at chance **but** propagation quality is poor (features not recoverable) | **inconclusive B3** — the empty-atom claim is not established; fix propagation or fall back to real-VASARI subjects only |

That third row is the one caveat this design introduces: propagated VASARI can
be weak, and a weak channel looks like an empty atom. `propagation_quality.csv`
(held-out per-feature MAE/AUROC on the seed) is the guard — cite it alongside
the MGMT verdict.

**Caveat on VASARI source once propagation exists.** `load_vasari_channel`
prefers `vasari_propagated.parquet` over the real scores, so after step 5 the
v1 IDH/1p19q run would silently switch to *propagated* VASARI. Keep the MGMT v2
run under a **separate `DATA_ROOT`** (or move/rename `vasari_propagated.parquet`
away) if you want to reproduce the v1 real-VASARI numbers.

### Compute

Encoder × 4 conditions × 5 folds on ~585 MGMT subjects × 24 slices, plus one
propagation trainer. A few hours on one H100; slices are cached so re-runs are
cheap.

---

## 3. Execution plan (ordered)

### Step 1 — Acquire raw data
- **BraTS 2021** imaging (~1,251 subjects, 4 sequences + segmentation) from
  TCIA/IDC (`download_glioma.sh`).
- **VASARI score CSVs** (TCGA-GBM/LGG) → seed for `fetch_vasari.py`
  (`download_glioma.sh --vasari`; manual drop into `vasari/raw/` per its
  instructions).
- **BraTS↔TCIA crosswalk** XLSX (BraTS 2021 TCIA page) to resolve BraTS ids to
  canonical TCGA/TCIA `subject_id`.
- **Genotype calls**: IDH mutation and 1p/19q codeletion from GDC/TCGA
  (cBioPortal `lgg_tcga` / `gbm_tcga`); **MGMT methylation** from the BraTS 2021
  MGMT table (RSNA-MICCAI) and/or TCGA methylation.

### Step 2 — `propagate_vasari.py` (source channel P) — **TODO**
- Train the VASARI-feature classifier on the annotated seed
  (`vasari_features.parquet`, 253 cases) — the ANORAK→LUAD analogue.
- Input = the SAME 2.5D slice embedding as `E` (or an independent MRI encoder);
  output = ~30 VASARI feature scores.
- **Propagate** to all BraTS subjects → `vasari_propagated.parquet`.
- **Report propagation quality** (held-out per-feature AUC/MAE on the seed);
  weak propagation contaminates `B3` and must be flagged.

### Step 3 — `prepare_glioma_slices.py` (encoder input) — **TODO**
- For each subject: load 4 co-registered sequences + segmentation, z-score each
  sequence within the brain mask, pick K=24 axial slices around the tumour
  centroid, centre-crop to 128², stack → `[K, 4, 128, 128]` float32 `.npy`.
- Build 5-fold patient-stratified `strat_fold` (stratify on the scarcest target,
  MGMT) and write `manifest.parquet`.

### Step 4 — `build_glioma_labels.py` (targets) — **TODO**
- Join IDH / 1p19q / MGMT calls to `subject_id`; write
  `glioma_labels.parquet`. Leave missing calls NaN.
- **Power check first** (LUAD/CPTAC lesson): report `n_+` per target on the
  imaging∩genotype intersection. Only targets clearing the estimation floor
  (n_+ ≳ 37, as in LUAD) yield a Conclusive-capable read; below that a
  near-chance MGMT is a **power** statement, not out-of-ontology. This gate is
  what protects the make-or-break MGMT claim.

### Step 5 — Run
```bash
source env.sh
python run_glioma_domainB.py --smoke          # sanity: 1 fold, 3 epochs
python run_glioma_domainB.py                  # full: idh onep19q mgmt, 5 folds
```
Outputs under `$DATA_ROOT/glioma/derived/domainB/`: `results.json`,
`regimes.json`, `choquet_interactions.csv`.

### Step 6 — Paper integration
- New table `tab_glioma_domainB` mirroring `tab:mutation_luad` (per-target,
  per-condition AUROC + verdict + signature).
- Confirm/deny the three pre-registered regime predictions and the falsification
  test (non-additive wins on 1p/19q only).

---

## 4. How the protocol maps to the pre-registered regimes

For each target, the regime is read from the **contrast** across conditions
(never a single head):

| Target | Predicted regime | Expected signature | Reads as |
|---|---|---|---|
| **IDH** | (i) saturation | `B2 ≈ best`, `FC/PI ≈ B2`, `B3` weaker | `encoder saturation` |
| **1p/19q** | (ii) synergy | `FC/PI > B2` (>+0.015), positive Choquet pair | `synergy (fusion beats encoder)` — **the only target where the non-additive aggregator should win** |
| **MGMT** | (iii) out-of-ontology | all conditions ≈ chance (0.50–0.56); `B3` at chance and fusion no help | `encoder saturation`-style but at chance = **empty atom**. If instead `B3 > B2` at low `n_+` → `pattern-side unique/floor` = **data floor**, a different sub-state to be flagged via SD vs effect size |

The MGMT branch is the whole point: the run must be able to say **empty vs
unestimable**, which is why it re-instantiates the four-condition protocol
rather than reapplying frozen weights (cf. the CPTAC model-transfer test, which
by construction cannot produce a regime).

---

## 5. Risks and open decisions

1. **Encoder design (largest unknown).** 2.5D+ABMIL is the v1 choice for
   consistency; 3D CNN is a pre-planned ablation if 2.5D underfits IDH
   (which should saturate near 0.88 — a low IDH ceiling would indicate an
   encoder problem, not a translatability result).
2. **Small annotated seed (n=253)** for the VASARI classifier → noisy
   propagation. Mitigation: report seed held-out propagation quality; consider
   feature-wise dropping of poorly propagated VASARI items from `P`.
3. **Label availability on imaging∩genotype** may shrink effective `n_+`,
   especially for MGMT — the exact CPTAC-style power risk that could
   masquerade as out-of-ontology. Step 4 power check is mandatory before any
   MGMT claim.
4. **Compute:** encoder × 3 targets × 4 conditions × 5 folds. Manageable on one
   H100 with 2.5D; batch size 8 subjects × 24 slices.
5. **Global norm / leakage:** per-fold intensity normalisation is preferable to
   a global one; v1 uses per-subject within-mask z-score (leakage-free), matching
   the ECG note about per-fold vs global stats.

---

## 6. Effort estimate

| Task | Estimate |
|---|---|
| Data + genotype labels (download + GDC/BraTS join) | 1–2 days |
| `propagate_vasari.py` (classifier + propagation + QA) | 2–3 days |
| `prepare_glioma_slices.py` (preprocessing + manifest) | 1–2 days |
| Runs + analysis + `tab_glioma_domainB` | 2 days |

`run_glioma_domainB.py` itself is done (this scaffold); only the four
preprocessing artefacts in §2 remain.
