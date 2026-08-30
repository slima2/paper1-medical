# Cross-domain validation data — download and preparation

Data acquisition for the two pre-registered instantiations of the translatability
framework (Table `tab:crossdomain_protocol` in the manuscript):

- **Domain B — glioma MRI → genotype.** Source ontology: VASARI. Targets: IDH, 1p/19q, MGMT.
- **Domain C — ECG → laboratory / structural heart disease.** Source ontology: SCP-ECG. Targets: 108 lab indicators, 12 SHD labels.

Everything here is designed to run headless on the DGX H100 (Linux x86_64, no sudo required).

---

## Access matrix

| Dataset | Domain | Access | Size | Route |
|---|---|---|---|---|
| [VASARI feature key + scores](https://wiki.cancerimagingarchive.net/display/Public/VASARI+Research+Project) | B | Public | < 5 MB | `fetch_vasari.py` (+ 2 manual CSVs) |
| [TCGA-GBM / TCGA-LGG DICOM](https://www.cancerimagingarchive.net) | B | Public | ~60 GB | `idc-index` |
| [UPENN-GBM](https://www.cancerimagingarchive.net) | B | Public (CC BY 4.0) | ~91 GB | `idc-index` |
| [BraTS 2021 (RSNA-ASNR-MICCAI)](https://www.cancerimagingarchive.net/analysis-result/rsna-asnr-miccai-brats-2021/) | B | Public | 142 GB (TCIA) / ~137 GB (Kaggle) | Kaggle CLI or TCIA Aspera |
| BraTS 2021 sequestered test set | B | Synapse `syn25829067` | — | `synapseclient` |
| [PTB-XL v1.0.3](https://physionet.org/content/ptb-xl/1.0.3/) | C | **Open** | 1.7 GB | `aws s3 --no-sign-request` |
| [PTB-XL+ v1.0.1](https://physionet.org/content/ptb-xl-plus/1.0.1/) | C | **Open** | 2.0 GB | `aws s3 --no-sign-request` |
| [MIMIC-IV-ECG v1.0](https://physionet.org/content/mimic-iv-ecg/1.0/) | C | **Open** (since v1.0) | 33.8 GB | `wget` / `aws s3` |
| [EchoNext v1.1.1](https://physionet.org/content/echonext/) | C | Signed DUA | ~20 GB | `wget --user` |
| [Symile-MIMIC v1.0.0](https://physionet.org/content/symile-mimic/1.0.0/) | C | Credentialed | ~15 GB | `wget --user` |

The three PhysioNet datasets that carry the framework's core claim for domain C
(PTB-XL, PTB-XL+, MIMIC-IV-ECG) are all **open access** — no credentialing, no DUA.
Start there.

---

## Quick start

```bash
cd datasets
cp env.sh.example env.sh          # then edit paths and credentials
source env.sh

# Domain C first: open access, small, and it is where the framework is
# cheapest to falsify (108 targets vs 6).
bash download_ecg.sh --core        # PTB-XL + PTB-XL+  (~4 GB, minutes)
python build_ptbxl_vocabulary.py   # builds the 71-d graded source-vocabulary channel

bash download_ecg.sh --mimic       # MIMIC-IV-ECG (33.8 GB)
bash download_ecg.sh --restricted  # EchoNext + Symile (needs PhysioNet login)

# Domain B
bash download_glioma.sh --vasari   # ontology scores + feature key (small)
python fetch_vasari.py             # normalise into one tidy table
bash download_glioma.sh --idc      # TCGA-GBM / TCGA-LGG / UPENN-GBM DICOM
bash download_glioma.sh --brats    # BraTS 2021 (needs Kaggle API token)
```

Every script is idempotent and resumable; re-running skips what is already present.

---

## Manual steps that cannot be scripted

1. **BraTS 2021 via Kaggle** requires accepting the competition rules once, in a
   browser, at
   <https://www.kaggle.com/c/rsna-miccai-brain-tumor-radiogenomic-classification/rules>.
   After that `kaggle competitions download` works headlessly.
2. **VASARI score CSVs.** TCIA serves these as supporting documents on the
   collection pages rather than through a stable REST path. `fetch_vasari.py`
   attempts discovery and, if that fails, prints the two page URLs and the exact
   filenames to drop into `$DATA_ROOT/glioma/vasari/raw/`. The feature key PDF is
   fetched automatically.
3. **EchoNext / Symile-MIMIC** require a PhysioNet account with a signed data use
   agreement (EchoNext) or credentialed access with CITI training (Symile).

## Disk budget

| Domain | Minimum (framework can be tested) | Full |
|---|---|---|
| C (ECG) | ~4 GB | ~75 GB |
| B (glioma) | ~5 MB + ~60 GB DICOM | ~300 GB |

Plan for **~400 GB** if you pull everything. The `--core` targets alone are under
5 GB and are enough to run the first falsification test.

---

## Domain-C run on PTB-XL alone (`run_ecg_domainC.py`)

`--core` (PTB-XL + PTB-XL+) is sufficient to instantiate the framework in the
ECG domain **without** any credentialed data. The manuscript's protocol target
(108 lab indicators, 12 SHD labels) needs MIMIC-IV-ECG + MIMIC-IV `labevents`
(credentialed) or EchoNext (DUA); those remain pre-registered. What is corrible
now, cleanly and without leakage, is the **physiological-state** target
(sex, age) predicted from the raw 12-lead signal — the direct analogue of the
retinal->systemic mapping the paper cites. The 71 graded SCP-ECG statements are
the source channel `P`; a 1D-CNN embedding of the raw signal is the encoder
channel `E`. Diagnostic super/sub-classes are **not** usable as targets here
because they are aggregations of `P` (source == target leakage).

```bash
source env.sh
pip install torch wfdb scikit-learn pyarrow          # if not already present

python run_ecg_domainC.py --smoke                    # 1 fold, 1500 recs, sanity (~mins)
python run_ecg_domainC.py                            # full: targets sex, age65
python run_ecg_domainC.py --targets sex age65 age75 --epochs 25
```

Conditions mirror the LUAD table: `B2` encoder-only (ceiling), `B3` SCP-only,
`PI` concat fusion, `FC` encoder + 2-additive fuzzy Choquet over the 71
statements. Splits use PTB-XL's 10 patient-disjoint `strat_fold`s grouped into
five held-out folds. Per target the script emits a Conclusive/Inconclusive
verdict (mean AUROC >= 0.70 under the best aggregator) and the PID's three
model-side signatures — exactly what the manuscript reports, rather than numeric
PID atoms. Outputs land in `$DATA_ROOT/ecg/derived/domainC/`
(`results.json`, `regimes.json`, `choquet_interactions.csv`).

---

## Domain-A' run: CPTAC-LUAD cohort-transfer

The within-domain cohort-transfer test: the same six genes and the same LUAD
pipeline, evaluated on CPTAC-LUAD (n=110) instead of TCGA-LUAD, to ask whether
each gene's regime assignment survives a genuine distribution shift. Unlike ECG
this reuses the LUAD model, so it needs the system code (`slima2/lchai`) and the
trained MIL weights + CTransPath checkpoint on the DGX; the scripts here cover
the cohort-agnostic parts (labels, WSIs, evaluation).

```bash
source env.sh

# 1. Labels (small, no credentials) — mutation calls from cBioPortal.
python fetch_cptac_labels.py           # -> $DATA_ROOT/luad_cptac/labels.parquet + prevalence.csv

# 2. WSIs (H&E only; skips the CT that CPTAC-LUAD also ships).
bash download_cptac_wsi.sh --dry-run   # size the pathology (SM) selection first
bash download_cptac_wsi.sh             # -> $DATA_ROOT/luad_cptac/wsi

# 3. Inference: wrap the lchai production entrypoint over the CPTAC WSIs.
#    Clone slima2/lchai on the DGX; point at its apps/inference-service and models/.
python run_cptac_inference.py \
    --lchai-app-dir /path/to/lchai/oncology-xai/apps/inference-service \
    --models-dir    /path/to/lchai/oncology-xai/models \
    --wsi-dir       "$DATA_ROOT/luad_cptac/wsi" --device cuda
#    -> $DATA_ROOT/luad_cptac/cptac_predictions.parquet (patient_id, prob_<gene>)
#    Uses the deployed pipeline verbatim (tiling 224px@level0 / CTransPath /
#    ANORAK / best-method-per-gene MIL). DICOM inputs are converted to pyramidal
#    TIFF via pyvips (needs libvips built against OpenSlide >= 4.0); if that is
#    unavailable, pull the .svs from the GDC and point --wsi-dir at those.

# 4. Evaluate transfer: AUROC + bootstrap 95% CI + regime concordance vs TCGA.
python eval_cptac_transfer.py --preds "$DATA_ROOT/luad_cptac/cptac_predictions.parquet"
#    -> $DATA_ROOT/luad_cptac/transfer_results.{csv,json}

# 5. Aggregation-sensitivity check (no re-inference): slide-level + max/top-k/best
#    from the per-slide predictions, to separate a biology/cohort effect from a
#    pooling artefact.
python reaggregate_cptac.py
#    -> $DATA_ROOT/luad_cptac/reaggregation_results.csv
```

Observed non-silent prevalences (queried live from `luad_cptac_2020`, n=110):
TP53 53.6% (59), EGFR 34.5% (38), KRAS 30.0% (33), STK11 18.2% (20),
KEAP1 10.9% (12), RBM10 4.5% (5). WSIs from IDC are DICOM VL Whole Slide
Microscopy; OpenSlide >= 4.0 reads them directly, else convert with `wsidicom`.

**Outcome (498 slides, 106 genotyped patients with imaging).** In the matched
subset only TP53 clears the n_+=37 floor (n_+=55); EGFR falls just short (36).
The transfer is a clean negative: TP53 collapses from a TCGA AUROC of 0.717 to
0.53 (95% CI 0.42–0.64), and across all six genes and five aggregators (mean,
max, top-2, top-3, largest-slide) plus the slide-level read-out, no CPTAC AUROC
exceeds 0.62 — so it is not a pooling artefact. Read off-the-shelf (no stain
normalisation / domain adaptation), the TCGA verdicts do not transfer.

**Refit outcome (`run_cptac_domainA_v2.py`).** Carrying over only the
representation and re-fitting the four conditions with patient-disjoint CV
inside CPTAC returns TP53 to 0.698 (B2 alone 0.665) against the TCGA 0.717,
so the coupling is present in CPTAC and the off-the-shelf collapse is a
property of the transported decision function, not of the encoder or the
tissue. The field-of-view-matched arm is worse in every condition (B2 0.561,
best 0.632), so scanner magnification is not the mechanism. The regime stays
unresolved in both arms: the deciding contrast PI−B2 is +0.032 (95% CI −0.041
to +0.102) native and +0.073 (−0.038 to +0.174) matched, both spanning zero.
See `tables/tab_cptac_transfer.tex`, `tables/tab_cptac_refit.tex` and
Discussion §CPTAC.
