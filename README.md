# A translatability verdict for cross-domain medical prediction that reports what a scarce cohort can adjudicate

Manuscript source, tables, figures, analysis scripts and released bootstrap
outputs for the article submitted to *Scientific Reports* (transferred from
*Communications Medicine*, COMMSMED-26-3259-T). The system the article
evaluates (LCHAI v2: pipeline, microservices, per-slide tooling, deployment)
lives in a separate repository, <https://github.com/slima2/lchai>.

## Layout

| Path | Contents |
|---|---|
| `main_SR.tex`, `preamble_SR.tex`, `sections/*_SR.tex` | Article (Introduction, Results, Discussion, Methods, Data and Code availability, back matter) |
| `supplementary_SR.tex`, `sections/09_supplementary_SR.tex` | Supplementary Information (Figures S1–S10, Tables S1–S19, Notes 1–5) |
| `tables/` | Every display table of the article and the SI as a standalone `.tex` |
| `figures/` | Figure files used by the two documents, plus the `gen_*.py` scripts that generate the schematic figures |
| `shared/references.bib` | Bibliography |
| `data/` | Released bootstrap outputs (see below) |
| `scripts/` | Acquisition, preparation, training and analysis scripts (see below) |
| `build_SR.ps1` | Builds `main_SR.pdf` and `supplementary_SR.pdf` (the two documents cross-reference each other through `xr-hyper`, so they are compiled alternately) |
| `wordcount_sr.py`, `check_citations.py`, `make_SR_bundle.py` | Journal-limit checks, citation audit, submission bundle |

## Released data (`data/`)

All cohorts are public (TCGA-LUAD, CPTAC-LUAD, TCGA-LGG/GBM with VASARI
scores, BraTS 2021, PTB-XL); the files here are the resampling outputs
behind the article's intervals, so every contrast can be re-read without
retraining.

- `luad_hierarchical_contrasts.csv`: the 48 between-condition contrasts of
  Table 2 (six genes x four contrasts x two outer protocols), with 95%
  hierarchical bootstrap intervals over five seeds and patients (or sites).
- `glioma_mgmt_bootstrap_{aurocs,contrasts}.csv`,
  `ecg_bootstrap_{aurocs,contrasts}.csv`: pooled out-of-fold AUROC per
  condition and the four contrasts per target for Tables 3 and 4
  (BraTS 2021 MGMT; PTB-XL sex and age >= 65), from 2,000 hierarchical draws
  over five seeds (42–46) and patients. The `.md` files are the same tables
  rendered for reading.
- `{glioma_mgmt,ecg}_results_seed*.json`: per-seed fold means and SDs behind
  Supplementary Tables S17 and S18.

## Scripts (`scripts/`)

Acquisition and preparation for the two cross-domain instantiations:
`download_glioma.sh`, `fetch_vasari.py`, `build_glioma_labels.py`,
`prepare_glioma_slices.py`, `propagate_vasari.py` (glioma MRI, domain B);
`download_ecg.sh`, `build_ptbxl_vocabulary.py` (PTB-XL, domain C);
`download_cptac_wsi.sh`, `fetch_cptac_labels.py`, `extract_cptac_features.py`,
`run_cptac_inference.py`, `run_cptac_domainA_v2.py`, `eval_cptac_transfer.py`
(CPTAC-LUAD transfer). Runners: `run_glioma_domainB.py`, `run_ecg_domainC.py`;
`patch_oof_crossdomain.py` and `patch_seed_crossdomain.py` add out-of-fold
prediction export and a `--seed` argument to them, and `run_e9b_dgx.sh` runs
the five-seed protocol end to end. Bootstraps: `bootstrap_crossdomain_contrasts.py`
(cross-domain arms), `make_contrasts_table.py` (renders Table 2 from
`data/luad_hierarchical_contrasts.csv`). Controls and audits:
`build_tumor_control.py`, `verify_controls.py`, `threshold_sensitivity.py`,
`literature_power_audit.py`, `audit_*.py`. Synthetic validation of the regime
rule (not reported in the article): `synthetic_pid_cohort.py`,
`run_synthetic_pid_benchmark.py`. Credentials are never stored in the
repository: copy `env.sh.example` to `env.sh` and export the variables it
lists.

## Building

Requires a TeX distribution with `latexmk` (tested with MiKTeX) and Python 3
for the checks.

```powershell
.\build_SR.ps1          # main_SR.pdf and supplementary_SR.pdf
.\build_SR.ps1 -Clean   # discard auxiliary files first
```

## Licence

Manuscript text and figures (c) the authors; all rights reserved until
publication. A licence for the scripts will be added at acceptance.
