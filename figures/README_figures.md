# Paper 1 — Medical — Figure inventory

All figures are reused from the thesis. The preamble already adds
`../../SLIMA_Thesis_Research_ver_2026_rev_9/chapters/figures/` and
`/figures/` to `\graphicspath{}`, so you can include them by filename
only.

## Main figures (target 4–6 for Nature Cancer; flexible for npj Prec Onc)

| Suggested # | Filename | Source | Purpose |
|---|---|---|---|
| Fig 1 | `lchai_full_system.svg` (or `.png`) | `D:\Dropbox\PHD\LCHAI v2\oncology-xai\docs\` | LCHAI v2 system overview + 6-level interpretability stack + dual anti-hallucination |
| Fig 2 | `signal_tree.png` | thesis ch06:1245–1295 | Three-regime gene taxonomy decision tree (CENTRAL FIGURE) |
| Fig 3 | `kras_lepidic_solid_proxy.png` + `choquet_kras_55_7815.png` | thesis ch06:1199–1218, 3705–3740 | KRAS IMA-proxy + Choquet interaction `+0.032` |
| Fig 4 | `lchai_v2_KRAS_TCGA-55-7815_combined.png`, plus thumbnails+overlays for TP53/EGFR/STK11/KEAP1/RBM10 representative cases | thesis ch06:2042–3349 | Per-gene quad-grids (thumbnail + pattern + ABMIL attention + combined) |
| Fig 5 | `auroc_auprc_prevalence.png` + `tab:viz_folds` Conclusive/Inconclusive verdict | thesis ch06:751–855 + 1900–1947 | Per-gene Conclusive verdict + AUROC vs AUPRC vs prevalence |
| Fig 6 (optional) | LCHAI screenshot + active-learning correction | `D:\Dropbox\PHD\LCHAI v2\oncology-xai\apps\webapp\src\panels\ViewerPanel.tsx` (live screenshot) | LCHAI as clinical inspection tool |

## Extended Data (up to 10 for Nature Cancer)

| ED # | Filename | Source | Purpose |
|---|---|---|---|
| ED 1 | `shap_decomp_{TP53,EGFR,KRAS,STK11,KEAP1,RBM10}_*_bar.png` | thesis ch06:3350–3555 | Per-gene SHAP embedding/pattern decomposition |
| ED 2 | `boundary_vs_size_v2.png` | thesis ch06:417–438 | Pattern classifier: boundary-vs-size (briefly here) |
| ED 3 | `choquet_variance_map.png` | thesis ch06:1740–1784 | When Choquet helps/hurts (3 regimes) |
| ED 4 | `active_learning_workflow.png` + `active_learning_delta_training.png` | thesis ch04:2013 + LCHAI UI | Active-learning Δ-train loop |
| ED 5 | `kg_three_layers_method.png` + `edge_store_query_flow.png` + `cancer_genetic_mutation_ncit_mondo.png` | thesis ch04:2079–2232 | NCIt+MONDO 3-layer KG architecture |
| ED 6 | CONSORT-style flowchart | NEW (generate from `tab:datasets_overview`, `tab:mutation_prevalence`) | Cohort flow (TCGA query → 1,054 slides → 687 analysable) |
| ED 7 | `tab:variant_types` + `tab:mutation_prevalence` | thesis ch03:1162–1343 | MAF processing |
| ED 8 | `tab:external_cohorts` (CPTAC-3, APOLLO, EAGLE-1) | thesis ch03:1479–1525 | External cohort plan |
| ED 9 | SOLCA explainability questionnaire summary | thesis appendix `LCHAI_Explainability_Questionnaire_SOLCA.{md,pdf}` | Pilot pathologist study |
| ED 10 | `tab:data_scale` | thesis ch03:1546–1633 | Data-scarcity vs foundation cohorts |

## How to fetch figures

```powershell
# From paper1_medical/, copy whatever you actually use into figures/
# (keeps the paper self-contained for archival):
Copy-Item "D:\Dropbox\PHD\THESIS\SLIMA_Thesis_Research_ver_2026_rev_9\chapters\figures\signal_tree.png" "figures\"
# … etc.
```

Or simply reference them by filename from the existing thesis path — the
`\graphicspath{}` in `preamble.tex` already covers both locations.
