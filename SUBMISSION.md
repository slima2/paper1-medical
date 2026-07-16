# Paper 1 — Medical — Submission checklist

**Primary target:** [npj Precision Oncology](https://www.nature.com/npjprecisiononcology/for-authors-and-referees/guide-to-authors)
**Stretch target:** [Nature Cancer](https://www.nature.com/natcancer/content) (Article)

---

## npj Precision Oncology — Article (primary)

| Item | Spec | Status |
|---|---|---|
| Title | ≤ 15 words, no punctuation/idioms | TODO |
| Abstract | ≤ 150 words, no subheadings | TODO |
| Main text | No hard limit; concise (free-format initial submission OK) | TODO |
| Display items | No hard cap; recommended ≤ 8, multi-panel on single page using (a), (b), (c) labels | TODO |
| Figure legends | ≤ 350 words per figure | TODO |
| References | No hard cap; numbered, no titles in ref list | TODO |
| Data availability statement | Required | TODO (TCGA-LUAD GDC + LCHAI GitHub) |
| Code availability statement | Required | TODO (https://github.com/slima2/lchai) |
| Reporting checklists | TRIPOD-AI for clinical-prediction modelling | TODO |
| Cover letter | 1 page; explain why npj Prec Onc fit; novelty vs Coudray/Saldanha | TODO |
| Author contributions (CRediT) | Required | TODO |
| Conflicts of interest | Required | TODO |
| Ethics statement | Required — TCGA is publicly available; SOLCA-CISOL approval present | TODO |
| Initial format | PDF or Word; LaTeX accepted at acceptance | OK (we keep LaTeX) |
| File submission | Manuscript + figures (embedded OK at submission) + supplementary as single PDF | — |

### Structure (npj allows free-form; recommended IMRD)

1. Title page + author block
2. Abstract (≤ 150 words)
3. Introduction
4. Results
5. Discussion
6. Methods
7. Data availability
8. Code availability
9. References
10. Acknowledgements
11. Author contributions
12. Competing interests
13. Figures + figure legends
14. Tables
15. Supplementary (single PDF; supp figs one per page)

### Reviewing standards
- TRIPOD-AI (transparent reporting of prediction models with AI).
- STARD-AI (for diagnostic-test components, e.g. Conclusive/Inconclusive flag).
- Independent test-cohort discussion (we use 5-fold patient-stratified CV + recommend prospective CPTAC-3 / APOLLO / EAGLE-1 validation).

---

## Nature Cancer — Article (stretch)

| Item | Spec | Status |
|---|---|---|
| Title | ≤ 15 words | TODO |
| Abstract | Up to 150 words, no headings | TODO |
| Main text | **3,500–6,000 words** (excl. abstract, methods, refs, legends) | TODO |
| Display items | **4–6 figures/tables/boxes** + up to 10 Extended Data items | TODO |
| References | Up to ~150 (~25 per 1,000 words) | TODO |
| Reporting summary | Mandatory | TODO |
| Editorial policy checklist | Mandatory | TODO |
| Methods section | Required, with subheadings | TODO |
| Data availability statement | Required, with persistent identifier | TODO |
| Code availability statement | Required | TODO |
| LLM-use declaration | Required in Methods (journal policy) | **OMITTED by author decision** |
| Cover letter | Required | TODO |
| Presubmission inquiry | Optional but recommended for Nature Cancer | RECOMMENDED |
| Initial format | PDF, Word, or compiled-PDF from LaTeX | OK |

### Mapping our content to Nat Cancer's 4–6 display items
| # | Display item | Source in thesis |
|---|---|---|
| Fig 1 | LCHAI v2 system overview + 6-level interpretability stack + dual anti-hallucination (NCIt/MONDO + fuzzy labels) | ch04:1521–2008; `lchai_full_system.svg` |
| Fig 2 | Three-regime gene taxonomy decision tree (`signal_tree.png`) + per-gene AUROC matrix | ch06:1245–1295 + `tab:mutation_luad` |
| Fig 3 | KRAS lepidic×solid IMA-proxy: Choquet interaction +0.032, per-slide overlay + Shapley readout | ch06:1199–1218, 3705–3740 |
| Fig 4 | Per-gene quad-grids: thumbnail + pattern overlay + ABMIL attention + combined (TP53 + EGFR + KRAS + STK11 + KEAP1 + RBM10) | ch06:2042–3349 |
| Fig 5 | Conclusive/Inconclusive verdict table + AUROC-vs-AUPRC-vs-prevalence (`auroc_auprc_prevalence.png`) + literature comparison heatmap | `tab:viz_folds`, `auroc_auprc_prevalence.png`, `tab:literature_comparison` |
| Fig 6 (optional / Extended Data) | LCHAI screenshot with pathologist HITL active-learning correction | ch05 LCHAI UI |

### Extended Data candidates (up to 10)
- ED1: SHAP embedding/pattern decomposition per gene (6 panels)
- ED2: Per-class F1 + boundary-vs-size analysis (cribriform vs micropapillary)
- ED3: Choquet variance map (`choquet_variance_map.png`)
- ED4: Active-learning Δ-train workflow
- ED5: NCIt+MONDO 3-layer KG architecture
- ED6: TCGA-LUAD cohort flowchart (consort-style)
- ED7: Variant-type table + MAF processing
- ED8: External-cohort plan (CPTAC-3, APOLLO, EAGLE-1)
- ED9: SOLCA explainability questionnaire summary
- ED10: Data-scarcity comparison (`tab:data_scale`)

---

## Reusable assets

| Asset | Source path |
|---|---|
| Bib | `../shared/references.bib` (53 KB — prune to ~80 for npj, ~150 for Nat Cancer) |
| Figures | `D:\Dropbox\PHD\THESIS\SLIMA_Thesis_Research_ver_2026_rev_9\chapters\figures\` and `figures\` (see `figures/README_figures.md`) |
| Tables | `tables/tab_mutation_luad.tex` (already extracted; copy `tab:gene_unified`, `tab:viz_folds`, `tab:literature_comparison` as needed) |
| LCHAI screenshots | `D:\Dropbox\PHD\LCHAI v2\oncology-xai\docs\` |

---

## Submission portal links
- npj Precision Oncology: https://mts-npjpo.nature.com/
- Nature Cancer: https://mts-naturecancer.nature.com/
- Manuscript transfer service (npj → Nat Cancer or vice-versa) supported between Nature Portfolio journals.
