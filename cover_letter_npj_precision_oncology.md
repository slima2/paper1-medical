# Cover letter — npj Precision Oncology

**To:** Editor-in-Chief, *npj Precision Oncology*
**Re:** Submission of original Article — *Translating Histology into Genotype: A Three-Regime, Six-Level Framework for Interpretable Mutation Prediction in Lung Adenocarcinoma.*

Dear Editor,

We are pleased to submit our manuscript for consideration as an Article in *npj Precision Oncology*. Our work addresses a problem the journal has prioritised in its recent issues — the **clinical interpretability gap** in deep-learning predictors of actionable driver mutations from haematoxylin-and-eosin (H&E) histopathology — and contributes a framework that complements, rather than competes with, the foundation-scale efforts of Wang *et al.* (CHIEF, *Nature* 2024; *n* = 60,530) and Campanella *et al.* (EAGLE, *Nature Medicine* 2025; *n* = 8,461).

## The clinical question

In lung adenocarcinoma (LUAD), treatment selection is biomarker-conditional: osimertinib for *EGFR*-mutant disease, sotorasib for *KRAS* G12C, and immune-checkpoint blockade calibrated by *STK11*/*KEAP1* co-mutation status. Molecular profiling is logistically costly, slow, and unevenly available in low- and middle-income settings — a context our SOLCA-CISOL (Sociedad de Lucha Contra el Cáncer del Ecuador) collaborators encounter daily. A pathologist-readable predictor from the digitised slide could triage cases and pre-prioritise sequencing. Black-box predictors cannot fill this role because the **pathologist must be able to argue with the model in the morphological vocabulary they already use**.

## What we report

On TCGA-LUAD (*n* = 687 slides, 668 patients, 5-fold patient-stratified cross-validation) we predict six driver mutations — *TP53*, *EGFR*, *KRAS*, *STK11*, *KEAP1*, *RBM10* — through a pattern-informed multiple-instance learning (MIL) pipeline that decomposes every slide into the six expert-defined ANORAK histological growth patterns (lepidic, acinar, papillary, micropapillary, solid, cribriform). The MIL aggregator is paired with a 2-additive fuzzy Choquet integral that captures supra-additive **pattern–pattern interactions** an additive attention head cannot represent.

We did not optimise for cohort-mean AUROC: our best per-gene values (TP53 0.717, EGFR 0.694, KRAS 0.609, STK11 0.695, KEAP1 0.610, RBM10 0.661) are by design comparable to Saldanha *et al.* (*npj Precision Oncology* 2023) and below Coudray *et al.* (Inception-v3 e2e, *Nature Medicine* 2018), which uses an order of magnitude more EGFR-positive slides than the public TCGA cohort supports.

The contribution is instead three-fold:

1. **A six-level interpretability stack** — MIL attention overlay, pattern overlay, SHAP decomposition, Choquet–Shapley values with pairwise interaction indices, per-slide channel ablation, and fuzzy linguistic labels — exposed to the pathologist as a live, editable, auditable interface.
2. **A dual anti-hallucination LLM explanation layer** grounded in a curated NCIt + MONDO knowledge graph that constrains *what* the agent may assert and fuzzy linguistic labels that constrain *how* it speaks (no spurious decimal precision; a probability of 0.59 is reported as *moderate*, not *high*).
3. **A gene-conditional three-regime taxonomy** that, for the first time to our knowledge, *partitions* the six target genes into mechanistic categories: encoder saturation (TP53, EGFR; the foundation encoder already captures the discriminative morphology), inter-pattern co-occurrence (KRAS; the 2-additive Choquet operator discovers a positive interaction *I*(lepidic × solid) = +0.032 consistent with the invasive mucinous adenocarcinoma proxy), and out-of-ontology (STK11, KEAP1, RBM10; the signal lives in stromal, biochemical, or splicing substrates that the 6-pattern ontology does not encode).

A per-gene Conclusive/Inconclusive flag (mean AUROC ≥ 0.70) gates the explanation pathway and decouples cohort-level signal presence from per-slide morphological confidence.

## Fit with *npj Precision Oncology*

We selected *npj Precision Oncology* over the broader *Nature* family because the journal's audience — translational oncologists, computational pathologists, biomarker developers — is precisely the readership we want to engage. Our work continues the line opened by Saldanha *et al.* (*npj Prec Oncol* 2023) on H&E-only mutation prediction and adds the interpretability and per-gene-verdict layers that have not yet been published.

The live system is publicly accessible at **https://lchai.gptfy.biz** (anonymous read-only credentials available on request to the editor or reviewers) and the source code is at **https://github.com/slima2/lchai**. We would welcome the opportunity for the reviewers to interrogate the system directly.

## Compliance

- The TCGA-LUAD cohort is publicly available under the NIH GDC controlled-access policy; no further IRB approval is required for the retrospective analysis reported here.
- The SOLCA-CISOL preliminary validation was approved by the Comité de Ética de Investigación en Seres Humanos (CEISH) of SOLCA Ecuador (approval 2026-014).
- No competing interests to declare.
- All authors consented to the submission and have approved the final manuscript.
- The manuscript is not under consideration elsewhere; no prior version is in press.
- LLM use during manuscript preparation is documented in the Methods section.

We suggest the following potential reviewers (we have no professional conflicts with any of them):

- **Jakob N. Kather** (TU Dresden) — pan-cancer mutation prediction from H&E
- **Faisal Mahmood** (BWH/Harvard) — interpretable MIL on histopathology
- **Olivier Gevaert** (Stanford) — multimodal genotype–phenotype integration in oncology
- **Anant Madabhushi** (Emory) — computational pathology and clinical decision support

Conversely, we ask the editor to consider excluding submissions from groups in direct competition on the same TCGA-LUAD cohort.

Thank you for considering our manuscript. We look forward to your decision.

Sincerely,

**Servio F. Lima Reina** (corresponding)
GPTfy Medical, Guayaquil, Ecuador
slima@gptfy.biz

On behalf of: Servio R. Lima Idrobo MD, Doménica N. López García MD, Luis Terán.
