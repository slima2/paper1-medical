# Cover letter — *Nature Cancer* (Article)

**To:** Editor-in-Chief, *Nature Cancer*
**Re:** Submission of original Article — *A three-regime translatability framework for cross-domain medical prediction with an instantiation in lung adenocarcinoma.*

Dear Editor,

We submit a manuscript that we believe meets the *Nature Cancer* threshold for an Article along two distinct axes — clinical relevance and methodological novelty — and asks the field to reframe how it reports computational-pathology mutation predictors.

## The two questions the paper answers

**(1) When the predictor is wrong on a per-gene basis, what should the model tell the pathologist?**
Recent foundation-scale systems — Wang *et al.* CHIEF (*Nature* 2024, *n* = 60,530), Campanella *et al.* EAGLE (*Nature Medicine* 2025, *n* = 8,461), Zhao *et al.* DeepGEM MIL (*Lancet Oncology* 2025, *n* = 3,637) — have decisively pushed cohort-mean AUROC upward for the common LUAD drivers, but the working pathologist still cannot argue with the resulting probability in the morphological vocabulary they were trained in. We provide a **six-level interpretability stack** (MIL attention, pattern overlay, SHAP, Choquet–Shapley, channel ablation, fuzzy linguistic labels) and a **dual anti-hallucination LLM explanation layer** (NCIt + MONDO knowledge graph constrains *what* the agent may claim; fuzzy linguistic labels constrain *how* it speaks) that exposes every quantitative result to the editable, auditable pathologist workflow.

**(2) Why is the cohort-mean metric misleading on a per-gene basis?**
Most reports collapse six genes into a single average AUROC. We show that *the genes themselves sit on qualitatively different points of the H&E-to-mutation pipeline*, and propose a **gene-conditional three-regime taxonomy** that, to our knowledge, has not previously been articulated:

- **Encoder saturation** (TP53, EGFR; the CTransPath foundation encoder already captures the relevant morphology and the pattern channel adds nothing).
- **Inter-pattern co-occurrence** (KRAS; only a non-additive aggregator — our 2-additive fuzzy Choquet integral — captures the lepidic × solid interaction = +0.032 that recovers the invasive mucinous adenocarcinoma proxy described histologically by Shim *et al.* 2015, Rekhtman *et al.* 2013, and Cha *et al.* 2017).
- **Out-of-ontology** (STK11, KEAP1, RBM10; the signal lives in stromal, biochemical, or splicing substrates that the 6-pattern WHO/ANORAK ontology does not encode, decomposing into label-space mismatch, no-morphological-projection, and collapse-to-prior).

The taxonomy is *immediately actionable* for the field: it identifies which genes would benefit from another architectural variant (KRAS — by adopting a non-additive aggregator) and which would benefit from multimodal fusion with RNA-seq or radiology (STK11, KEAP1, RBM10) rather than another loss function.

**The framework is not LUAD-specific.** Because the taxonomy is a decomposition of any expert vocabulary against any learned encoder, it is portable, and the manuscript demonstrates this on two further public cohorts. In glioma MRI → genotype it recovers the same regimes (*IDH* encoder-saturated at AUROC 0.86; *1p/19q* recovered only by channel fusion at 0.74), and in the electrocardiogram it transfers the identical protocol to a different modality family (sex 0.90, age 0.84). Critically, an off-the-shelf transfer of the LUAD predictor to the independent CPTAC-LUAD cohort collapses (*TP53* 0.72 → 0.53, and 0.48 after stain normalisation) and the verdict correctly turns Inconclusive — a built-in guard against the over-confident deployment that clinical oncology cannot tolerate.

## Why *Nature Cancer*

Three reasons.

1. **Audience.** The conclusion — that we should report computational-pathology predictors with per-gene Conclusive/Inconclusive flags, with explanations grounded in the WHO histological vocabulary, and with a multi-regime decomposition that places each gene on the H&E-to-mutation spectrum — is for the *integrated* cancer audience (oncologists + pathologists + computational scientists) that *Nature Cancer* serves. The result is methodological yet directly clinical.

2. **Trajectory.** Kather *et al.* (*Nature Cancer* 2020) is a foundational paper for our line of work; we extend it on a point Kather's pan-cancer paper explicitly left open — interpretability of the per-gene call.

3. **Demonstrability.** The live system runs at **https://lchai.gptfy.biz**, with anonymous read-only credentials available to reviewers. Every claim in the paper is reproducible from the public TCGA-LUAD cohort and the open-source code (**https://github.com/slima2/lchai**).

## What we explicitly do not claim

We deliberately *do not* claim to outperform CHIEF or EAGLE on cohort-mean AUROC — we cannot at this cohort size (*n* = 687 TCGA-LUAD slides, 668 patients, two orders of magnitude smaller). What we claim is that the public-data + interpretability paradigm complements the foundation-scale brute-force paradigm and that pathologist-facing deployment requires the interpretability layer regardless of cohort size.

## Compliance summary

- **Length:** main text within the 3,500–6,000 word window.
- **Display items:** the manuscript currently carries 8 main figures and 9 main tables plus supplementary figures (S1–S5) and tables (ST1–ST3); we are glad to rebalance main and Extended Data items to *Nature Cancer*'s display-item guidance.
- **Reporting summary, editorial-policy checklist, data-availability and code-availability statements:** included.
- **Ethics:** TCGA-LUAD is a publicly available NIH GDC dataset; SOLCA pilot approved by CEISH 2026-014.
- **Competing interests:** three of the four authors are affiliated with GPTfy Medical, which developed and hosts the LCHAI system reported here; this is declared in full in the manuscript's Competing interests statement.
- **Not under consideration elsewhere.**

We would welcome a presubmission inquiry at the editor's discretion. We can provide the live demo URL, a 5-minute demonstration video, and the compiled PDF on request.

Sincerely,

**Servio F. Lima Reina** (corresponding)
GPTfy Medical, Guayaquil, Ecuador
slima@gptfy.biz

On behalf of: Servio R. Lima Idrobo MD, Doménica N. López García MD, Luis Terán.
