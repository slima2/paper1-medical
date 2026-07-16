# Paper 1 (Medical) — Translating Histology into Genotype

Source for the manuscript:

> **Translating Histology into Genotype: A Three-Regime, Six-Level Framework
> for Interpretable Mutation Prediction in Lung Adenocarcinoma**

A pattern-informed multiple-instance-learning (MIL) pipeline (LCHAI v2) that
predicts six driver mutations (*TP53, EGFR, KRAS, STK11, KEAP1, RBM10*) from
H&E whole-slide images of the public TCGA-LUAD cohort, framed as an imperfect
**translation** from tissue morphology to genotype.

## Core contributions

1. **Three-regime taxonomy** of per-gene predictability — *encoder saturation*
   (TP53, EGFR), *inter-pattern co-occurrence* (KRAS), and *out-of-ontology*
   (STK11, KEAP1, RBM10).
2. **Six-level per-slide interpretability stack** reframed as *translation made
   legible* (a bilingual dictionary of source patterns ↔ target genes), with
   **untranslatability promoted to a first-class result**.
3. **Per-gene Conclusive/Inconclusive verdict** (a translatability flag).
4. **Dual anti-hallucination LLM explanation layer** grounded in an
   NCIt + MONDO knowledge graph (the target-domain lexicon).
5. **Fidelity spectrum** situating the LUAD map among analogous medical
   cross-domain inference tasks (radiogenomics, pathomics→genomics,
   retinal→cardiovascular, ECG→function, virtual staining).

## Repository layout

```
main.tex                 # entry point (Nature-style: Intro -> Results -> Discussion -> Methods)
preamble.tex             # packages, macros, \graphicspath
sections/                # 01_abstract .. 06_data_code_avail
tables/                  # LaTeX table inputs
figures/                 # figures (+ gen_domain_translation.py generator)
shared/references.bib    # bibliography (vendored)
cover_letter_*.md        # journal cover letters
SUBMISSION.md            # submission notes
```

## Build

```bash
pdflatex main
bibtex   main
pdflatex main
pdflatex main
```

(or `latexmk -pdf main.tex`). The bibliography is resolved from
`shared/references.bib`, so compile from the repository root.

## Regenerating the concept figures

```bash
cd figures
python gen_domain_translation.py   # -> domain_translation_concept.png, fidelity_spectrum.png
```

Requires `matplotlib` and `numpy`.
