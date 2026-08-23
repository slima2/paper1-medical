# Cover letter — npj Digital Medicine

**To:** Editor-in-Chief, *npj Digital Medicine*
**Re:** Submission of original Article — *A three-regime translatability framework for cross-domain medical prediction with an instantiation in lung adenocarcinoma.*

Dear Editor,

We submit this manuscript for consideration as an Article. It addresses a reporting problem that cuts across digital medicine rather than a question confined to one disease: a large family of clinical AI systems predicts, from an accessible source modality, a target that modality does not directly measure — histology onto genotype, MRI onto molecular markers, the electrocardiogram onto physiological and laboratory state — and every one of them is currently reported as a single cohort-average AUROC. That number hides which targets are actually reachable from the source and why the others are not, and it is the reason a decade of negative results on targets such as MGMT-promoter methylation has accumulated as community-wide defeat rather than as a finding.

## The contribution

We propose that such systems be reported by **per-target translatability** instead. The framework has three parts.

First, a **three-regime taxonomy** — encoder saturation, inter-pattern co-occurrence, out-of-ontology — that is *derived* rather than descriptive. The regimes are the redundant, synergistic and unique atoms of a partial information decomposition of an expert-defined source ontology against a learned encoder, which is what makes them portable to any domain rather than a relabelling of one cohort's AUROC table.

Second, a **level-explicit Conclusive/Inconclusive verdict**. We are deliberate that this is a population-level statement and cannot be read as a per-case one. A gene can be Inconclusive across a cohort for three distinct reasons — information absent at every level, information real but confined to a morphologically defined subpopulation and therefore diluted in the average, or information present but unestimable at the available prevalence — and only the first is untranslatability. The manuscript states, for every quantity reported, whether it was estimated at the population, patient, slide or tile level and what a claim at that level does and does not license.

Third, a **six-level per-slide explanation stack (LCHAI)** wrapped in a knowledge-graph-grounded anti-hallucination language layer, in which the levels are the per-case instruments of the same decomposition. The fuzzy linguistic labels constrain the vocabulary the language agent may use, an NCIt + MONDO knowledge graph constrains the entity set it may assert, and a literature layer constrains its citations.

## What the instantiation shows

On the public TCGA-LUAD cohort (687 slides, 668 patients, 5-fold patient-stratified cross-validation) the six driver genes partition cleanly: *TP53* and *EGFR* saturate the encoder, *KRAS* is synergistic with a lepidic × solid Choquet interaction of +0.032 matching the invasive-mucinous phenotype, and *STK11*, *KEAP1* and *RBM10* fall outside the six-pattern ontology.

We want to be explicit that we did not optimise for cohort-mean AUROC and do not outperform the foundation-scale published competition; at this cohort size we would not expect to. The result we are asking reviewers to weigh is the framework, and *KRAS* is the case that demonstrates why it matters: the gene averages 0.609 and is Inconclusive as a cohort, yet slide TCGA-55-7815 carries P = 0.965 with that same interaction localised in the tissue. Under current reporting practice that slide is invisible. Under ours it is the point.

## Fit with *npj Digital Medicine*

We chose this journal over the oncology titles deliberately. The contribution is a reporting and evaluation standard for clinical AI, and its natural readership is the digital-medicine community that builds and deploys cross-modality predictors, not the LUAD biology community. The lung adenocarcinoma work is the instantiation, not the thesis.

Consistent with that claim, the manuscript instantiates the framework on two further fully public modalities and reports the results in the main text. In **glioma MRI → genotype**, a standardised radiology-feature vocabulary supplies the source ontology: on the TCGA-LGG cohort *IDH* is Conclusive and encoder-saturated (AUROC 0.86) and *1p/19q* is a complementarity case that neither channel captures alone but their fusion recovers (0.74), while *MGMT* — the best-documented untranslatable target in medical machine learning — scored at full power on the balanced BraTS cohort (n = 577, ~52% methylated), is Inconclusive in every condition (0.53–0.56): the same instrument that certifies IDH and recovers 1p/19q correctly declines to certify MGMT, issuing a decade of documented failure as a diagnostic verdict rather than a defeat. In the **electrocardiogram → physiological state**, the 71 graded, SNOMED-CT-mapped SCP-ECG statements of PTB-XL supply the source vocabulary: patient sex (AUROC 0.90) and age (0.84) carry the identical four-condition protocol intact, both encoder-saturated, confirming the machinery transfers across a modality family rather than only in principle. We also stress-test the framework *within* pathology through an off-the-shelf cohort transfer to CPTAC-LUAD: the frozen predictor collapses (*TP53* 0.72 → 0.53, and 0.48 after stain normalisation) and — this is the point — the verdict correctly turns Inconclusive, so the diagnostic reports its own limits under distribution shift instead of asserting a spurious call. The acquisition and preparation scripts for every instantiation are released with the manuscript.

The live system is publicly accessible at **https://lchai.gptfy.biz** (anonymous read-only credentials available on request to the editor or reviewers) and the source code is at **https://github.com/slima2/lchai**. We would welcome the opportunity for reviewers to interrogate the deployment directly rather than only the manuscript.

## Compliance

- The manuscript is formatted to the *npj Digital Medicine* Article content type: title of 14 words free of punctuation, a 150-word unsubheaded abstract, no subheadings in the Introduction, subheadings in Results and Methods, and a Discussion written as continuous prose with no limitations or conclusions sections.
- Data availability, Code availability, Acknowledgements (including funding), Author contributions and Competing interests statements are provided. Three of the four authors are affiliated with GPTfy Medical, which developed and hosts the LCHAI system reported here; this is declared in full in the Competing interests statement.
- The reference list currently stands at 76 entries against the journal's 60-reference guideline. We are glad to reduce it on request; the excess is concentrated in the cross-domain instantiations, where we cite the primary evidence for the MGMT and laboratory-indicator literature that our predictions must be judged against.
- TCGA-LUAD is publicly available under the NIH GDC policy; no further IRB approval is required for the retrospective analysis reported here. No new human-subject data were collected and no patient-identifiable information was accessed.
- All authors consented to the submission and approved the final manuscript. The manuscript is not under consideration elsewhere and no prior version is in press.
- LLM use during manuscript preparation is documented in the Methods section.

## Suggested reviewers

We have no professional conflicts with any of the following:

- **Jakob N. Kather** (TU Dresden) — mutation prediction from H&E across cancer types
- **Faisal Mahmood** (BWH / Harvard) — interpretable multiple-instance learning on histopathology
- **Nils Strodthoff** (Oldenburg) — PTB-XL and ECG representation learning
- **Olivier Gevaert** (Stanford) — multimodal genotype–phenotype integration and radiogenomics
- **Spyridon Bakas** (Indiana) — BraTS and the glioma radiogenomic benchmarks

Thank you for considering our manuscript.

Sincerely,

**Servio F. Lima Reina** (corresponding)
GPTfy Medical, Guayaquil, Ecuador
slima@gptfy.biz

On behalf of: Servio R. Lima Idrobo MD, Doménica N. López García MD, Luis Terán.
