# Handoff — npj Digital Medicine baseline → new session: retarget for *Nature Machine Intelligence*

**Purpose.** Snapshot of the competitive/bibliographic analysis and the venue
decision made on 2026-08-30, so that a **new session** can take the current
manuscript as a baseline and retarget it at *Nature Machine Intelligence* (NMI)
**without losing the npj Digital Medicine version**, which is submission-ready
as-is.

Read this first; it carries context the new session will not have.

---

## 0. TL;DR for the next session

- The manuscript in this repo is **finished and formatted for npj Digital
  Medicine**. Do not overwrite it. Work on a **copy**.
- **We do compete** with one group: Liang et al., NeurIPS 2023
  (`liang2023quantifying`). Everything else in the PID neighbourhood is either
  old theory, a low-impact journal, or an unrefereed preprint.
- **No one has published multimodal/medical PID in a Nature-family journal.**
  That is the white space this paper is aiming at.
- **Blocking issue in the current cover letter:** Faisal Mahmood is suggested as
  a reviewer under a "no professional conflicts" heading, but he is a **co-author
  of `liang2023quantifying`**, our nearest prior art. Fix before any submission.
- **Venue recommendation: NMI** (IF 22.28 vs npj DM 18.0), better conceptual fit,
  but it costs a serious restructuring: main text must drop from **12,293 → ~5,000–6,000
  words**, and 27 display items must be re-tiered.

---

## 1. Frozen baseline — do not lose this

The npj Digital Medicine version is complete: manuscript, compiled `main.pdf`,
and `cover_letter_npj_digital_medicine.md` / `.pdf` (updated 2026-08-30 with the
CPTAC refit results, the four Inconclusive states, and the 84-reference count).

- Canonical source: `d:\Dropbox\PHD\THESIS PAPERS\SLIMA Pattern-informed ...\paper1_medical\`
- Build copy / git root: `D:\repos\paper1-medical`
- Remotes: `github` → `slima2/paper1-medical`, `gitlab` → `serviolimareina/paper1-medical`
- Bibliography: `shared/references.bib` (84 entries)
- LaTeX toolchain: MiKTeX installed locally; **no Perl**, so `latexmk` does not
  run. Build manually: `pdflatex` → `bibtex` → `pdflatex` → `pdflatex`.

**Recommendation:** branch or fork rather than edit in place, e.g. keep `main`
as the npj DM submission and create `nmi` for the retarget. That way the npj DM
version stays a one-command checkout if NMI desk-rejects.

---

## 2. Competitive analysis of the PID bibliography

### 2.1 The PID-adjacent citations and where they live

| Citation | Authors | Year | Venue | Type |
|---|---|---|---|---|
| `williams2010pid` | Williams, Beer | 2010 | arXiv | preprint (founding PID paper) |
| `bertschinger2014unique` | Bertschinger, Rauh, Olbrich, Jost, Ay | 2014 | *Entropy* (MDPI) | journal, low IF |
| `chan2017pidc` | Chan, Stumpf, Babtie | 2017 | *Cell Systems* | journal, strong |
| `krohova2019hrv` | Krohova, Faes, … Javorka | 2019 | *Entropy* (MDPI) | journal, low IF |
| `liang2023quantifying` | Liang, Cheng, Fan, Ling, Nie, **R. J. Chen**, Deng, Allen, Auerbach, **F. Mahmood**, Salakhutdinov, Morency | 2023 | **NeurIPS** | conference |
| `zhang2025evaluatingpid` | Zhang, Ding, Luong, Hsu (UCLA) | 2025 | *J. Biomedical Informatics* | journal, mid-tier |
| `padegal2025modality` | Padegal, Kashyap, Rayasam, Shekhar, Narayan | 2025 | arXiv | preprint |
| `choi2025icym2i` | Choi, Jeanselme, Elias, Joshi | 2025 | arXiv | preprint |
| `saisan2026ceilings` | Saisan, Patel | 2026 | **bioRxiv** | preprint |
| `chopra2026pidmri` | Chopra, Kurt | 2026 | arXiv | preprint |
| `madaan2026primo` | Madaan, Chopra, Cho | 2026 | arXiv | preprint |

### 2.2 The structural finding

PID theory lives in arXiv and *Entropy*. Its biomedical applications top out at
*Cell Systems* (2017, gene regulatory networks — not multimodality) and the
*Journal of Biomedical Informatics* (2025). The flagship multimodal-PID paper is
a **conference** paper, not a journal article. **Everything from 2025–2026 is
still an unrefereed preprint.**

Consequence: there is no incumbent occupying a high-impact journal in this space.
We are not crashing anyone's venue, and a Nature-family placement would be the
first for medical multimodal PID.

### 2.3 What the competition actually is

**Liang et al. 2023 (NeurIPS)** — the real competitor. They *estimate PID atoms
numerically* in order to select a fusion method. We *explicitly do not estimate
atoms*; we read the regime off a **model-side signature** (the ordering of
B1/B2/B3/PI/ABMIL/FC). Different axis, same vocabulary.

The reviewer risk is a one-line dismissal: *"this is Liang's taxonomy applied to
pathology."* The rebuttal must be pre-loaded in the manuscript, not improvised.

**Zhang et al. 2025 (JBI)** — an **ally**, not a competitor. They document the
limitations of numerical PID estimation for fusion-method selection in biomedical
data, which is precisely why we avoid it. Cite them as justification for the
model-side design choice, prominently.

### 2.4 Our defensible differentiators

1. **Adjudication with statistical power** — `Inconclusive` as a first-class
   verdict, with the bootstrap CI machinery and the n₊ floor. No competitor has
   this; they all report point estimates.
2. **The MGMT negative control** — an instrument that certifies IDH, recovers
   1p/19q, and *correctly declines* to certify the best-documented untranslatable
   target in medical ML. This is a falsification test nobody else runs.
3. **Three modalities** (histology, MRI, ECG) plus a real cohort transfer
   (TCGA→CPTAC) with the FOV/magnification control.

### 2.5 Watch list — priority risk

`saisan2026ceilings` (Saisan & Patel, bioRxiv, "conserved information ceilings")
overlaps conceptually with our **encoder-saturation** construct and is **not yet
peer-reviewed**. If it lands in a good venue first, our saturation framing looks
derivative. Re-check its publication status before submitting, and make sure the
manuscript states clearly what saturation means *operationally in our signature*
rather than as a general principle.

Second tier to monitor: `choi2025icym2i` and `madaan2026primo`, both arXiv, both
about characterising modality informativeness.

---

## 3. Blocking issue — the suggested-reviewer conflict

`cover_letter_npj_digital_medicine.md` currently reads:

> ## Suggested reviewers
> We have no professional conflicts with any of the following:
> - **Jakob N. Kather** (TU Dresden) …
> - **Faisal Mahmood** (BWH / Harvard) — interpretable multiple-instance learning on histopathology
> - …

**Faisal Mahmood is co-author #10 of `liang2023quantifying`**, and Richard J. Chen
(co-author #6) is from his lab. We are therefore nominating the co-author of our
nearest prior art, while declaring no conflict.

The declaration is technically true (there is no personal conflict) and the
justification given is legitimate (he is cited for MIL, not PID). But he is the
single person with the most authority and the most incentive to write *"we did
this in NeurIPS 2023."* An editor who checks the bibliography will see it.

**NMI explicitly supports the clean fix.** Its submission guidelines state that
the cover letter may *"provide the name and institution of reviewers you would
like to recommend and/or people you would like to be excluded from peer review
(explaining why)."* Use the exclusion mechanism for Mahmood and Paul Pu Liang,
with the honest reason: direct prior art on the same framework.

NMI also offers **double-anonymised peer review** as an option, which is a second
lever worth considering here.

---

## 4. Venue comparison (JCR 2026, citation data 2025)

| Journal | IF | Quartile | Fit | Verdict |
|---|---|---|---|---|
| Nature Biomedical Engineering | 26.3 | Q1 | Wants a **new biomedical technology/device**. We offer a reporting framework. | Likely desk reject |
| Lancet Digital Health | 25.5 | Q1 | Wants **clinical outcome evidence / prospective validation**. Ours is retrospective and methodological. | Likely desk reject |
| **Nature Machine Intelligence** | **22.28** | Q1 | **Strong** — see §5 | **Recommended** |
| npj Digital Medicine | 18.0 | Q1 | Strong; tolerates current length; letter already written | Solid fallback |
| Medical Image Analysis | ~11 | Q1 | ECG domain sits outside "image analysis" | Downgrade |

---

## 5. Why NMI

- The paper's core claim is an **epistemology-of-ML claim** — what an AUROC can
  and cannot certify — instantiated in medicine. That is NMI's centre of gravity.
- NMI already published `rudin2019stop` and `lundberg2020treeshap`, **both in our
  bibliography**. We would be intervening in a conversation the journal hosts.
- NMI's stated evaluation criteria map onto this paper almost literally: they ask
  for **generalisation across multiple datasets/domains/institutions**, for
  **robustness with distribution shift and failure analysis**, and they name
  *"average accuracy without worst-case behaviour"* as the common mismatch — which
  is exactly the failure mode this paper argues current practice has.
- Liang et al. published at NeurIPS, i.e. to the ML community. NMI is the journal
  that community reads, which maximises our claim on the medical instantiation.
- +4.3 impact-factor points over npj DM.

---

## 6. What retargeting concretely requires

### 6.1 Word budget (measured 2026-08-30)

| Section | Current | NMI target | Action |
|---|---|---|---|
| `01_abstract` | 171 | ~150 | minor trim (already at 150 for npj DM) |
| `02_introduction` | 2,441 | ~1,200–1,500 | cut ~40% |
| `03_results` | 3,301 | ~2,500–3,000 | light trim, protect this |
| `04_discussion` | **6,551** | ~2,000–2,500 | **cut ~65% — the main job** |
| `05_methods` | 3,852 | unchanged | Methods are online-only, not in the word budget |
| `06_data_code_avail` | 553 | unchanged | — |
| `07_supplementary` | 1,336 | grows | absorbs the discussion cuts |
| **Main text (intro+results+disc)** | **12,293** | **~5,000–6,000** | |
| **Total** | **18,205** | — | |

The Discussion at 6,551 words is **twice the length of the Results** (3,301).
That imbalance is the thing to fix, and it is a genuine improvement regardless of
venue — it was already flagged in the npj DM pass.

> **Verify before you start:** NMI's "Preparing your material" page does not state
> a hard word cap for initial submission. The ~5,000–6,000 figure is the working
> Nature-family Article norm, not a quoted limit. Confirm on the Content Types
> page or with the editor; a presubmission enquiry is cheap and would settle both
> the length and the scope question in one go.

### 6.2 Display items — 27 currently

14 tables (`tables/tab_*.tex`) + 13 `\includegraphics` calls = **27 items**.
Nature-family structure gives three tiers:

- **Main display items** — target ~6–8. Candidates: the concept figure, the
  regime table, the KRAS Choquet slide, the cross-domain protocol table.
- **Extended Data** — **maximum 10 items**, online-only, "essential background
  … not included as main display items due to space constraints." This is the
  natural home for the domain-B/domain-C tables and the pipeline figures.
- **Supplementary Information** — everything else (~11 items), sent to reviewers
  alongside the manuscript.

Note that Extended Data is a *different mechanism* from the supplementary
reorganisation already done for npj DM (`SUPPLEMENTARY_REORG_PROPOSAL.md`); the
tiering has to be redone, not merely reused.

### 6.3 References

84 entries. Nature-family Articles usually run leaner. Check whether NMI imposes
a cap; if so, the trim should come from the LUAD-biology block, which is dense
and partly redundant for an ML readership.

### 6.4 Cover letter

Rewrite from scratch. `cover_letter_npj_digital_medicine.md` argues clinical-AI
reporting standards; NMI wants to hear why the *diverse readership of a machine
intelligence journal* should care. Lead with the measurement/epistemology claim,
not the oncology. Add the excluded-reviewer paragraph (§3). NMI also requires
disclosing related manuscripts under consideration elsewhere and any prior
discussion with an NMI editor. LLM use must be documented in Methods — already
done, verify it survives the rewrite.

### 6.5 Reporting guidelines

NMI's clinical-research policy requires that *"studies developing, validating, or
updating prediction models for diagnosis or prognosis must follow the TRIPOD
statement, or the TRIPOD-AI extension if … machine learning methods have been
used."* This paper builds and validates prediction models, so **TRIPOD+AI applies**.
Budget time for the checklist; it was not required in this form for the npj DM
submission.

---

## 7. Risks

| Risk | Mitigation |
|---|---|
| "This is Liang et al. 2023 applied." | Pre-load the differentiation: we do not estimate atoms; the contribution is adjudication under power + the MGMT negative control. Make it a titled paragraph, not a buried sentence. |
| Reviewer is Mahmood or Liang. | Use NMI's exclusion mechanism with a stated reason; consider double-anonymised review. |
| `saisan2026ceilings` publishes first. | Re-check status pre-submission; frame saturation operationally via our signature. |
| "You don't beat SOTA." | Already handled in the npj DM text — the framing is a measurement instrument, not a predictor. Keep that paragraph through the cuts; it matters more at NMI, not less. |
| Losing 65% of the Discussion damages the argument. | Move, don't delete. Supplementary Information is sent to reviewers, so the full argument stays reviewable. |
| Desk rejection. | npj DM version stays on `main`, ready to submit unchanged. Consider a presubmission enquiry to NMI first. |

---

## 8. Checklist for the new session

1. [ ] Copy/branch the repo; confirm the npj DM version is preserved intact.
2. [ ] Consider a presubmission enquiry to NMI (settles scope + length cheaply).
3. [ ] Verify NMI word limit and reference cap on the Content Types page.
4. [ ] Re-check `saisan2026ceilings` publication status.
5. [ ] Cut Discussion 6,551 → ~2,200, moving material to Supplementary Information.
6. [ ] Cut Introduction 2,441 → ~1,400.
7. [ ] Re-tier 27 display items into main (~7) / Extended Data (≤10) / Supplementary.
8. [ ] Strengthen the differentiation-from-Liang paragraph.
9. [ ] Write the NMI cover letter; include the excluded-reviewer paragraph.
10. [ ] Complete the TRIPOD+AI checklist.
11. [ ] Rebuild (`pdflatex` → `bibtex` → `pdflatex` ×2 — no `latexmk`, no Perl).

---

## 9. Provenance

Analysis performed 2026-08-30. Impact factors from JCR 2026 (2025 citation data).
NMI submission requirements verified against
`nature.com/natmachintell/submission-guidelines/preparing-your-submission` and
the journal's clinical-research editorial policy. Word counts and display-item
counts measured directly from `sections/`, `tables/` and `shared/references.bib`
in this repo on that date.
