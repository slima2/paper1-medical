# Display-item reorganisation plan (npj Digital Medicine)

**Goal.** npj Digital Medicine sets no page limit for Articles, but its
guidelines ask authors to *avoid unnecessary figures* and keep the main text
concise. After the cross-domain (ECG, glioma) additions the manuscript now
carries **25 display items (13 figures + 12 tables)** — far too many for the
venue. This plan trims the **main** set to **10 items (5 figures + 5 tables)**
that carry the core contributions, and moves the remaining **15 items
(8 figures + 7 tables)** to **Supplementary Information (SI)** without deleting
any content or any narrative text.

The main-text argument spine is preserved:
*framing → three-regime taxonomy → headline LUAD result → KRAS linchpin →
interpretability in the deployed system → literature benchmark → conservative
reporting under transfer (CPTAC) → cross-domain reproduction (summary)*.
All corroborative, methodological and per-domain-detail items move to SI; the
Discussion text that references them stays in the main body and simply points to
"Fig. S…/Table ST…".

Nothing is executed yet — this is for approval before we move blocks and update
cross-references.

---

## Current inventory (13 figures)

| # | Label | Section | Content | Decision |
|---|-------|---------|---------|----------|
| F1 | `fig:domain_translation` | Intro | Histology→mutation as imperfect domain translation (core framing) | **MAIN** |
| F2 | `fig:signal_tree` | Results | Three-regime taxonomy of predictive signal (core) | **MAIN** |
| F3 | `fig:kras_lepidic_solid_proxy` (a,b) | Results | KRAS regime (ii): lepidic×solid proxy + Choquet readout (linchpin) | **MAIN** |
| F4 | `fig:lchai_six_levels` | Results | Schematic six-level stack (same concept as F5) | → SI **S1** |
| F5 | `fig:lchai_ui_stack` | Results | Real deployed-UI six-level composite (KRAS, TCGA-55-7815) | **MAIN** |
| F6 | `fig:case_grid` (6 panels) | Results | Per-gene case studies (TP53/EGFR/KRAS/STK11/KEAP1/RBM10) | → SI **S2** |
| F7 | `fig:fidelity_spectrum` | Discussion | Fidelity spectrum of cross-domain mapping (conceptual scaffold) | **MAIN** |
| F8 | `fig:two_channels` | Methods | The two channels over which the PID is computed | → SI **S3** |
| F9 | `fig:kg_three_layers` | Methods | Three-layer knowledge graph | → SI **S4** |
| F10 | `fig:pipeline_ecg` | Discussion | ECG (PTB-XL) domain-C pipeline diagram | → SI **S5** |
| F11 | `fig:ecg_ptbxl` | Discussion | Real PTB-XL 12-lead ECG example | → SI **S6** |
| F12 | `fig:pipeline_glioma` | Discussion | Glioma/VASARI domain-B pipeline diagram | → SI **S7** |
| F13 | `fig:glioma_example` | Discussion | Real TCGA-LGG multiparametric MRI example | → SI **S8** |

## Current inventory (12 tables)

| # | Label | Section | Content | Decision |
|---|-------|---------|---------|----------|
| T1 | `tab:mutation_luad` | Results | Mutation-prediction AUROC (headline result) | **MAIN** |
| T2 | `tab:pid_regimes` | Results | Three regimes as PID atoms (core) | **MAIN** |
| T3 | `tab:literature_comparison` | Results | Per-gene AUROC vs published work (benchmark) | **MAIN** |
| T4 | `tab:cptac_transfer` | Discussion | CPTAC-LUAD off-the-shelf transfer (key diagnostic result) | **MAIN** |
| T5 | `tab:crossdomain_protocol` | Discussion | Framework instantiated across LUAD/ECG/glioma (thesis support) | **MAIN** |
| T6 | `tab:levels_of_analysis` | Methods | Four levels of analysis (cross-referenced scaffold) | → SI **ST1** |
| T7 | `tab:levels_crossdomain` | Discussion | Level structure across domains | → SI **ST2** |
| T8 | `tab:ecg_domainc` | Discussion | PTB-XL / ECG domain-C empirical run | → SI **ST3** |
| T9 | `tab:glioma_domainb` | Discussion | Glioma/VASARI domain-B v1 pilot run | → SI **ST4** |
| T10 | `tab:cptac_power` | Discussion | Positive-set sizes / power for CPTAC | → SI **ST5** |
| T11 | `tab:mutation_prevalence` | Methods | Mutation prevalence in the cohort | → SI **ST6** |
| T12 | `tab:viz_folds_methods` | Methods | Conclusive/Inconclusive verdict bands | → SI **ST7** |

---

## Resulting main set (10 items)

- **Figures (5):** F1 domain translation · F2 signal tree · F3 KRAS proxy ·
  F5 live-UI six-level stack · F7 fidelity spectrum.
- **Tables (5):** T1 LUAD AUROC · T2 PID regimes · T3 literature comparison ·
  T4 CPTAC transfer · T5 cross-domain protocol.

Every core contribution keeps a main-text display item:
- **C1 three-regime framework** → F2 (taxonomy) + T2 (PID atoms).
- **C2 LUAD instantiation + interpretability** → T1 (AUROC) + T3 (benchmark) +
  F3 (KRAS linchpin) + F5 (deployed six-level UI).
- **C3 self-reporting verdict / generality** → T4 (CPTAC diagnostic) +
  T5 (cross-domain reproduction) + F1/F7 (framing + fidelity scaffold).

**Moved to SI:** 8 figures (S1–S8) + 7 tables (ST1–ST7).

### Two judgement calls

1. **F4 (schematic six-level) → SI, F5 (real UI) stays main.** Same concept;
   the deployed interface is the stronger evidence for the actionable claim, and
   its panels double as the pipeline overview. The schematic stays in SI for
   readers who want the derivation diagram.
2. **T5 crossdomain_protocol stays main, but the per-domain detail tables
   (T8 ECG, T9 glioma) and all pipeline/real-data figures (F10–F13) go to SI.**
   The reader gets the "regimes reproduce across modalities and cohorts" claim
   from one summary table in the main text; the ECG/glioma numbers, pipelines
   and example images live in SI. This keeps the two-axes thesis visible without
   spending four figures + two tables on it in the body.

### Optional leaner set (8 items)

If a tighter body is preferred, additionally move **T3 literature_comparison →
SI** and **F7 fidelity_spectrum → SI**, leaving 4 figures + 4 tables. Not
recommended: the benchmark table is a common reviewer ask, and the fidelity
figure anchors the Discussion.

---

## SI ordering (proposed)

Group SI items so each stands on its own:

- **Figures.** S1 six-level schematic · S2 per-gene case grid ·
  S3 two-channel architecture · S4 three-layer knowledge graph ·
  S5 ECG pipeline · S6 real ECG example · S7 glioma pipeline ·
  S8 real glioma MRI example.
- **Tables.** ST1 levels of analysis · ST2 levels across domains ·
  ST3 ECG domain-C run · ST4 glioma domain-B run · ST5 CPTAC power ·
  ST6 mutation prevalence · ST7 verdict bands.

---

## Execution plan (once approved)

1. **Create SI scaffold.** Add `sections/07_supplementary.tex` after the body,
   opened with `\clearpage`, `\setcounter{figure}{0}`, `\setcounter{table}{0}`,
   `\renewcommand{\thefigure}{S\arabic{figure}}` and
   `\renewcommand{\thetable}{S\arabic{table}}` so items number S1, S2… / ST1…
   (using a `\section*{Supplementary Information}`). Add
   `\input{sections/07_supplementary}` in `main.tex` after `06_data_code_avail`.
2. **Move blocks.** Cut each demoted `\begin{figure}…\end{figure}` /
   `\begin{table}…\input{…}…\end{table}` block from its section into
   `07_supplementary.tex`, in the SI order above. The `\input{tables/…}` files
   and figure PNGs are unchanged — no file moves, no regenerated images.
3. **Fix cross-references.** Labels are unchanged, so every `\cref{fig:…}` /
   `\cref{tab:…}` to a moved item auto-resolves to "Fig. S…"/"Table ST…". Two
   manual checks: (a) the six-level enumeration in `03_results.tex` should point
   to `fig:lchai_ui_stack` (main), not `fig:lchai_six_levels` (SI); (b) the
   Discussion cross-domain paragraph should read naturally when ECG/glioma
   detail becomes "see Fig. S…/Table ST…".
4. **Panel letters.** npj wants multi-panel figures on one page with a)/b)/…
   labels. F3 and F5 (and SI S2 case grid) need panel letters at acceptance.
5. **Acceptance-stage only.** Moving figure legends after references and tables
   to end of document is required at acceptance, not initial submission.

## Notes / open checks
- **Abstract:** ~147 words, unstructured — fine for an Article; confirm the
  exact limit in the submission system.
- **Structure already compliant:** Intro → Results (subheadings) → Discussion
  (no separate Limitations/Conclusions) → Methods.
- **Wide tables:** watch T3, T4, T5 and SI ST3/ST4 for horizontal overflow;
  wrap in `\resizebox`/`tabularx` or set landscape if they exceed the text block.
- **Delete stale backup:** `sections/04_discussion_SUBHEADED_backup.tex.bak`
  duplicates figure/table labels; remove it before compiling the SI so labels
  stay unique.
