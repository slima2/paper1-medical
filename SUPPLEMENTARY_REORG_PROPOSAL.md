# Display-item reorganisation proposal (npj Digital Medicine)

**Goal.** npj Digital Medicine has no page limit for Articles, but its
guidelines explicitly ask authors to *avoid unnecessary figures* and keep the
main text concise. The manuscript currently carries **20 display items
(9 figures + 11 tables)** — high for the venue. This proposal trims the **main**
set to **10 items (5 figures + 5 tables)** and moves the rest to
**Supplementary Information (SI)**, without deleting any content.

There is no hard rule; this is an editorial recommendation. Nothing here is
executed yet — it is for your approval before we move blocks and update
cross-references.

---

## Current inventory

### Figures (9 environments)

| # | Label | Where | Content | Proposed |
|---|-------|-------|---------|----------|
| F1 | `fig:domain_translation` | Intro | Histology→mutation as imperfect domain translation (core framing) | **Main** |
| F2 | `fig:signal_tree` | Results | Three-regime taxonomy of predictive signal (core) | **Main** |
| F3 | `fig:kras_lepidic_solid_proxy` (a,b) | Results | KRAS regime (ii): lepidic×solid proxy + Choquet readout (linchpin case) | **Main** |
| F4 | `fig:lchai_six_levels` | Results | Schematic system overview + six-level stack | → **SI (S1)** |
| F5 | `fig:lchai_ui_stack` | Results | Real deployed-UI six-level composite (KRAS, TCGA-55-7815) | **Main** |
| F6 | `fig:case_grid` (6 panels) | Results | Per-gene case studies (TP53/EGFR/KRAS/STK11/KEAP1/RBM10) | → **SI (S2)** |
| F7 | `fig:fidelity_spectrum` | Discussion | Fidelity spectrum of cross-domain mapping | **Main** |
| F8 | `fig:two_channels` | Methods | The two channels over which the PID is computed | → **SI (S3)** |
| F9 | `fig:kg_three_layers` | Methods | Three-layer knowledge graph | → **SI (S4)** |

### Tables (11)

| # | Label | Where | Content | Proposed |
|---|-------|-------|---------|----------|
| T1 | `tab:mutation_luad` | Results | Mutation-prediction AUROC (headline result) | **Main** |
| T2 | `tab:pid_regimes` | Results | Three regimes as PID atoms (core) | **Main** |
| T3 | `tab:literature_comparison` | Results | Per-gene AUROC vs published work (benchmark) | **Main** |
| T4 | `tab:cptac_transfer` | Discussion | CPTAC-LUAD off-the-shelf transfer (key negative result) | **Main** |
| T5 | `tab:levels_of_analysis` | Methods | Four levels of analysis (widely cross-referenced) | **Main** |
| T6 | `tab:crossdomain_protocol` | Discussion | Framework instantiated in three domains | → **SI (ST1)** |
| T7 | `tab:levels_crossdomain` | Discussion | Level structure across domains | → **SI (ST2)** |
| T8 | `tab:ecg_domainc` | Discussion | PTB-XL / ECG domain-C empirical run | → **SI (ST3)** |
| T9 | `tab:cptac_power` | Discussion | Positive-set sizes / power for CPTAC | → **SI (ST4)** |
| T10 | `tab:mutation_prevalence` | Methods | Mutation prevalence in the cohort | → **SI (ST5)** |
| T11 | `tab:viz_folds_methods` | Methods | Conclusive/Inconclusive verdict bands | → **SI (ST6)** |

---

## Resulting main set (10 items)

- **Figures (5):** F1 domain translation · F2 signal tree · F3 KRAS proxy ·
  F5 live-UI six-level stack · F7 fidelity spectrum.
- **Tables (5):** T1 AUROC · T2 PID regimes · T3 literature comparison ·
  T4 CPTAC transfer · T5 levels of analysis.

**Moved to SI:** 4 figures (S1–S4) + 6 tables (ST1–ST6).

This keeps the argument spine intact: *framing → taxonomy → headline result →
KRAS linchpin → interpretability in the real system → benchmark → honesty
(CPTAC) → conceptual scaffold*. Everything methodological or corroborative
moves to SI.

### Rationale for the two judgement calls

1. **F4 (schematic) → SI, F5 (real UI) stays main.** They cover the same
   six-level concept. The real deployed interface is the stronger evidence for
   the paper's actionable claim ("the explanation layer is what lets a
   pathologist use the prediction"), and its panels double as the pipeline
   overview. The schematic is retained in SI for readers who want the
   derivation diagram.
2. **F6 (6-panel case grid) → SI.** The KRAS panel already appears in the main
   narrative via F3/F5; the full per-gene grid is corroborative detail.

### Optional further trim (to reach 8 main items)

If a leaner set is preferred, also move **T3 literature_comparison → SI** and
**F7 fidelity_spectrum → SI**, leaving 4 figures + 4 tables.

---

## Execution plan (once approved)

1. **Create SI scaffold.** Add `sections/07_supplementary.tex` after the main
   body, begun with `\clearpage`, `\setcounter{figure}{0}`,
   `\setcounter{table}{0}` and `\renewcommand{\thefigure}{S\arabic{figure}}` /
   `\renewcommand{\thetable}{S\arabic{table}}` so items number S1, S2, …
   Add `\input{sections/07_supplementary}` in `main.tex` after
   `06_data_code_avail`.
2. **Move blocks.** Cut each demoted `\begin{figure}…\end{figure}` /
   `\begin{table}…\input{…}…\end{table}` block from its section into
   `07_supplementary.tex`, preserving order F→SI then T→SI. The `\input{tables/…}`
   files and figure PNGs are unchanged (no file moves needed).
3. **Fix cross-references.** All `\cref{fig:…}` / `\cref{tab:…}` to moved items
   keep the same labels, so they auto-resolve to “Fig. S1”, “Table ST3”, etc.
   Verify the six-level enumeration text now points to `fig:lchai_ui_stack`
   (main) rather than `fig:lchai_six_levels` (SI); one sentence to adjust in
   `03_results.tex`.
4. **Relabel multi-panel figures a), b), c).** npj requires multi-panel figures
   on a single page with panel letters. F3, F5 (composite) and F6 need a)/b)/…
   panel labels at acceptance. F5's badges (L1–L6) can be complemented with
   a)–g) if the editor insists.
5. **Move figure legends after references** and **tables to end of document**
   — required only at acceptance-stage formatting, not for initial submission.

## Notes / open checks
- **Abstract length:** currently ~147 words. npj Article abstracts are
  unstructured and typically fine at this length; confirm the exact limit in the
  submission system (the 70-word figure applies to Brief Communications /
  Matters Arising, not Articles).
- **Structure already compliant:** Intro (no subheadings) → Results
  (subheadings) → Discussion (no separate Limitations/Conclusions) → Methods.
- **Wide tables:** watch T3/T8/T4 for horizontal overflow at column width; wrap
  in `\resizebox`/`tabularx` or set landscape if they exceed the text block.
