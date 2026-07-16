# Handoff — Paper 2 (FuzzyArcLoss) → Next session: growth patterns for mutation prediction

**Purpose.** Self-contained snapshot of everything established in **Paper 2**
(the FuzzyArcLoss loss-family paper) plus the material that was *deliberately cut*
from it, so a **new session** can continue the work with a different focus:
**using LUAD histopathological growth patterns to predict genetic mutations**
(the MIL / Choquet / mutation-mapping direction, i.e. the thesis / "Paper 3"
track).

Read this first; it carries the context that the new session will not have.

---

## 0. TL;DR for the next session

- Paper 2 is a *loss-function* paper. It proves that a **sample-wise fuzzy
  angular margin (FuzzyArcLoss V2)** wins the **ambiguous class boundary** in
  fine-grained pathology, and ties the strongest fixed-margin baseline
  (ArcFace) on the aggregate. It is scoped **only** to tile-level classification.
- The **MIL + 2-additive Choquet aggregator + mutation-prediction** block was
  **removed** from Paper 2 (it was methodologically disconnected and
  statistically weak for a loss paper). **That removed block is exactly your
  next project.** See §7–§9.
- The biology you need (which growth pattern predicts which mutation) is already
  curated as citations in `shared/references.bib`; the map is in §8.

---

## 1. Paper 2 identity

- **Title:** *Fuzzy Angular-Margin Losses: A Sample-Wise Family for
  Boundary-Ambiguous Classification in Computational Pathology.*
- **Target venue:** IEEE **TMI** (Transactions on Medical Imaging), `IEEEtran`
  class. (Camera-ready TODO: drop `compsoc` option; convert `\IEEEcompsoc*`
  thanks/bios to plain IEEEtran style.)
- **Repos (mirrored):**
  - GitHub: `https://github.com/slima2/fuzzyarcloss-paper2` — **currently PRIVATE, must be made PUBLIC before submission** (the paper now cites it).
  - GitLab: `https://gitlab.com/serviolimareina/fuzzyarcloss-paper2`
  - Live system (login-gated): `https://lchai.gptfy.biz` (LCHAI v2.0)
  - Umbrella repo (public): `https://github.com/slima2/lchai`
- **Paper source (local, canonical):**
  `d:\Dropbox\PHD\THESIS PAPERS\SLIMA Fuzzy Angular-Margin Losses ...\paper2_engineering\`
  (main.tex + `sections/` + `tables/` + `figures/`; bib at `..\shared\references.bib`).
- **Standalone build copy:** `D:\repos\fuzzyarcloss-paper2\paper\` (vendored
  `shared/references.bib`, figures committed).

---

## 2. Core contribution — the FuzzyArcLoss family

**Mechanism.** Angular-margin softmax (ArcFace/CosFace/SphereFace) applies a
*fixed, sample-blind* margin `m` to every sample. FuzzyArcLoss modulates the
margin **per sample** by a **fuzzy membership** `μ_i ∈ [0.3, 1.0]` of the cosine
similarity between the embedding and its correct-class prototype:

- Confident, unambiguous samples → membership near 1.0 → full (wide) margin.
- Boundary-ambiguous samples (where experts disagree) → lower membership →
  narrower margin, so the model is **not** forced to memorise contested labels.
- **Principled schedule (NOT "margin-free"):** the *shape* of the per-sample
  margin is fixed by **one hard constraint + three design principles**
  (`a_H+b_H=1`; high-confidence floor; positive global floor; doubled gradient
  sensitivity `b_L=2b_H`) — not hand-tuned. The few remaining scalars (scale `s`,
  base margin `m`, breakpoint `τ`) **are** tuned by Optuna, but **under the same
  budget granted to every matched baseline** (tuning parity), and V2 still wins.
  ⚠️ The paper **no longer claims V2 "removes the margin hyperparameter entirely"**
  — that was reworded everywhere (abstract/intro C2/results/discussion/conclusion),
  because V2-Optuna does tune `m` (m*≈0.518). Use the parity framing, not
  "margin-free".

**Variants benchmarked (loss family):**
- FuzzyArcLoss **V2** (Optuna)  ← the headline method
- FuzzyArcLoss **V2-ClassParams** (per-class τ/margin/scale)
- FuzzyArcLoss **V3 SubCenters** (Optuna)
- FuzzyArcLoss **V3.1 SubCenter** (targeted)
- FuzzyArcLoss **V4 Boundary**
- (V1 legacy)

**Baselines:** ArcFace, ArcFace-matched, CosFace-matched, SphereFace,
SphereFace-matched, CurricularFace, Softmax-CE (+ auxiliary Triplet+Angular,
Center+Softmax in the 18-loss screen).

---

## 3. Benchmark, data, encoder

- **Primary benchmark:** LUAD growth patterns (ANORAK-derived), **n = 637 tiles**
  of **224×224 px** at level 0 (RGB + tissue mask = 4-channel input),
  **6 classes**: `acinar, cribriform, lepidic, micropapillary, papillary, solid`.
  (Tile size is 224×224 to match CTransPath; a stray "256×256" in the method was
  corrected. External cervix uses 128×128 cell crops.)
- **Ambiguous boundary of interest:** **cribriform** (and cribriform↔acinar,
  lepidic↔acinar); κ = 0.38 for predominant pattern (Thunnissen 2012).
- **Encoder:** **CTransPath** (frozen self-supervised foundation encoder) + a
  lightweight trained head, so class geometry is dictated almost entirely by the
  head's loss. (UNI / GigaPath / CHIEF are the planned robustness encoders.)
- **Data note:** the repo ships **synthetic** ImageFolder data for smoke tests;
  real tiles/CTransPath checkpoint live on the DGX.

---

## 4. Statistical protocol (reuse this for the mutation paper)

- **50-run K-fold:** 5 folds × 10 seeds = 50 runs per method.
- **Seeds:** `[42, 123, 2026, 7, 17, 71, 202, 512, 1234, 31415]`; 120 epochs.
- **Screening → promotion:** 18-loss family screened on a single split. The 11
  K-fold methods are **not a pure subset** of the 18 (a reviewer will cross-check
  this): **5** are promoted directly (top single-run performer per paradigm:
  V2-Optuna, V3-SubCenters, ArcFace, SphereFace, Softmax-CE) **+ 6 added** to make
  the comparison *harder* — CurricularFace (replacing AdaptiveFace as the P3 rep),
  the 3 equal-budget **"-matched"** re-tunings (ArcFace/CosFace/SphereFace), and 2
  boundary-oriented fuzzy variants (V4-Boundary, V3.1-SubCenter) introduced in
  revision. Total = **11**. (This reconciliation is now spelled out in the abstract
  and intro C1.)
- **Test:** pre-registered **one-sided paired t-test** (H1: FuzzyLoss > baseline),
  paired by (seed, fold). **MDE ≈ d_z ≈ 0.35.**
- **Boundary-pair metrics:** binary F1, AUROC, ECE (calibration).
- Results JSON: `kfold_validation_results.json` (schema:
  `methods[name].runs[*]` with `seed, fold, f1_per_class, boundary_pair_f1,
  boundary_pair_auroc`; plus `mean_f1, std_f1, per_class_mean/std, ece_mean`).

---

## 5. Key results

**11-method aggregate macro-F1 (50-run, mean %):**

| Rank | Method | Macro-F1 |
|---|---|---|
| 1 | **FuzzyArcLoss V2 (Optuna)** | **92.43** |
| 2 | FuzzyArcLoss V4 Boundary | 92.07 |
| 3 | FuzzyArcLoss V3 SubCenters | 92.05 |
| 4 | ArcFace (default) | 92.02 |
| 5 | ArcFace-matched | 91.87 |
| 6 | CurricularFace | 91.75 |
| 7 | FuzzyArcLoss V3.1 SubCenter | 91.75 |
| 8 | SphereFace-matched | 91.62 |
| 9 | CosFace-matched | 91.59 |
| 10 | SphereFace | 91.03 |
| 11 | Softmax-CE | 90.84 |

**Headline claims (as written in the paper — keep them exact):**
- V2 **tops the aggregate ranking** and **significantly beats every non-fuzzy
  baseline except default ArcFace** (+0.6 to +1.6 pp, p<0.05), including all
  three equal-budget *matched* baselines and adaptive CurricularFace.
- V2 vs default **ArcFace = statistical TIE** (macro +0.42 pp, p1 = 0.086;
  cribriform +1.22 pp, p1 = 0.119). **No superiority claim over ArcFace.**
- On the **cribriform boundary**, V2 reaches significance vs **SphereFace**
  (+3.1 pp, p = 0.002), softmax-CE and all matched fixed-margin baselines.
  V2 cribriform per-class F1 = **83.6**.
- V2 vs CurricularFace: macro p1 = 0.025; cribriform +2.0 pp (83.6 vs 81.5),
  borderline p1 = 0.052.
- The two boundary-targeted variants (V4, V3.1) do **not** beat V2 → V2's
  sample-wise membership already captures the exploitable boundary signal.

**External cross-domain replication (5-fold, single-seed).** The external tables
(`tables/tab_external_{cervix,prostate}.tex`) now report **both** a one-sided
paired t-test (`p_t`) **and** a one-sided exact Wilcoxon signed-rank test (`p_W`)
across the 5 folds for V2 vs ArcFace/SphereFace/Softmax-CE, plus class
prevalences in the captions; results are also plotted in
`figures/external_boundary_results.png`.
- **Cervix ASC-US/LSIL** (CRIC cytology, κ = 0.46; ASC-US=606 / LSIL=1,360): V2
  **+2.35 pp over ArcFace, wins 5/5 folds** → supports the boundary hypothesis.
  *(Upgrading to the full 50-run protocol for tighter p-values is still the
  highest-value pending DGX experiment; runbook in `benchmark/README.md`.)*
- **Prostate Gleason 3-vs-4** (SICAPv2, clinical κ = 0.27): near-ceiling **tie
  ~96.1%** — consistent with the mechanism because the benchmark isolates *pure*
  G3/G4 patches (instance-level boundary is crisp).
- **Melanoma MPATH-Dx Class II/III:** prediction only (no open benchmark).

---

## 6. Boundary-dependent transfer hypothesis (falsifiable)

> FuzzyArcLoss V2 should improve over ArcFace **whenever** the task has an
> adjacent class pair with low expert inter-rater agreement (κ ∈ [0.35, 0.50]),
> and should **offer no additional advantage — reducing to a tie with a
> well-tuned fixed margin — when all boundaries are crisp.**

Testable in 3 steps: (1) find the ambiguous pair from published κ; (2) verify
high inter-method per-class F1 range there on a fixed-margin baseline; (3) train
V2 → expect +2–3 pp on the ambiguous pair, tie elsewhere.

**When V2 offers no additional advantage (it ties, it does not lose):** (i) all
boundaries crisp; (ii) ambiguous but data large enough to memorise (fixed margin
can be tuned to match); (iii) signal not morphological in the encoder's feature
space. ⚠️ Writing convention: the phrase **"should not help" was removed
everywhere** — always say V2 *offers no additional advantage / reduces to a tie*.

---

## 7. What was CUT from Paper 2 = your next project

To keep Paper 2 a focused loss paper, the following was **removed** (it is
methodologically separate and was statistically weak for this venue). **This is
the seed of the mutation-prediction paper:**

- **2-additive fuzzy Choquet aggregator** for slide-level MIL readout
  (interpretable via Shapley / interaction indices).
- **PI-ABMIL** baseline; **FC-MIL** architecture.
- **TCGA-LUAD MIL mutation benchmark** (slide-level prediction of driver
  mutations: KRAS, EGFR, TP53, STK11, KEAP1, RBM10).
- The **growth-pattern → mutation** interpretation layer.

**Where the cut material still lives:**
- Sibling paper folder (primary home of the MIL/mutation work):
  `d:\Dropbox\PHD\THESIS PAPERS\SLIMA Pattern-informed multiple-instance learning for interpretable mutation prediction in lung adenocarcinoma a six-level explanation stack\`
- The PhD thesis (connects patterns → mutations → MIL end-to-end).
- Git history of `fuzzyarcloss-paper2` **before** commit `a6b3539` (which pruned
  the MIL figures `abmil_architecture`, `fcmil_architecture`,
  `choquet_interpretability`, `choquet_regimes`, `choquet_variance_map`,
  `choquet_kras_*`, `choquet_rbm10_*`).

---

## 8. Bridge — growth pattern → mutation biology (with existing citations)

These associations are already vetted and cited in `shared/references.bib`. Use
them as the biological prior / evaluation targets for the mutation paper.

| Growth pattern / feature | Associated gene | Reference key |
|---|---|---|
| Solid | **KRAS** (+ TILs) | `rekhtman2013krassolid` |
| Solid / micropapillary / cribriform (high-grade) | **TP53** | `ahn2022tp53highgrade`, `mendoza2024tp53nuclear` |
| Micropapillary | **TP53** | `zhang2022micropaptp53` |
| Papillary | **KEAP1 / NRF2** | `li2011keap1papillary` |
| Lepidic | **EGFR** | `yatabe2005egfrlepidic`, `suda2017egfrlepidic`, `yoshizawa2013egfrmorph` |
| Invasive mucinous (IMA) | **KRAS** | `shim2015krasmucinous`, `cha2017imapatterns`, `hwang2017nonlepidicima` |
| EGFR-mutant co-mutation | **RBM10** | `cao2024rbm10egfr`, `bechara2013rbm10` |
| STK11-mutant morphology | **STK11** | `pineda2025stk11morphology`, `hong2022stk11biomedinf`, `skoulidis2018stk11pd1` |
| KRAS molecular subsets | **KRAS** subtypes | `skoulidis2015krassubsets` |
| EGFR general / therapy | EGFR | `pao2023egfr`, `pao2023sci`, `soria2018flaura` (FLAURA) |
| KRAS G12C therapy | KRAS | `skoulidis2021codebreak` (CodeBreaK) |
| Molecular-integrated classification | — | `hung2020molecular`, `travis2011iaslc`, `who2021thoracic` |

**MIL / aggregation / interpretability reference keys already in the bib:**
`dietterich1997mil`, `ilse2018attention` (ABMIL), `shao2021transmil`,
`lu2021data` (CLAM); Choquet/fuzzy-measure: `grabisch2009aggregation`,
`choquet1954theory`, `grabisch1996application`, `grabisch2000choquet`,
`grabisch1997kapprox`, `marichal2000determination`, `mccurley2023mici`,
`bhowal2021fuzzybach`; attribution: `shapley1953value`, `lundberg2017shap`,
`lundberg2020treeshap`.

---

## 9. Concrete next steps for the mutation-prediction paper

1. **Pipeline:** frozen CTransPath (and/or UNI) tile embeddings → MIL aggregator
   → slide-level mutation prediction. Use FuzzyArcLoss V2 at the tile head to get
   better-separated, boundary-aware embeddings feeding the aggregator.
2. **Interpretable aggregator:** 2-additive fuzzy Choquet integral for
   slide-level readout; report Shapley + interaction indices per growth pattern
   so predictions map back to the biology in §8.
3. **Benchmark:** TCGA-LUAD (slide-level MAF labels; `gdc2024maf`). Targets:
   KRAS, EGFR, TP53, STK11, KEAP1, RBM10. Baselines: ABMIL, TransMIL, CLAM,
   PI-ABMIL, mean/max pooling.
4. **Statistics (reuse Paper 2's protocol):** multi-seed CV, one-sided paired
   tests where directional, AUROC/AUPRC (`davis2006auprc`) + calibration (ECE);
   pre-register the pattern→gene directional hypotheses from §8.
5. **Fix the weakness that got it cut:** the earlier Choquet/MIL result was
   statistically underpowered. Prioritise **power** (more slides/seeds, effect
   sizes, CIs) and a **falsifiable pattern→mutation claim** analogous to Paper
   2's boundary hypothesis.
6. **Robustness:** second encoder (UNI) and/or second cohort to answer the
   "single small dataset" objection.

---

## 10. Reusable assets & where they live

- **Loss family + K-fold protocol:** `scripts/SLIMA_kfold_statistical_validation_*.py`,
  `scripts/SLIMA fuzzy_arc_loss_v2 *.py`, `scripts/SLIMA_ablation_study_loss_functions_*.py`,
  `scripts/SLIMA_optuna_fuzzyarcloss_v2_*.py`.
- **Paired stats helpers:** `scripts/perclass_paired_tests.py` (`paired_stats`,
  one-sided aware), `scripts/run_boundary_paired_all.py`.
- **Multi-domain benchmark driver:** `benchmark/run_benchmark.py`
  (`--losses kfold5 --folds 5 --seeds 10`), `benchmark/dataset_registry.py`,
  `benchmark/losses.py` (`KFOLD5_LOSSES`), `benchmark/model.py`
  (backbones: ctranspath / resnet18 / tiny — **UNI not yet wired**).
- **Figure generators:** `figures/gen_kfold_bars.py`,
  `figures/gen_boundary_vs_size.py`, `figures/gen_boundary_multidomain.py`,
  `figures/gen_boundary_calibration.py` (Pair-F1/AUROC/ECE distributions),
  `figures/gen_external_boundary.py` (external per-fold F1),
  `figures/gen_angular_margin_domains.py` (**new** Fig. 1 motivation:
  coarse-vs-fine geometry + cross-domain adoption map),
  `figures/gen_transfer_kappa_trapezoids.py`, `figures/gen_pattern_fuzzy_sets.py`,
  `tables/gen_tab_perclass_f1.py`.
- **New table:** `tables/tab_class_distribution.tex` (Zenodo-ANORAK per-class
  tile counts, matching thesis Table 3.7: acinar 181, papillary 125, solid 107,
  lepidic 86, micropapillary 69, cribriform 69; total 637).
- **Results JSON (50-run, correct 11-method):**
  `kfold_validation_results.json` (also at `...\SLIMA_papers_2026\SCRIPTS\`).
- **DGX runbook for external 50-run replication:** `benchmark/README.md` §2b.

---

## 11. Open items carried over from Paper 2 (not blocking the mutation work)

- Make `fuzzyarcloss-paper2` **public** before submission (the paper cites it);
  pull the updated `references.bib` into Overleaf and re-run BibTeX (many new
  keys added — see §12); set Overleaf **Main document = `main.tex`**.
- **[REQUIRES DGX]** Cervix 50-run (tighter p-values) + optional prostate 50-run;
  runbook in `benchmark/README.md`.
- **[OPTIONAL, REQUIRES DGX+UNI]** In-domain robustness with a 2nd encoder (UNI
  not yet wired in `benchmark/model.py`) or a 2nd LUAD cohort.
- Author bios: still confirm specialisation / degree institution / years for the
  two MD co-authors (`main.tex`). *(Servio R. Lima Idrobo's bio was already
  rewritten to drop the removed "conclusive/inconclusive" mutation content; the
  "Resilient Systems" group at U. Fribourg under Edy Portmann was verified.)*

**Writing conventions now enforced (keep them):**
- **Never** use the word **"honest"** (replaced by "explicit/transparent/conservative").
- **Never** say V2 **"should not help"** → say *offers no additional advantage /
  reduces to a tie*.
- **Never** claim V2 is **"margin-free" / "removes the margin hyperparameter"** →
  use the tuning-parity framing (principled *schedule shape*; `s,m,τ` tuned under
  equal budget).
- **Never** call angular-margin the **"dominant"** objective *in pathology* → it
  is "emerging" downstream in pathology but "leading" in adjacent open-set domains.

---

## 12. Changelog — manuscript polishing since this handoff was first written

Purely editorial/rigor fixes to Paper 2 (no re-runs; all committed & pushed to
GitHub+GitLab). Listed so the next session knows the manuscript's current state:

- **Margin claim reframed** as tuning parity (abstract, intro C2, results,
  discussion, conclusion); `(s,m,τ)` tuned under the same Optuna budget as every
  matched baseline.
- **"principled" defined** operationally at first use (intro C2) and de-duplicated.
- **18→11 reconciliation** added (abstract + intro C1): 5 promoted + 6 added;
  rationale for the post-hoc boundary variants (V4, V3.1) added in results.
- **"-matched" baselines made self-explanatory**: definition in the K-fold table
  caption + a one-line glossary at first mention.
- **New Fig. 1** (`angular_margin_domains.png`): (a) coarse-vs-fine hypersphere
  geometry with the margin wedge; (b) schematic cross-domain adoption map.
- **Scope paragraph** (intro) expanded with the layered pathology-stack rationale
  (SSL pretraining dominates encoders; CE/MIL dominate downstream; angular-margin
  emerging) — **fully cited**.
- **New citations added** to `shared/references.bib`: cross-domain
  (`desplanques2020ecapa`, `sun2020circle`, `cao2020delg`, `cermak2024wildlife`),
  medical (`alirezazadeh2023chasing`, `zhu2023supervised`), out-of-scope paradigms
  (`khosla2020supcon`, `cui2019classbalanced`, `cao2019ldam`,
  `menon2021logitadjust`, `movshovitz2017proxynca`, `kim2020proxyanchor`), and SSL
  (`chen2020simclr`, `he2020moco`, `caron2021dino`, `zhou2022ibot`).
- **AUROC saturation** explained (results); **boundary-pair (i)/(ii)/(iii)**
  metrics reformatted as a list; **Eq. 8** split to fit the column.
- **Zenodo-ANORAK class-distribution table** added; all stale "~55-tile" minority
  counts corrected to **69**; `boundary_vs_size` figure regenerated.
- **Wilcoxon signed-rank** added alongside paired t-tests in external + per-class
  paired tables, with methodology text.
- **Figures enlarged** (Fig. 2/6/9/10 promoted to `figure*`); Fig. 3 & Fig. 13
  boundary axes corrected to cribriform↔acinar (Kadota 2014 / ANORAK 2024).
- **Bug fixes:** `\softmax` macro made text-safe (`\ensuremath`); tile size
  256→**224**; `subfig`→`subcaption` (`[t]0.48` artefact); `@dataset`→`@misc` in bib.
- **Links** updated to `github.com/slima2/fuzzyarcloss-paper2`.

---

*Generated as a handoff from the Paper 2 (FuzzyArcLoss) session. All numbers,
p-values and reference keys above are transcribed from the Paper 2 source and its
`kfold_validation_results.json`. Last updated after the manuscript-polishing pass
(margin reframing, Fig. 1, scope/citations, and consistency fixes).*
