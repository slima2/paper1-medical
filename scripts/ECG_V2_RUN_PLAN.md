# ECG Domain-C v2 — laboratory-state targets on MIMIC-IV-ECG (NT-proBNP, troponin I, ...)

Roadmap and fixed decisions for the **measured** version of the pre-registered
ECG laboratory targets. v1 (`run_ecg_domainC.py`) ran only the leakage-free
physiological-state targets (sex, age) on PTB-XL. v2 measures the laboratory
targets that motivate the "borderline / untranslatable" bands the manuscript
currently cites from AnyECG-Lab, turning them from predictions into results.

The single new ingredient over v1 is that **labs are observed on a
non-random subset** of ECGs, so the information estimate is biased and must be
reweighted by inverse probability of observation (Choi et al., ICYM2I). This is
exactly the "missingness correction becomes mandatory" case flagged in the
Discussion.

---

## 0. What we are testing (and the three outcomes that are all findings)

Per lab target `Y` (e.g. NT-proBNP, troponin I), read the model-side signature
across the four conditions B2/B3/PI/FC and emit a Conclusive/Inconclusive
verdict + regime, **after IPW reweighting**:

- **Regime (i)** — encoder saturates, vocabulary adds nothing: the electrical
  signal already projects onto the state (expected for NT-proBNP, a BNP proxy).
- **Regime (ii)** — fusion adds (ii-a additive / ii-b non-additive with
  `FC > PI`, `I_jk > 0`): a statement combination carries the state.
- **Regime (iii)** — every condition at chance with the total near the floor:
  the state is not written into the 12-lead signal (expected for troponin I, a
  purely biochemical marker) — the ECG counterpart of KEAP1.

A near-chance result on a **well-powered, IPW-corrected** target is a
regime-(iii) finding, not a data limitation. `propensity_quality.csv` is the
guardian: if the "lab observed" model is itself uninformative the IPW weights
are degenerate and the verdict must be withheld.

---

## 1. Prerequisites (access — do once)

- **PhysioNet credentialed access** + signed DUA for:
  - `mimic-iv-ecg` (waveforms, 12-lead, 10 s, 500 Hz).
  - `mimic-iv` **hosp** module (`labevents.csv.gz`, `d_labitems.csv.gz`,
    `admissions.csv.gz`, `patients.csv.gz`) for lab values + demographics.
- Reuse the DGX `cptac_venv` (has torch, pandas, pyarrow, wfdb, sklearn).
- Keep everything under `$DATA_ROOT/ecg/` next to the PTB-XL v1 artefacts.

```bash
source /home/rapids/notebooks/slima/paper1_nature/glioma/env.sh   # or the ECG env.sh
export DATA_ROOT=/home/rapids/notebooks/slima/data
mkdir -p "$DATA_ROOT/ecg/mimic4/waveforms" "$DATA_ROOT/ecg/mimic4/hosp"
```

---

## 2. Data artefacts to obtain

| Artefact | Path | Notes |
|---|---|---|
| MIMIC-IV-ECG record list + waveforms | `$DATA_ROOT/ecg/mimic4/waveforms/` | `record_list.csv` gives `subject_id, study_id, ecg_time, path` |
| `labevents.csv.gz` | `$DATA_ROOT/ecg/mimic4/hosp/` | huge (~40 GB); can be pre-filtered to the target `itemid`s on download host |
| `d_labitems.csv.gz` | `$DATA_ROOT/ecg/mimic4/hosp/` | resolve `itemid` for NT-proBNP, troponin I, etc. |
| `admissions/patients` | `$DATA_ROOT/ecg/mimic4/hosp/` | demographics + admission context for the propensity model |

Target `itemid`s (verify against `d_labitems`, values drift between releases):
- **NT-proBNP** ~ `50963`; **Troponin T** ~ `51003`; **Troponin I** (site-specific);
  extend to the AnyECG-Lab panel as needed. Always resolve by label match in
  `d_labitems`, do not hard-code blindly.

---

## 3. Pipeline (scripts to create, mirroring the glioma/LUAD naming)

### 3.1 `build_ecg_lab_labels.py` — pair each ECG to a lab value
- Join `record_list` (ECG time) to `labevents` on `subject_id`.
- For each target, take the lab measurement **closest in time within a window**
  (fixed decision: `±24 h` primary, `±72 h` sensitivity) to the ECG.
- Binarise by clinical cutoff (fixed decisions, stated in advance):
  - NT-proBNP: `> 300 pg/mL` (rule-in for HF; sensitivity at 125/450/900).
  - Troponin I: `>` assay 99th-percentile URL (site-specific; store the cutoff).
- Emit `ecg_lab_labels.parquet`: `study_id, subject_id, <target>=0/1/NaN`,
  plus `observed_<target>=0/1` (1 if a lab exists in the window). **NaN label
  where not observed — this is the missingness the IPW step corrects.**
- Print coverage per target: `n_observed`, `n_+`, floor check (`n_+ >= ~37`).

### 3.2 `propagate_scp_to_mimic.py` — the source vocabulary channel P
MIMIC-IV-ECG has **no native SCP-ECG statements**. Train the vocabulary
renderer on PTB-XL (which has the 71 graded statements) and propagate:
- Train a 1D-CNN multi-label head PTB-XL signal → 71 SCP statement likelihoods
  (reuse the v1 encoder backbone; targets = PTB-XL `scp_membership`).
- Apply to every MIMIC-IV-ECG waveform → `scp_propagated.parquet`
  (`study_id × 71`).
- Report `propagation_quality.csv` (held-out PTB-XL AUROC per statement) — the
  same guardian used for VASARI propagation in glioma. A broken P channel must
  not be read as an empty atom.

### 3.3 `fit_lab_propensity.py` — the IPW weights (the v2-specific step)
- For each target, fit a propensity model `P(observed_<target>=1 | X)` where `X`
  = demographics (age, sex), admission type/service, and cheap ECG-derived
  features (heart rate, QRS/QT from the record header if present).
- Store `ipw_weights.parquet`: `w_i = 1 / \hat P(observed | X_i)` on the
  observed subset (stabilised, trimmed at the 1st/99th percentile).
- Emit `propensity_quality.csv` (AUROC of the observed-vs-not model). If it is
  at chance, observation is ~random and IPW ≈ unweighted (report both).

### 3.4 `run_ecg_domainC_v2.py` — the four-condition run with IPW
- Port `run_ecg_domainC.py`; encoder E = same 1D-CNN, vocabulary P =
  `scp_propagated`, targets from `ecg_lab_labels`.
- **Every metric (AUROC and the condition contrasts) is IPW-weighted** using
  `ipw_weights`; report unweighted alongside as a sensitivity row.
- 5-fold patient-disjoint on `subject_id`.
- Emit `results_v2.json`, `regimes_v2.json` (verdict + directional regime per
  the `tab:atom_operationalization` rules), `choquet_interactions_v2.csv`.

---

## 4. Fixed decisions (pre-register before running)

1. Targets, cutoffs and the `±24 h` pairing window are fixed above; `±72 h` is a
   pre-declared sensitivity, not a search.
2. Regime assignment is **directional** (sign of the condition contrasts and of
   `I_jk`), never absolute AUROC — identical to LUAD/ECG-v1/glioma.
3. IPW is the primary estimate; unweighted is the sensitivity. A target is only
   read when `propensity_quality` and `propagation_quality` both clear their
   guardians.
4. Expected regimes (stated in advance): NT-proBNP → (i); troponin I → (iii);
   the borderline AnyECG-Lab band → mixed (ii)/(iii).

---

## 5. Runbook (unattended, `nohup`)

```bash
source .../env.sh; cd "$DATA_ROOT/ecg"
python build_ecg_lab_labels.py --targets ntprobnp troponin_i --window-h 24
python propagate_scp_to_mimic.py --smoke && python propagate_scp_to_mimic.py
python fit_lab_propensity.py --targets ntprobnp troponin_i
python run_ecg_domainC_v2.py --smoke --targets ntprobnp
LOG="ecg_v2_$(date +%Y%m%d_%H%M%S).log"
nohup python run_ecg_domainC_v2.py --targets ntprobnp troponin_i > "$LOG" 2>&1 &
echo "PID $! ; log $(pwd)/$LOG"
```

---

## 6. How this lands in the paper

- Moves NT-proBNP/troponin I from "reported by AnyECG-Lab (not run here)" to a
  measured row in `tab:ecg_domainc` (or a companion table), with IPW verdicts.
- Makes the ECG domain carry a genuine **regime-(iii)** case (troponin I) — the
  ECG analogue of KEAP1 — instead of leaning on the literature for the
  untranslatable end.
- Exercises the missingness correction the Methods promise, closing the last
  "predicted, not measured" item in `tab:crossdomain_protocol` for ECG.
