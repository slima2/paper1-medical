#!/usr/bin/env python
"""Evaluate the CPTAC-LUAD cohort-transfer: AUROC, CI, and regime concordance.

This is the read-out step of the within-domain cohort-transfer test. It is
deliberately decoupled from the lchai model internals: it consumes a
predictions table that the lchai inference produces on the CPTAC WSIs and the
labels table from fetch_cptac_labels.py, and it answers the pre-registered
question -- do the TCGA-LUAD regime assignments survive a genuine distribution
shift, judged by overlapping confidence intervals?

Inputs
------
--preds PATH   parquet/csv with one row per patient and columns
               `patient_id` plus a probability column per gene named
               `<prefix><GENE>` (default prefix `prob_`), e.g. prob_TP53.
               This is exactly what the lchai MIL forward pass emits per slide,
               pooled to the patient.
--labels PATH  defaults to $DATA_ROOT/luad_cptac/labels.parquet

Output
------
$DATA_ROOT/luad_cptac/transfer_results.{csv,json}: per gene the CPTAC AUROC
with a bootstrap 95% CI, n_+, a power flag against the n_+=37 floor, the
reference TCGA-LUAD AUROC/verdict/regime, and a concordance verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ESTIMATION_FLOOR = 37
CONCLUSIVE = 0.70

# Reference TCGA-LUAD results reported in the manuscript (best aggregator).
TCGA = {
    "TP53":  {"auroc": 0.717, "verdict": "Conclusive",   "regime": "(i) saturation"},
    "EGFR":  {"auroc": 0.701, "verdict": "Conclusive",   "regime": "(i) saturation"},
    "STK11": {"auroc": 0.695, "verdict": "Inconclusive",  "regime": "(iii) out-of-ontology"},
    "KRAS":  {"auroc": 0.609, "verdict": "Inconclusive",  "regime": "(ii) synergy"},
    "KEAP1": {"auroc": 0.610, "verdict": "Inconclusive",  "regime": "(iii) out-of-ontology"},
    "RBM10": {"auroc": 0.661, "verdict": "Inconclusive",  "regime": "estimation floor"},
}


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "luad_cptac"


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def bootstrap_ci(y, p, n_boot=2000, seed=42):
    rng = np.random.default_rng(seed)
    n = len(y)
    stats = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        stats.append(roc_auc_score(y[idx], p[idx]))
    if not stats:
        return (float("nan"), float("nan"))
    return (float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--preds", required=True, help="lchai predictions parquet/csv")
    ap.add_argument("--labels", default=None)
    ap.add_argument("--prob-prefix", default="prob_")
    args = ap.parse_args()

    out_dir = resolve_root(args.data_root)
    labels_path = Path(args.labels) if args.labels else out_dir / "labels.parquet"
    if not labels_path.exists():
        sys.exit(f"missing {labels_path}\nrun: python fetch_cptac_labels.py")

    labels = read_table(labels_path)
    if "patient_id" not in labels.columns:
        # fetch_cptac_labels.py may persist patient_id as the frame index.
        labels = labels.reset_index()
    labels = labels.set_index("patient_id")
    preds = read_table(Path(args.preds))
    if "patient_id" not in preds.columns:
        sys.exit("predictions table needs a `patient_id` column")
    preds = preds.set_index("patient_id")

    results = []
    for gene, ref in TCGA.items():
        pcol = f"{args.prob_prefix}{gene}"
        if gene not in labels.columns or pcol not in preds.columns:
            print(f"[skip] {gene}: missing label or prediction column ({pcol})")
            continue
        df = pd.DataFrame({"y": labels[gene], "p": preds[pcol]}).dropna()
        y = df["y"].to_numpy().astype(int)
        p = df["p"].to_numpy().astype(float)
        n_pos = int(y.sum())
        if len(np.unique(y)) < 2:
            print(f"[skip] {gene}: only one class present")
            continue
        auroc = float(roc_auc_score(y, p))
        lo, hi = bootstrap_ci(y, p)
        powered = n_pos > ESTIMATION_FLOOR
        # Concordance: the TCGA point estimate falls inside the CPTAC 95% CI,
        # and the Conclusive/Inconclusive verdict agrees.
        ci_overlap = lo <= ref["auroc"] <= hi
        verdict_cptac = "Conclusive" if auroc >= CONCLUSIVE else "Inconclusive"
        verdict_agrees = verdict_cptac == ref["verdict"]
        results.append({
            "gene": gene,
            "n_eval": int(len(y)), "n_pos": n_pos, "powered": powered,
            "cptac_auroc": round(auroc, 3),
            "cptac_ci95": [round(lo, 3), round(hi, 3)],
            "cptac_verdict": verdict_cptac,
            "tcga_auroc": ref["auroc"], "tcga_verdict": ref["verdict"],
            "regime": ref["regime"],
            "ci_contains_tcga": bool(ci_overlap),
            "verdict_agrees": bool(verdict_agrees),
            "concordant": bool(ci_overlap and verdict_agrees),
            "note": "" if powered else "underpowered (n_+ <= 37): null is about power",
        })

    if not results:
        sys.exit("no genes evaluable; check --preds column names and --prob-prefix")

    res = pd.DataFrame(results)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "transfer_results.csv", index=False)
    (out_dir / "transfer_results.json").write_text(json.dumps(results, indent=2))
    print(res.to_string(index=False))
    print(f"\nwrote {out_dir / 'transfer_results.csv'}")
    powered = res[res["powered"]]
    if len(powered):
        agree = powered["concordant"].mean()
        print(f"\nwell-powered genes ({', '.join(powered['gene'])}): "
              f"{100 * agree:.0f}% concordant with TCGA (CI contains TCGA point "
              f"and verdict agrees)")


if __name__ == "__main__":
    main()
