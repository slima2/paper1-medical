#!/usr/bin/env python
"""Aggregation-sensitivity check for the CPTAC-LUAD cohort-transfer.

The patient-level read-out in eval_cptac_transfer.py mean-pools every slide a
patient contributed. CPTAC ships ~5 slides/patient, many of them
normal-adjacent or low-tile, so a plain mean can bury a focal mutant signal --
whereas the TCGA-LUAD reference is effectively one diagnostic slide per patient.
Before we attribute a transfer failure to genuine cohort-relativity, we re-read
the SAME per-slide predictions under several a-priori aggregators and also at the
slide level, to separate a biology/cohort effect from an aggregation artefact.

No re-inference: this consumes cptac_predictions_per_slide.parquet.

Aggregators (patient level):
    mean     mean over the patient's slides (== eval_cptac_transfer default)
    max      max over slides (does mutant morphology appear anywhere?)
    top2     mean of the two highest-scoring slides
    top3     mean of the three highest-scoring slides
    best     single slide with the most tiles (closest to a TCGA 1-slide/patient)
Plus a slide-level AUROC where each slide inherits its patient's genotype.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ESTIMATION_FLOOR = 37
CONCLUSIVE = 0.70
GENES = ["TP53", "EGFR", "STK11", "KRAS", "KEAP1", "RBM10"]

# Reference TCGA-LUAD (best aggregator), mirrors eval_cptac_transfer.py.
TCGA = {
    "TP53": 0.717, "EGFR": 0.701, "STK11": 0.695,
    "KRAS": 0.609, "KEAP1": 0.610, "RBM10": 0.661,
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


def topk_mean(s: pd.Series, k: int) -> float:
    return float(s.nlargest(k).mean())


def aggregate(slide_df: pd.DataFrame, gene: str, how: str) -> pd.Series:
    pcol = f"prob_{gene}"
    g = slide_df.dropna(subset=[pcol])
    if how == "mean":
        return g.groupby("patient_id")[pcol].mean()
    if how == "max":
        return g.groupby("patient_id")[pcol].max()
    if how == "top2":
        return g.groupby("patient_id")[pcol].apply(lambda s: topk_mean(s, 2))
    if how == "top3":
        return g.groupby("patient_id")[pcol].apply(lambda s: topk_mean(s, 3))
    if how == "best":  # slide with most tiles
        idx = g.groupby("patient_id")["n_tiles"].idxmax()
        return g.loc[idx].set_index("patient_id")[pcol]
    raise ValueError(how)


def score(y: np.ndarray, p: np.ndarray, gene: str) -> dict:
    n_pos = int(y.sum())
    auroc = float(roc_auc_score(y, p))
    lo, hi = bootstrap_ci(y, p)
    return {
        "n": int(len(y)), "n_pos": n_pos,
        "auroc": round(auroc, 3), "ci_lo": round(lo, 3), "ci_hi": round(hi, 3),
        "verdict": "Conclusive" if auroc >= CONCLUSIVE else "Inconclusive",
        "powered": n_pos > ESTIMATION_FLOOR,
        "ci_contains_tcga": bool(lo <= TCGA[gene] <= hi) if not np.isnan(lo) else False,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--per-slide", default=None,
                    help="defaults to $DATA_ROOT/luad_cptac/cptac_predictions_per_slide.parquet")
    ap.add_argument("--labels", default=None)
    args = ap.parse_args()

    out_dir = resolve_root(args.data_root)
    per_slide = Path(args.per_slide) if args.per_slide else out_dir / "cptac_predictions_per_slide.parquet"
    labels_path = Path(args.labels) if args.labels else out_dir / "labels.parquet"
    for pth in (per_slide, labels_path):
        if not pth.exists():
            sys.exit(f"missing {pth}")

    slide_df = read_table(per_slide)
    labels = read_table(labels_path)
    if "patient_id" not in labels.columns:
        labels = labels.reset_index()
    labels = labels.set_index("patient_id")

    aggregators = ["mean", "max", "top2", "top3", "best"]
    rows = []

    # Slide-level: each slide inherits its patient's genotype (closest to a
    # per-slide TCGA read-out and the largest sample size available).
    for gene in GENES:
        pcol = f"prob_{gene}"
        g = slide_df.dropna(subset=[pcol]).copy()
        g["y"] = g["patient_id"].map(labels[gene])
        g = g.dropna(subset=["y"])
        y = g["y"].to_numpy().astype(int)
        p = g[pcol].to_numpy().astype(float)
        if len(np.unique(y)) < 2:
            continue
        rows.append({"gene": gene, "agg": "slide", "tcga_auroc": TCGA[gene], **score(y, p, gene)})

    # Patient-level aggregators.
    for how in aggregators:
        for gene in GENES:
            s = aggregate(slide_df, gene, how)
            df = pd.DataFrame({"p": s}).join(labels[gene].rename("y"), how="inner").dropna()
            y = df["y"].to_numpy().astype(int)
            p = df["p"].to_numpy().astype(float)
            if len(np.unique(y)) < 2:
                continue
            rows.append({"gene": gene, "agg": how, "tcga_auroc": TCGA[gene], **score(y, p, gene)})

    res = pd.DataFrame(rows)
    order = {"slide": 0, "mean": 1, "top2": 2, "top3": 3, "max": 4, "best": 5}
    res = res.sort_values(["gene", "agg"], key=lambda c: c.map(order) if c.name == "agg" else c)
    out_dir.mkdir(parents=True, exist_ok=True)
    res.to_csv(out_dir / "reaggregation_results.csv", index=False)

    cols = ["gene", "agg", "n", "n_pos", "powered", "auroc", "ci_lo", "ci_hi",
            "verdict", "tcga_auroc", "ci_contains_tcga"]
    print(res[cols].to_string(index=False))
    print(f"\nwrote {out_dir / 'reaggregation_results.csv'}")

    # Focus on the only well-powered gene(s): does ANY a-priori aggregator
    # recover the TCGA point inside the CPTAC CI?
    for gene in GENES:
        sub = res[(res["gene"] == gene) & (res["powered"])]
        if len(sub):
            best = sub.loc[sub["auroc"].idxmax()]
            recov = sub["ci_contains_tcga"].any()
            print(f"[{gene}] well-powered: best AUROC={best['auroc']} via '{best['agg']}'; "
                  f"TCGA point recovered by some aggregator: {recov}")


if __name__ == "__main__":
    main()
