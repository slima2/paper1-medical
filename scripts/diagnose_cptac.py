#!/usr/bin/env python
"""Differential diagnosis for the CPTAC-LUAD transfer collapse (no re-inference).

Consumes cptac_predictions_per_slide.parquet + labels.parquet and asks, for the
one well-powered gene (TP53) and the rest:

  1. Signal direction: mean predicted prob for mutant vs wild-type patients.
     A collapse to chance can be (a) predictions squashed to the prior [features
     off-distribution] or (b) predictions well spread but uncorrelated with the
     label [a cohort-specific morphological correlate / batch effect]. We report
     the spread and the pos/neg means so the two are distinguishable.
  2. Site stratification (C3L vs C3N tissue-source prefixes): if AUROC differs by
     site, that points at site signatures (Howard 2021) rather than biology.
  3. Tile-quality gate: AUROC restricted to slides with many tiles (drops the
     low-content / low-resolution slides flagged during inference).
  4. Precision@k: of the k patients the model is most confident are mutant, how
     many actually are -- an intuitive read that does not depend on the AUROC.

Everything is patient-level mean-pool unless noted.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

GENES = ["TP53", "EGFR", "STK11", "KRAS", "KEAP1", "RBM10"]


def resolve_root(cli_root):
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "luad_cptac"


def read_table(path):
    return pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)


def safe_auroc(y, p):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def patient_mean(slide_df, gene):
    pcol = f"prob_{gene}"
    return slide_df.dropna(subset=[pcol]).groupby("patient_id")[pcol].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--per-slide", default=None)
    ap.add_argument("--labels", default=None)
    ap.add_argument("--tile-gate", type=int, default=2000,
                    help="min tiles for the tile-quality-gated AUROC")
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

    n_slides = len(slide_df)
    n_pat = slide_df["patient_id"].nunique()
    print(f"per-slide table: {n_slides} slides, {n_pat} patients")
    spp = slide_df.groupby("patient_id").size()
    print(f"slides/patient: median={spp.median():.0f} range=[{spp.min()},{spp.max()}]")
    tstats = slide_df["n_tiles"].describe()[["min", "25%", "50%", "75%", "max"]]
    print("n_tiles per slide:", {k: int(v) for k, v in tstats.items()})

    # 1. Signal direction + spread ------------------------------------------
    print("\n== 1. signal direction (patient-level mean-pool) ==")
    print(f"{'gene':6} {'n_pos':>5} {'auroc':>6} {'mean_pos':>8} {'mean_neg':>8} "
          f"{'delta':>7} {'p5':>6} {'p95':>6}")
    for gene in GENES:
        s = patient_mean(slide_df, gene)
        df = pd.DataFrame({"p": s}).join(labels[gene].rename("y"), how="inner").dropna()
        y = df["y"].to_numpy().astype(int); p = df["p"].to_numpy().astype(float)
        if len(np.unique(y)) < 2:
            continue
        mpos, mneg = p[y == 1].mean(), p[y == 0].mean()
        print(f"{gene:6} {int(y.sum()):>5} {safe_auroc(y, p):>6.3f} "
              f"{mpos:>8.3f} {mneg:>8.3f} {mpos - mneg:>+7.3f} "
              f"{np.percentile(p, 5):>6.3f} {np.percentile(p, 95):>6.3f}")
    print("read: delta>0 with wide p5..p95 = spread but ~uncorrelated (batch/cohort "
          "correlate); p5..p95 narrow = squashed to prior (features off-distribution).")

    # 2. Site stratification -------------------------------------------------
    print("\n== 2. site stratification (tissue-source prefix) ==")
    site = slide_df["patient_id"].str.extract(r"^(C3[LN])")[0]
    for gene in GENES:
        s = patient_mean(slide_df, gene)
        df = pd.DataFrame({"p": s})
        df["site"] = df.index.str.extract(r"^(C3[LN])")[0].values
        df = df.join(labels[gene].rename("y"), how="inner").dropna()
        parts = []
        for sv, sub in df.groupby("site"):
            a = safe_auroc(sub["y"], sub["p"])
            parts.append(f"{sv}: AUROC={a:.3f} (n={len(sub)}, n+={int(sub['y'].sum())})")
        print(f"{gene:6} " + " | ".join(parts))

    # 3. Tile-quality gate ---------------------------------------------------
    gate = args.tile_gate
    print(f"\n== 3. tile-quality gate (slides with > {gate} tiles only) ==")
    hi = slide_df[slide_df["n_tiles"] > gate]
    kept = hi["patient_id"].nunique()
    print(f"kept {len(hi)}/{n_slides} slides across {kept} patients")
    for gene in GENES:
        s = patient_mean(hi, gene)
        df = pd.DataFrame({"p": s}).join(labels[gene].rename("y"), how="inner").dropna()
        if df["y"].nunique() < 2:
            continue
        print(f"{gene:6} AUROC={safe_auroc(df['y'], df['p']):.3f} "
              f"(n={len(df)}, n+={int(df['y'].sum())})")

    # 4. Precision@k for the well-powered gene ------------------------------
    print("\n== 4. precision@k (TP53, patient-level mean-pool) ==")
    s = patient_mean(slide_df, "TP53")
    df = pd.DataFrame({"p": s}).join(labels["TP53"].rename("y"), how="inner").dropna()
    df = df.sort_values("p", ascending=False)
    base = df["y"].mean()
    for k in (10, 20, 30, 55):
        k = min(k, len(df))
        prec = df["y"].head(k).mean()
        print(f"top-{k:>2}: precision={prec:.3f}  (base rate={base:.3f}, "
              f"lift={prec / base:.2f}x)")
    print("read: precision ~ base rate = no usable ranking; >> base rate = some signal "
          "concentrated at the top even if global AUROC is low.")


if __name__ == "__main__":
    main()
