#!/usr/bin/env python
"""Inverse-probability-of-observation weights for the ECG lab targets.

Domain-C v2 step 3 -- the only piece with no analogue in LUAD/glioma. Labs are
ordered on a NON-RANDOM subset of encounters (sicker patients get NT-proBNP),
so an information estimate on the observed subset is biased. We fit a propensity
model P(lab observed | X) and store stabilised, trimmed IPW weights that
run_ecg_domainC_v2.py applies to every AUROC and condition contrast, following
the correction of Choi et al. (ICYM2I) the manuscript specifies for the
multimodal setting.

Design matrix X (cheap, always available): age, sex, ECG hour-of-day, and a
missingness-free demographic block. Extend with admission service / vitals if
joined upstream.

Guardian: propensity_quality.csv reports the AUROC of the observed-vs-not model
per target. If it is ~0.5 observation is effectively random and IPW collapses
to unweighted (both are reported downstream); if it is high the weights matter.

Inputs (under $DATA_ROOT/ecg/derived/):
    ecg_lab_labels.parquet    with <t>_obs columns from build_ecg_lab_labels.py

Outputs (under $DATA_ROOT/ecg/derived/):
    ipw_weights.parquet       study_id + w_<t> (1 on unobserved rows, unused)
    propensity_quality.csv    per-target observed-model AUROC + observed rate

Usage:
    python fit_lab_propensity.py --targets ntprobnp troponin_i
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

SEED = 42
TRIM = (0.01, 0.99)   # weight trimming percentiles


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "ecg"


def design_matrix(df: pd.DataFrame) -> pd.DataFrame:
    X = pd.DataFrame(index=df.index)
    X["age"] = pd.to_numeric(df.get("age"), errors="coerce")
    X["sex"] = pd.to_numeric(df.get("sex"), errors="coerce")
    t = pd.to_datetime(df.get("ecg_time"), errors="coerce")
    X["hour"] = t.dt.hour.astype(float)
    X["dow"] = t.dt.dayofweek.astype(float)
    # median-impute; keep an explicit missing flag for age
    X["age_missing"] = X["age"].isna().astype(float)
    X = X.fillna(X.median(numeric_only=True)).fillna(0.0)
    return X


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--targets", nargs="+", default=["ntprobnp", "troponin_i"])
    args = ap.parse_args()

    ecg_root = resolve_root(args.data_root)
    labels_path = ecg_root / "derived" / "ecg_lab_labels.parquet"
    if not labels_path.exists():
        sys.exit(f"missing {labels_path}\nrun: python build_ecg_lab_labels.py")

    df = pd.read_parquet(labels_path)
    X = design_matrix(df)

    weights = pd.DataFrame({"study_id": df["study_id"].values})
    quality_rows = []

    for t in args.targets:
        obs_col = f"{t}_obs"
        if obs_col not in df.columns:
            print(f"[warn] {obs_col} not in labels; skipping {t}")
            continue
        obs = df[obs_col].fillna(0).astype(int).to_numpy()
        rate = float(obs.mean())
        if obs.sum() in (0, len(obs)):
            print(f"[warn] {t}: observation is constant ({rate:.3f}); w=1")
            weights[f"w_{t}"] = 1.0
            quality_rows.append({"target": t, "observed_rate": rate, "auroc": np.nan})
            continue

        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=1000, C=1.0))
        # out-of-fold propensities to avoid optimistic weights
        phat = cross_val_predict(clf, X.to_numpy(), obs, cv=5,
                                 method="predict_proba")[:, 1]
        phat = np.clip(phat, 1e-3, 1 - 1e-3)
        auroc = float(roc_auc_score(obs, phat))

        # stabilised IPW on the observed rows: w = P(obs) / phat
        w = np.where(obs == 1, rate / phat, 1.0)
        lo, hi = np.quantile(w[obs == 1], TRIM)
        w = np.clip(w, lo, hi)
        weights[f"w_{t}"] = w

        quality_rows.append({"target": t, "observed_rate": rate, "auroc": auroc})
        guard = "near-random (IPW ~ unweighted)" if auroc < 0.55 else "informative"
        print(f"{t}: observed_rate={rate:.3f}  propensity AUROC={auroc:.3f}  "
              f"[{guard}]  w in [{lo:.2f},{hi:.2f}]")

    out_dir = ecg_root / "derived"
    weights.to_parquet(out_dir / "ipw_weights.parquet")
    pd.DataFrame(quality_rows).to_csv(out_dir / "propensity_quality.csv", index=False)
    print(f"\nwrote {out_dir / 'ipw_weights.parquet'}")
    print(f"wrote {out_dir / 'propensity_quality.csv'}")


if __name__ == "__main__":
    main()
