#!/usr/bin/env python
"""Propagate the 71 SCP-ECG statements from PTB-XL onto MIMIC-IV-ECG.

Domain-C v2 step 2. MIMIC-IV-ECG has no native SCP-ECG statement annotations,
so the expert source-vocabulary channel P cannot be read directly the way it is
for PTB-XL. We therefore train the vocabulary *renderer* where the labels exist
(PTB-XL, from build_ptbxl_vocabulary.py) and apply it where they do not
(MIMIC-IV-ECG) -- exactly the ANORAK->TCGA and VASARI->BraTS propagation used in
the other two domains.

    renderer:  1D-CNN(raw 12-lead) -> 71 statement likelihoods in [0,1]
    train on:  PTB-XL scp_membership (soft targets, BCE)
    apply to:  every MIMIC-IV-ECG waveform (resampled 500->100 Hz)

Guardian: propagation_quality.csv reports held-out PTB-XL AUROC per statement.
A statement the renderer cannot recover is a broken channel, not an empty atom;
run_ecg_domainC_v2.py must cite this file next to any regime-(iii) verdict.

Inputs:
    $DATA_ROOT/ecg/derived/scp_membership.parquet     (PTB-XL, 71 statements)
    $DATA_ROOT/ecg/ptb-xl/                              (PTB-XL waveforms, 100 Hz)
    $DATA_ROOT/ecg/mimic4/waveforms/record_list.csv    (MIMIC records, 500 Hz)

Outputs, under $DATA_ROOT/ecg/derived/:
    scp_propagated.parquet     study_id x 71 statement likelihoods for MIMIC
    propagation_quality.csv    per-statement held-out PTB-XL AUROC + prevalence

Usage (DGX, after `source env.sh`):
    python propagate_scp_to_mimic.py --smoke     # tiny train + 500 MIMIC records
    python propagate_scp_to_mimic.py             # full
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
except Exception as exc:  # pragma: no cover
    sys.exit(f"PyTorch is required: {exc}")

try:
    import wfdb
except Exception as exc:  # pragma: no cover
    sys.exit(f"wfdb is required (pip install wfdb): {exc}")

from sklearn.metrics import roc_auc_score

SEED = 42
KEY_COLS = ["patient_id", "strat_fold", "age", "sex", "filename_lr", "filename_hr"]


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "ecg"


class StatementRenderer(nn.Module):
    """1D-CNN over the raw 12-lead signal -> per-statement likelihoods."""

    def __init__(self, n_stmt: int, in_ch: int = 12, dim: int = 128):
        super().__init__()
        def block(ci, co, k=7, s=2):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, stride=s, padding=k // 2),
                nn.BatchNorm1d(co), nn.ReLU(inplace=True),
            )
        self.net = nn.Sequential(
            block(in_ch, 32), block(32, 64), block(64, 128),
            block(128, 128), block(128, dim),
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(dim, n_stmt)

    def forward(self, x):
        h = self.pool(self.net(x)).squeeze(-1)
        return self.head(h)


def load_ptbxl_waveforms(ptbxl_dir: Path, filenames: pd.Series, limit: int | None) -> np.ndarray:
    n = len(filenames) if limit is None else min(limit, len(filenames))
    sig = np.zeros((n, 12, 1000), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        data, _ = wfdb.rdsamp(str(ptbxl_dir / filenames.iloc[i]))
        x = data.T.astype(np.float32)
        if x.shape[1] != 1000:
            x = x[:, :1000] if x.shape[1] > 1000 else np.pad(x, ((0, 0), (0, 1000 - x.shape[1])))
        sig[i] = x
        if (i + 1) % 2000 == 0:
            print(f"  PTB-XL {i + 1}/{n} ({time.time() - t0:.0f}s)", end="\r")
    print()
    return sig


def load_mimic_waveform(path: str, wf_dir: Path) -> np.ndarray | None:
    """Read one MIMIC-IV-ECG record (500 Hz, 5000 samples) -> [12, 1000] at 100 Hz."""
    rec = str(wf_dir / path)
    try:
        data, fields = wfdb.rdsamp(rec)
    except Exception:
        return None
    x = data.T.astype(np.float32)                 # [12, T]
    if x.shape[0] != 12:
        return None
    # resample to 1000 samples (10 s @ 100 Hz) by simple decimation/interp
    T = x.shape[1]
    if T >= 1000:
        idx = np.linspace(0, T - 1, 1000).astype(int)
        x = x[:, idx]
    else:
        x = np.pad(x, ((0, 0), (0, 1000 - T)))
    return np.nan_to_num(x)


def znorm(sig: np.ndarray) -> np.ndarray:
    mu = sig.mean(axis=(0, 2), keepdims=True)
    sd = sig.std(axis=(0, 2), keepdims=True) + 1e-6
    return (sig - mu) / sd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ecg_root = resolve_root(args.data_root)
    memb_path = ecg_root / "derived" / "scp_membership.parquet"
    if not memb_path.exists():
        sys.exit(f"missing {memb_path}\nrun: python build_ptbxl_vocabulary.py")

    df = pd.read_parquet(memb_path)
    stmt_cols = [c for c in df.columns if c not in KEY_COLS]
    print(f"{len(df)} PTB-XL records x {len(stmt_cols)} statements")

    # --- train the renderer on PTB-XL (folds 1-8 train, 9 val) -------------
    limit = 2000 if args.smoke else None
    if args.smoke:
        df = df.iloc[:2000].copy()
        args.epochs = min(args.epochs, 3)
    Xp = znorm(load_ptbxl_waveforms(ecg_root / "ptb-xl", df["filename_lr"], limit))
    Yp = df[stmt_cols].to_numpy(dtype=np.float32)[: len(Xp)]
    fold = df["strat_fold"].to_numpy()[: len(Xp)]
    tr = fold <= 8
    va = fold == 9

    Xt = torch.from_numpy(Xp); Yt = torch.from_numpy(Yp)
    model = StatementRenderer(len(stmt_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    lossf = nn.BCEWithLogitsLoss()
    loader = DataLoader(TensorDataset(Xt[tr], Yt[tr]), batch_size=args.batch_size,
                        shuffle=True)
    for ep in range(args.epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss = lossf(model(xb.to(device)), yb.to(device))
            loss.backward(); opt.step()
        print(f"  epoch {ep + 1}/{args.epochs} loss {loss.item():.4f}", end="\r")
    print()

    # --- held-out quality on PTB-XL val fold ------------------------------
    model.eval()
    with torch.no_grad():
        pv = torch.sigmoid(model(Xt[va].to(device))).cpu().numpy()
    yv = Yp[va]
    rows = []
    for j, name in enumerate(stmt_cols):
        yb = (yv[:, j] >= 0.5).astype(int)
        prev = float(yb.mean())
        auroc = float(roc_auc_score(yb, pv[:, j])) if 0 < yb.sum() < len(yb) else np.nan
        rows.append({"statement": name, "val_auroc": auroc, "prevalence": prev})
    quality = pd.DataFrame(rows)
    print(f"propagation quality: median AUROC {quality['val_auroc'].median():.3f}")

    # --- apply to MIMIC-IV-ECG --------------------------------------------
    wf_dir = ecg_root / "mimic4" / "waveforms"
    rl = pd.read_csv(wf_dir / "record_list.csv")
    rl = rl[["study_id", "path"]].dropna()
    if args.smoke:
        rl = rl.iloc[:500].copy()
    print(f"propagating to {len(rl):,} MIMIC records ...")

    study_ids, feats = [], []
    t0 = time.time()
    buf, buf_ids = [], []

    def flush():
        if not buf:
            return
        xb = znorm(np.stack(buf))
        with torch.no_grad():
            p = torch.sigmoid(model(torch.from_numpy(xb).to(device))).cpu().numpy()
        feats.append(p); study_ids.extend(buf_ids)
        buf.clear(); buf_ids.clear()

    for i, (_, r) in enumerate(rl.iterrows()):
        x = load_mimic_waveform(r["path"], wf_dir)
        if x is None:
            continue
        buf.append(x); buf_ids.append(int(r["study_id"]))
        if len(buf) >= 256:
            flush()
        if (i + 1) % 2000 == 0:
            print(f"  MIMIC {i + 1}/{len(rl)} ({time.time() - t0:.0f}s)", end="\r")
    flush()
    print()

    P = np.concatenate(feats) if feats else np.zeros((0, len(stmt_cols)))
    prop = pd.DataFrame(P, columns=stmt_cols)
    prop.insert(0, "study_id", study_ids)

    out_dir = ecg_root / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    prop.to_parquet(out_dir / "scp_propagated.parquet")
    quality.to_csv(out_dir / "propagation_quality.csv", index=False)
    print(f"wrote {out_dir / 'scp_propagated.parquet'}  "
          f"({len(prop):,} studies x {len(stmt_cols)} statements)")
    print(f"wrote {out_dir / 'propagation_quality.csv'}")


if __name__ == "__main__":
    main()
