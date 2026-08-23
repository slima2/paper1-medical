#!/usr/bin/env python
"""Train the VASARI-feature classifier on the annotated seed and propagate the
~30 features to every BraTS subject -> vasari_propagated.parquet.

This is the ANORAK -> TCGA-LUAD move in MRI: fetch_vasari.py builds the seed
table (subject x feature on the 178 LGG + 75 GBM TCIA-annotated cases); this
script learns feature -> from -> the same 2.5D CNN + ABMIL encoder used by the
domain-B run, then predicts the features for the full cohort so run_glioma_%
domainB.py has a source channel P defined on all subjects.

Pipeline:
  1. seed features (vasari_features.parquet), min-max scaled to [0,1] per feature;
  2. seed subjects are TCGA ids -> mapped to BraTS subject_id via the crosswalk,
     keeping those with a prepared slice tensor;
  3. MRIEncoder + sigmoid multi-output head, trained to regress the features;
     k-fold on the seed reports held-out per-feature quality (MAE, and AUROC on
     a median-split binarisation) so weak features can be dropped from P;
  4. retrain on the full seed and predict features for every manifest subject.

Usage (on the DGX, after prepare_glioma_slices.py and fetch_vasari.py):
    python propagate_vasari.py
    python propagate_vasari.py --smoke           # few epochs, quick check
    python propagate_vasari.py --min-auroc 0.6   # drop features below this on seed CV
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    sys.exit(f"PyTorch is required: {exc}")

from sklearn.metrics import roc_auc_score

# reuse the exact encoder + constants of the domain-B run
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_glioma_domainB import (  # noqa: E402
    MRIEncoder, SLICES_PER_SUBJECT, N_SEQUENCES, SLICE_HW,
)
from build_glioma_labels import load_crosswalk  # noqa: E402

SEED = 42
N_FOLDS = 5


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "glioma"


class SliceRegDataset(Dataset):
    def __init__(self, slice_paths: list[str], Y: np.ndarray):
        self.slice_paths = slice_paths
        self.Y = torch.from_numpy(Y.astype(np.float32))

    def __len__(self):
        return len(self.slice_paths)

    def __getitem__(self, i):
        arr = np.load(self.slice_paths[i]).astype(np.float32)
        exp = (SLICES_PER_SUBJECT, N_SEQUENCES, SLICE_HW, SLICE_HW)
        if arr.shape != exp:
            a = np.zeros(exp, dtype=np.float32)
            k = min(arr.shape[0], SLICES_PER_SUBJECT)
            a[:k] = arr[:k, :N_SEQUENCES, :SLICE_HW, :SLICE_HW]
            arr = a
        return torch.from_numpy(arr), self.Y[i]


class FeatureNet(nn.Module):
    def __init__(self, n_out: int, dim: int = 128):
        super().__init__()
        self.enc = MRIEncoder(dim=dim)
        self.head = nn.Linear(dim, n_out)

    def forward(self, x):
        return torch.sigmoid(self.head(self.enc(x)))


def load_seed(glioma_root: Path) -> tuple[pd.DataFrame, list[str]]:
    path = glioma_root / "vasari" / "vasari_features.parquet"
    if not path.exists():
        sys.exit(f"missing {path}; run fetch_vasari.py first")
    df = pd.read_parquet(path)
    drop = [c for c in ("source_file",) if c in df.columns]
    df = df.drop(columns=drop)
    feats = {}
    for c in df.columns:
        col = pd.to_numeric(df[c], errors="coerce")
        if col.notna().mean() < 0.5:
            continue
        col = col.fillna(col.median())
        lo, hi = col.min(), col.max()
        feats[c] = (col - lo) / (hi - lo + 1e-9)
    scaled = pd.DataFrame(feats, index=df.index)
    print(f"seed: {len(scaled)} cases x {scaled.shape[1]} usable features")
    return scaled, list(scaled.columns)


def train(net, ds, epochs, bs, lr, device):
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-4)
    lossf = nn.MSELoss()
    dl = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=2)
    for _ in range(epochs):
        net.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(net(xb), yb)
            loss.backward()
            opt.step()


def predict(net, ds, bs, device):
    net.eval()
    out = []
    with torch.no_grad():
        for xb, _ in DataLoader(ds, batch_size=bs, num_workers=2):
            out.append(net(xb.to(device)).cpu().numpy())
    return np.concatenate(out, 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-auroc", type=float, default=0.0,
                    help="drop features whose seed-CV AUROC is below this from P")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")
    if args.smoke:
        args.epochs = min(args.epochs, 3)

    glioma_root = resolve_root(args.data_root)
    seed_df, feat_cols = load_seed(glioma_root)

    manifest = pd.read_parquet(glioma_root / "derived" / "manifest.parquet")
    path_of = dict(zip(manifest["subject_id"], manifest["slices_path"]))

    # map seed TCGA ids -> slice path. Two routes, identity first:
    #   1) identity: the BraTS-TCGA-LGG/GBM NIfTI are named by TCGA id, so a seed
    #      subject usually already has a prepared slice tensor under its own id
    #      -> no crosswalk needed (this is the path the MGMT v2 run uses).
    #   2) crosswalk: fall back to the BraTS<->TCIA XLSX for seed subjects whose
    #      imaging is only present under a BraTS2021_XXXXX folder.
    xwalk = load_crosswalk(glioma_root)
    tcga_to_brats = dict(zip(xwalk["tcga_id"], xwalk["subject_id"])) if len(xwalk) else {}
    seed_rows = []
    n_identity = n_xwalk = 0
    for tcga_id, row in seed_df.iterrows():
        sid = str(tcga_id).upper()
        if sid in path_of:
            seed_rows.append((path_of[sid], row.values)); n_identity += 1
            continue
        brats = tcga_to_brats.get(sid)
        if brats and brats in path_of:
            seed_rows.append((path_of[brats], row.values)); n_xwalk += 1
    print(f"seed mapped to imaging: {n_identity} by identity, {n_xwalk} via crosswalk")
    if len(seed_rows) < 20:
        sys.exit(f"only {len(seed_rows)} seed cases have prepared slices; "
                 "check prepare_glioma_slices.py coverage (and the crosswalk if "
                 "the seed imaging is under BraTS2021_* folders)")
    seed_paths = [r[0] for r in seed_rows]
    seed_Y = np.stack([r[1] for r in seed_rows]).astype(np.float32)
    print(f"trainable seed with imaging: {len(seed_paths)} cases")

    # ---- seed CV quality report ----------------------------------------
    from sklearn.model_selection import KFold
    kf = KFold(n_splits=min(N_FOLDS, len(seed_paths)), shuffle=True, random_state=SEED)
    preds = np.zeros_like(seed_Y)
    for tr, te in kf.split(seed_paths):
        net = FeatureNet(len(feat_cols)).to(device)
        train(net, SliceRegDataset([seed_paths[i] for i in tr], seed_Y[tr]),
              args.epochs, args.batch_size, args.lr, device)
        preds[te] = predict(net, SliceRegDataset([seed_paths[i] for i in te], seed_Y[te]),
                            args.batch_size, device)
    mae = np.mean(np.abs(preds - seed_Y), axis=0)
    aucs = []
    for j in range(len(feat_cols)):
        yb = (seed_Y[:, j] >= np.median(seed_Y[:, j])).astype(int)
        try:
            aucs.append(roc_auc_score(yb, preds[:, j]) if len(np.unique(yb)) == 2 else np.nan)
        except Exception:
            aucs.append(np.nan)
    aucs = np.array(aucs)
    qual = pd.DataFrame({"feature": feat_cols, "cv_mae": mae, "cv_auroc": aucs})
    qual = qual.sort_values("cv_auroc", ascending=False)
    print("\nseed-CV propagation quality (top/bottom):")
    print(qual.head(8).to_string(index=False))
    print("...")
    print(qual.tail(5).to_string(index=False))

    keep = [f for f, a in zip(feat_cols, aucs) if not (a == a) or a >= args.min_auroc] \
        if args.min_auroc > 0 else feat_cols
    if len(keep) < len(feat_cols):
        print(f"[info] dropping {len(feat_cols) - len(keep)} features below "
              f"AUROC {args.min_auroc}")

    # ---- retrain on all seed, predict all manifest subjects ------------
    net = FeatureNet(len(feat_cols)).to(device)
    train(net, SliceRegDataset(seed_paths, seed_Y),
          args.epochs, args.batch_size, args.lr, device)
    all_paths = manifest["slices_path"].astype(str).tolist()
    all_pred = predict(net, SliceRegDataset(all_paths, np.zeros((len(all_paths), len(feat_cols)))),
                       args.batch_size, device)

    prop = pd.DataFrame(all_pred, columns=feat_cols, index=manifest["subject_id"].values)
    prop.index.name = "subject_id"
    prop = prop[keep]
    out = glioma_root / "vasari" / "vasari_propagated.parquet"
    out.parent.mkdir(parents=True, exist_ok=True)
    prop.to_parquet(out)
    qual.to_csv(glioma_root / "vasari" / "propagation_quality.csv", index=False)
    print(f"\nwrote {out}  ({prop.shape[0]} subjects x {prop.shape[1]} features)")
    print(f"wrote {glioma_root / 'vasari' / 'propagation_quality.csv'}")


if __name__ == "__main__":
    main()
