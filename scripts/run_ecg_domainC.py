#!/usr/bin/env python
"""Domain-C run: ECG -> latent physiological state, on PTB-XL alone.

This instantiates the translatability framework of the manuscript in the
electrocardiogram domain, using only the openly downloaded PTB-XL data
(no MIMIC/EchoNext credentialing required). It is the ECG counterpart of
the TCGA-LUAD pipeline:

    source channel  P : 71 graded SCP-ECG statements   (scp_membership.parquet)
    encoder channel E : a learned 1D-CNN embedding of the raw 12-lead signal
    target          Y : a physiological state NOT derivable from P

Target choice matters. The PTB-XL diagnostic super/sub-classes are
aggregations of the 71 SCP statements, so predicting them from P would be
label leakage (source == target). The clean, leakage-free targets present
in PTB-XL are the physiological-state labels age and sex -- the direct
analogue of the retinal->systemic mapping the manuscript cites. The
laboratory/SHD targets of the pre-registered protocol need MIMIC-IV-ECG +
labevents (credentialed) or EchoNext (DUA) and are out of scope here.

Faithful to the manuscript, the partial-information decomposition is
reported through its three MODEL-SIDE SIGNATURES (which condition wins,
B3-vs-B2, PI/FC-vs-B2, and the Choquet interaction indices) rather than as
numeric PID atoms, and a Conclusive/Inconclusive verdict is emitted per
target at mean AUROC >= 0.70 under the best aggregator.

Conditions (mirroring tab:mutation_luad):
    B2  encoder embedding only            (ceiling baseline)
    B3  SCP statement channel only
    PI  concat fusion  [E ; P]
    FC  E  +  2-additive fuzzy Choquet over the 71 statements

Splits: PTB-XL ships 10 patient-disjoint strat_folds; we group them into
five held-out folds ({1,2},{3,4},{5,6},{7,8},{9,10}) to reproduce the
5-fold patient-stratified protocol used for LUAD.

Usage (on the DGX, after `source env.sh`):
    python run_ecg_domainC.py --smoke          # 1 fold, 1500 records, sanity
    python run_ecg_domainC.py                   # full run, targets sex,age65
    python run_ecg_domainC.py --targets sex age65 --epochs 25 --choquet
    python run_ecg_domainC.py --no-choquet      # skip FC if you only want B2/B3/PI

Outputs under $DATA_ROOT/ecg/derived/domainC/:
    results.json     per-target, per-condition mean+-SD AUROC over the 5 folds
    regimes.json     verdict and model-side regime signature per target
    choquet_interactions.csv   top +/- statement-pair interaction indices (FC)
"""

from __future__ import annotations

import argparse
import json
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

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

FOLD_GROUPS = {1: [1, 2], 2: [3, 4], 3: [5, 6], 4: [7, 8], 5: [9, 10]}
SEED = 42
CONCLUSIVE_THRESHOLD = 0.70


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "ecg"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def build_targets(keys: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """Construct leakage-free binary targets from PTB-XL metadata."""
    out = {}
    for name in names:
        if name == "sex":
            # PTB-XL: sex 0 = male, 1 = female. Predicting sex from ECG.
            out["sex"] = keys["sex"].astype(float)
        elif name == "age65":
            out["age65"] = (keys["age"] >= 65).astype(float)
        elif name == "age75":
            out["age75"] = (keys["age"] >= 75).astype(float)
        else:
            sys.exit(f"unknown target '{name}'. Supported: sex, age65, age75")
    return pd.DataFrame(out, index=keys.index)


def load_waveforms(ptbxl_dir: Path, filenames: pd.Series, limit: int | None) -> np.ndarray:
    """Preload the 100 Hz 12-lead records into a single float32 array [N, 12, 1000]."""
    n = len(filenames) if limit is None else min(limit, len(filenames))
    sig = np.zeros((n, 12, 1000), dtype=np.float32)
    t0 = time.time()
    for i in range(n):
        rec = str(ptbxl_dir / filenames.iloc[i])
        data, _ = wfdb.rdsamp(rec)          # [1000, 12]
        x = data.T.astype(np.float32)       # [12, 1000]
        if x.shape[1] != 1000:              # guard against odd lengths
            x = x[:, :1000] if x.shape[1] > 1000 else np.pad(x, ((0, 0), (0, 1000 - x.shape[1])))
        sig[i] = x
        if (i + 1) % 2000 == 0:
            print(f"  loaded {i + 1}/{n} waveforms ({time.time() - t0:.0f}s)")
    # per-lead z-normalisation across the loaded set (avoids leaking test stats
    # into training would require per-fold norm; kept global for a v1 baseline)
    mu = sig.mean(axis=(0, 2), keepdims=True)
    sd = sig.std(axis=(0, 2), keepdims=True) + 1e-6
    sig = (sig - mu) / sd
    return sig


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class ECGEncoder(nn.Module):
    """Compact 1D-CNN encoder of the raw 12-lead signal -> `dim`-d embedding."""

    def __init__(self, in_ch: int = 12, dim: int = 128):
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
        self.dim = dim

    def forward(self, x):                    # x: [B, 12, 1000]
        h = self.net(x)
        return self.pool(h).squeeze(-1)      # [B, dim]


class Choquet2Additive(nn.Module):
    """Differentiable 2-additive fuzzy Choquet integral over `n` statements.

    g(A) = sum_{k in A} phi_k + sum_{{j,k} subset A} I_jk. Returns the integral
    C_g(P) for each record; the learned interaction matrix I_jk is the L4
    read-out (positive => synergy, negative => redundancy).
    """

    def __init__(self, n: int, l1: float = 1e-3):
        super().__init__()
        self.n = n
        self.psi = nn.Parameter(torch.zeros(n))            # singletons via softmax
        self.iflat = nn.Parameter(torch.zeros(n * (n - 1) // 2))  # upper-tri pairs
        self.l1 = l1
        iu = torch.triu_indices(n, n, offset=1)
        self.register_buffer("iu", iu)

    def interaction_matrix(self) -> torch.Tensor:
        I = torch.zeros(self.n, self.n, device=self.iflat.device)
        I[self.iu[0], self.iu[1]] = self.iflat
        return I + I.t()

    def forward(self, P):                                  # P: [B, n] in [0,1]
        phi = torch.softmax(self.psi, dim=0)               # [n]
        I = self.interaction_matrix()                      # [n, n], zero diagonal
        Bsz = P.size(0)
        order = torch.argsort(P, dim=1)                    # ascending
        rank = torch.argsort(order, dim=1)                 # rank[b,i] in 0..n-1
        Psorted = torch.gather(P, 1, order)
        C = torch.zeros(Bsz, device=P.device)
        prev = torch.zeros(Bsz, device=P.device)
        # A_(k) = {elements whose sorted rank >= k}; masks avoid in-place ops
        for k in range(self.n):
            mask = (rank >= k).to(P.dtype)                 # [B, n]
            gk = (mask * phi.unsqueeze(0)).sum(1)
            gk = gk + 0.5 * torch.einsum("bi,bj,ij->b", mask, mask, I)
            C = C + (Psorted[:, k] - prev) * gk
            prev = Psorted[:, k]
        return C.unsqueeze(1)                              # [B, 1]

    def l1_penalty(self) -> torch.Tensor:
        return self.l1 * self.iflat.abs().sum()


class Head(nn.Module):
    """One of the four conditions -> a single logit."""

    def __init__(self, condition: str, edim: int, pdim: int):
        super().__init__()
        self.condition = condition
        self.enc = ECGEncoder(dim=edim) if condition in ("B2", "PI", "FC") else None
        self.choquet = Choquet2Additive(pdim) if condition == "FC" else None
        if condition == "B2":
            self.clf = nn.Linear(edim, 1)
        elif condition == "B3":
            self.clf = nn.Sequential(nn.Linear(pdim, 64), nn.ReLU(), nn.Linear(64, 1))
        elif condition == "PI":
            self.clf = nn.Linear(edim + pdim, 1)
        elif condition == "FC":
            self.clf = nn.Linear(edim + 1, 1)
        else:
            raise ValueError(condition)

    def forward(self, x, P):
        if self.condition == "B2":
            return self.clf(self.enc(x))
        if self.condition == "B3":
            return self.clf(P)
        if self.condition == "PI":
            return self.clf(torch.cat([self.enc(x), P], dim=1))
        # FC
        c = self.choquet(P)
        return self.clf(torch.cat([self.enc(x), c], dim=1))


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def train_eval(condition, Xtr, Ptr, ytr, Xte, Pte, yte, epochs, bs, lr, device):
    model = Head(condition, edim=128, pdim=Ptr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pw = torch.tensor([neg / max(pos, 1.0)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    tr = DataLoader(TensorDataset(Xtr, Ptr, ytr), batch_size=bs, shuffle=True, drop_last=False)
    for _ in range(epochs):
        model.train()
        for xb, pb, yb in tr:
            xb, pb, yb = xb.to(device), pb.to(device), yb.to(device)
            opt.zero_grad()
            logit = model(xb, pb).squeeze(1)
            loss = lossf(logit, yb)
            if condition == "FC":
                loss = loss + model.choquet.l1_penalty()
            loss.backward()
            opt.step()

    model.eval()
    scores = []
    with torch.no_grad():
        te = DataLoader(TensorDataset(Xte, Pte, yte), batch_size=512, shuffle=False)
        for xb, pb, _ in te:
            logit = model(xb.to(device), pb.to(device)).squeeze(1)
            scores.append(torch.sigmoid(logit).cpu().numpy())
    prob = np.concatenate(scores)
    auroc = roc_auc_score(yte.numpy(), prob)
    inter = None
    if condition == "FC":
        inter = model.choquet.interaction_matrix().detach().cpu().numpy()
    return float(auroc), inter


def regime_signature(cond_means: dict) -> str:
    """Assign the model-side regime signature from the condition means."""
    b2, b3 = cond_means.get("B2"), cond_means.get("B3")
    pi, fc = cond_means.get("PI"), cond_means.get("FC")
    best = max(v for v in cond_means.values())
    fused = max([v for v in (pi, fc) if v is not None] or [b2])
    if b3 is not None and b2 is not None and b3 > b2 + 0.01:
        return "pattern-side unique/floor (B3 > B2)"
    if fused > b2 + 0.015:
        return "synergy (fusion beats encoder)"
    if best - (b2 or 0) <= 0.02:
        return "encoder saturation (fusion ~ encoder)"
    return "mixed"


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--targets", nargs="+", default=["sex", "age65"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--choquet", dest="choquet", action="store_true", default=True)
    ap.add_argument("--no-choquet", dest="choquet", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="1 fold, 1500 records")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ecg_root = resolve_root(args.data_root)
    ptbxl_dir = ecg_root / "ptb-xl"
    memb_path = ecg_root / "derived" / "scp_membership.parquet"
    if not memb_path.exists():
        sys.exit(f"missing {memb_path}\nrun: python build_ptbxl_vocabulary.py")

    df = pd.read_parquet(memb_path)
    key_cols = ["patient_id", "strat_fold", "age", "sex", "filename_lr", "filename_hr"]
    stmt_cols = [c for c in df.columns if c not in key_cols]
    keys = df[key_cols].copy()

    if args.smoke:
        df = df.iloc[:1500].copy(); keys = keys.iloc[:1500].copy()
        folds_to_run = [5]
        args.epochs = min(args.epochs, 3)
    else:
        folds_to_run = list(FOLD_GROUPS)

    limit = 1500 if args.smoke else None
    print(f"loading {len(df)} waveforms from {ptbxl_dir} ...")
    X = load_waveforms(ptbxl_dir, keys["filename_lr"], limit)
    P = df[stmt_cols].to_numpy(dtype=np.float32)[: len(X)]
    keys = keys.iloc[: len(X)].copy()
    Y = build_targets(keys, args.targets)

    Xt = torch.from_numpy(X)
    Pt = torch.from_numpy(P)
    conditions = ["B2", "B3", "PI"] + (["FC"] if args.choquet else [])

    results, regimes, interactions = {}, {}, {}
    for target in args.targets:
        y = Y[target].to_numpy(dtype=np.float32)
        print(f"\n=== target: {target}  (prev {y.mean():.3f}, n_+={int(y.sum())}) ===")
        per_cond = {c: [] for c in conditions}
        fc_inter_acc = None
        for gf in folds_to_run:
            te_folds = FOLD_GROUPS[gf]
            te_mask = keys["strat_fold"].isin(te_folds).to_numpy()
            tr_mask = ~te_mask
            yt = torch.from_numpy(y)
            for c in conditions:
                auroc, inter = train_eval(
                    c, Xt[tr_mask], Pt[tr_mask], yt[tr_mask],
                    Xt[te_mask], Pt[te_mask], yt[te_mask],
                    args.epochs, args.batch_size, args.lr, device,
                )
                per_cond[c].append(auroc)
                print(f"  fold{gf} {c}: AUROC {auroc:.3f}")
                if c == "FC" and inter is not None:
                    fc_inter_acc = inter if fc_inter_acc is None else fc_inter_acc + inter

        cond_means = {c: float(np.mean(v)) for c, v in per_cond.items()}
        cond_sds = {c: float(np.std(v)) for c, v in per_cond.items()}
        results[target] = {"mean": cond_means, "sd": cond_sds, "folds": per_cond}
        best = max(cond_means.values())
        regimes[target] = {
            "best_auroc": best,
            "verdict": "Conclusive" if best >= CONCLUSIVE_THRESHOLD else "Inconclusive",
            "signature": regime_signature(cond_means),
        }
        print(f"  -> best {best:.3f} [{regimes[target]['verdict']}] "
              f"signature: {regimes[target]['signature']}")

        if fc_inter_acc is not None:
            n_folds = len(folds_to_run)
            I = fc_inter_acc / n_folds
            pairs = []
            for a in range(len(stmt_cols)):
                for b in range(a + 1, len(stmt_cols)):
                    pairs.append((stmt_cols[a], stmt_cols[b], float(I[a, b])))
            pairs.sort(key=lambda t: abs(t[2]), reverse=True)
            interactions[target] = pairs[:25]

    out_dir = ecg_root / "derived" / "domainC"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "regimes.json").write_text(json.dumps(regimes, indent=2))
    if interactions:
        rows = []
        for target, pairs in interactions.items():
            for j, k, v in pairs:
                rows.append({"target": target, "stmt_j": j, "stmt_k": k, "I_jk": v})
        pd.DataFrame(rows).to_csv(out_dir / "choquet_interactions.csv", index=False)

    print(f"\nwrote {out_dir}")
    print(json.dumps(regimes, indent=2))


if __name__ == "__main__":
    main()
