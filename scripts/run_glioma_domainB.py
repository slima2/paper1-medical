#!/usr/bin/env python
"""Domain-B run: multiparametric MRI -> glioma genotype, on TCGA/BraTS.

This is the glioma counterpart of run_ecg_domainC.py and the TCGA-LUAD
pipeline. It instantiates the SAME translatability framework in a third
modality, so almost all of the machinery (the 2-additive fuzzy Choquet
integral, the four-condition protocol, the Conclusive/Inconclusive verdict
and the model-side regime signature) is ported verbatim from the ECG run.
Only the two modality-specific channels change:

    source channel  P : ~30 VASARI semantic MR features, propagated from the
                        TCIA-annotated seed (178 LGG + 75 GBM) to the full
                        BraTS 2021 cohort by a feature classifier
                        (the ANORAK -> TCGA-LUAD move, in MRI).
    encoder channel E : a learned 2.5D CNN + ABMIL embedding of the
                        co-registered 4-sequence volume (T1, T1c, T2, FLAIR),
                        treating each subject as a bag of axial slices --
                        the direct analogue of the tile-bag + ABMIL used for
                        the whole-slide images in LUAD.
    target          Y : IDH mutation, 1p/19q codeletion, MGMT methylation.

Pre-registered regime predictions (tab:crossdomain_protocol, column B):
    IDH     -> regime (i)  encoder saturation  (AUROC ~ 0.85-0.90; E ~= fused)
    1p/19q  -> regime (ii) synergy             (FC/PI beat B2; +Choquet pair)
    MGMT    -> regime (iii) out-of-ontology    (B3 at chance, fusion no help;
                                                ~0.50-0.56 across architectures)

MGMT is the single claim on which the framework should be judged: the run
must be able to distinguish an EMPTY VASARI-side atom (out-of-ontology, the
prediction) from a DATA FLOOR (atom present but unestimable at small n_+),
exactly as RBM10-vs-KEAP1 was separated in LUAD. That distinction is read
from the model-side signatures (B3-vs-B2 and the Choquet interactions), not
from a single number -- hence the four-condition protocol below rather than
a frozen-predictor read-out.

Conditions (mirroring tab:mutation_luad and the ECG run):
    B2  encoder embedding only                 (ceiling baseline)
    B3  VASARI feature channel only
    PI  concat fusion  [E ; P]
    FC  E  +  2-additive fuzzy Choquet over the ~30 VASARI features

Splits: 5-fold patient-stratified, matching LUAD/ECG.

STUBS. Three inputs must be produced on the DGX before this runs; each is
marked "==== STUB (connect on DGX) ====" below and documented in
VASARI_RUN_PLAN.md:
    1. per-subject preselected axial slice tensors  (prepare_glioma_slices.py)
    2. VASARI features propagated to all BraTS subjects (propagate_vasari.py)
    3. genotype labels IDH / 1p19q / MGMT            (build_glioma_labels.py)
The model definitions, training loop, verdict and regime logic are complete
and are the faithful port of run_ecg_domainC.py.

Usage (on the DGX, after `source env.sh`, once the three inputs exist):
    python run_glioma_domainB.py --smoke                 # 1 fold, small n, sanity
    python run_glioma_domainB.py                         # full run, all targets
    python run_glioma_domainB.py --targets idh mgmt --epochs 30 --choquet
    python run_glioma_domainB.py --no-choquet            # B2/B3/PI only

Outputs under $DATA_ROOT/glioma/derived/domainB/:
    results.json     per-target, per-condition mean+-SD AUROC over the 5 folds
    regimes.json     verdict and model-side regime signature per target
    choquet_interactions.csv   top +/- VASARI-feature-pair interaction indices
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
    from torch.utils.data import DataLoader, Dataset
except Exception as exc:  # pragma: no cover
    sys.exit(f"PyTorch is required: {exc}")

from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

SEED = 42
CONCLUSIVE_THRESHOLD = 0.70
N_FOLDS = 5
N_SEQUENCES = 4          # T1, T1c, T2, FLAIR (BraTS co-registered channels)
SLICES_PER_SUBJECT = 24  # K axial slices selected around the tumour centroid
SLICE_HW = 128           # in-plane crop fed to the slice encoder
TARGETS_DEFAULT = ["idh", "onep19q", "mgmt"]
MIN_POS = 10             # min positives (or negatives) to attempt a target;
                         # below this a target is reported underpowered, not run


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "glioma"


# --------------------------------------------------------------------------
# Data
# ==== STUB (connect on DGX) ====
# These three loaders are the only glioma-specific I/O. Each expects an
# artefact produced by a preprocessing script (see VASARI_RUN_PLAN.md). The
# contracts are fixed here so the rest of the pipeline is fully defined.
# --------------------------------------------------------------------------

def load_manifest(glioma_root: Path) -> pd.DataFrame:
    """Subject manifest joining imaging, VASARI and genotype availability.

    Expected file: $DATA_ROOT/glioma/derived/manifest.parquet
    Required columns:
        subject_id   canonical TCIA/TCGA id (BraTS ids resolved via crosswalk)
        strat_fold   int in 1..N_FOLDS, patient-stratified (built once, fixed)
        slices_path  path to the per-subject slice tensor .npy (see below)
    Produced by prepare_glioma_slices.py, which also writes the .npy files.
    """
    path = glioma_root / "derived" / "manifest.parquet"
    if not path.exists():
        sys.exit(
            f"missing {path}\n"
            "run prepare_glioma_slices.py first (see VASARI_RUN_PLAN.md step 3)."
        )
    df = pd.read_parquet(path)
    need = {"subject_id", "strat_fold", "slices_path"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"{path} is missing columns: {sorted(missing)}")
    return df.set_index("subject_id")


def load_vasari_channel(glioma_root: Path, subjects: pd.Index) -> tuple[np.ndarray, list[str]]:
    """The VASARI source channel P in [0, 1], aligned to `subjects`.

    Two sources are supported, in priority order:
      1. vasari_propagated.parquet -- the full-cohort table written by
         propagate_vasari.py (BraTS-centric run). Values already in [0, 1].
      2. vasari_features.parquet   -- the *real* expert scores written by
         fetch_vasari.py (TCGA-LGG-centric v1). These are ordinal integer
         scores, so each feature is min-max rescaled to [0, 1] here. The
         rescale is a fixed monotone map of an expert ordinal (not learned
         from labels), so it introduces no target leakage.
    Non-numeric bookkeeping columns (e.g. source_file) are dropped.
    """
    prop = glioma_root / "vasari" / "vasari_propagated.parquet"
    real = glioma_root / "vasari" / "vasari_features.parquet"
    if prop.exists():
        df = pd.read_parquet(prop)
        source = "propagated"
    elif real.exists():
        df = pd.read_parquet(real)
        source = "real"
    else:
        sys.exit(
            f"missing both {prop} and {real}\n"
            "run: python fetch_vasari.py   (real seed scores; LGG-centric v1), or\n"
            "     python propagate_vasari.py  (full-cohort propagated; BraTS run).\n"
            "See VASARI_RUN_PLAN.md step 2."
        )

    df = df.drop(columns=[c for c in ("source_file",) if c in df.columns],
                 errors="ignore")
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.reindex(subjects)

    if source == "real":
        lo = df.min(axis=0)
        span = (df.max(axis=0) - lo).replace(0.0, 1.0)
        df = (df - lo) / span

    if df.isna().any().any():
        n_bad = int(df.isna().any(axis=1).sum())
        print(f"[warn] {n_bad} subjects missing VASARI features; filling 0.5 (neutral)")
        df = df.fillna(0.5)
    feature_cols = list(df.columns)
    P = np.clip(df.to_numpy(dtype=np.float32), 0.0, 1.0)
    print(f"[ok] VASARI channel: {source}, {P.shape[0]} subjects x {P.shape[1]} features")
    return P, feature_cols


def load_labels(glioma_root: Path, subjects: pd.Index, names: list[str]) -> pd.DataFrame:
    """Binary genotype targets aligned to `subjects`.

    Expected file: $DATA_ROOT/glioma/derived/glioma_labels.parquet
        index = subject_id, columns include:
            idh       1 = IDH-mutant,        0 = IDH-wildtype
            onep19q   1 = 1p/19q codeleted,  0 = intact
            mgmt      1 = MGMT methylated,   0 = unmethylated
    Assembled by build_glioma_labels.py from GDC/TCGA (IDH, 1p19q) and the
    BraTS 2021 MGMT table. Missing calls are left as NaN and dropped per target.
    """
    path = glioma_root / "derived" / "glioma_labels.parquet"
    if not path.exists():
        sys.exit(
            f"missing {path}\n"
            "run: python build_glioma_labels.py  (GDC/TCGA + BraTS MGMT).\n"
            "See VASARI_RUN_PLAN.md step 4."
        )
    df = pd.read_parquet(path).reindex(subjects)
    for n in names:
        if n not in df.columns:
            sys.exit(f"label '{n}' not in {path}; available: {list(df.columns)}")
    return df[names]


class GliomaDataset(Dataset):
    """Lazy per-subject reader: returns (slices[K, C, H, W], P[pdim], y).

    Slice tensors are read from disk on demand (MRI volumes are far too large
    to hold a whole cohort in RAM, unlike the ECG waveform array). Each .npy is
    written by prepare_glioma_slices.py as float32 [K, C, H, W], already
    intensity-normalised (z-score per sequence within brain mask) and centred
    on the tumour so no augmentation is needed for a v1 baseline.
    """

    def __init__(self, slice_paths: list[str], P: np.ndarray, y: np.ndarray):
        self.slice_paths = slice_paths
        self.P = torch.from_numpy(P)
        self.y = torch.from_numpy(y)

    def __len__(self) -> int:
        return len(self.slice_paths)

    def __getitem__(self, i: int):
        arr = np.load(self.slice_paths[i]).astype(np.float32)   # [K, C, H, W]
        if arr.shape != (SLICES_PER_SUBJECT, N_SEQUENCES, SLICE_HW, SLICE_HW):
            arr = _fit_slice_tensor(arr)
        return torch.from_numpy(arr), self.P[i], self.y[i]


def _fit_slice_tensor(arr: np.ndarray) -> np.ndarray:
    """Guard: coerce an odd-shaped slice tensor to [K, C, H, W]."""
    k, c, h, w = arr.shape
    if k >= SLICES_PER_SUBJECT:
        idx = np.linspace(0, k - 1, SLICES_PER_SUBJECT).round().astype(int)
        arr = arr[idx]
    else:
        pad = np.repeat(arr[-1:], SLICES_PER_SUBJECT - k, axis=0)
        arr = np.concatenate([arr, pad], axis=0)
    if (h, w) != (SLICE_HW, SLICE_HW):  # centre-crop / pad in-plane
        arr = _centre_fit(arr, SLICE_HW)
    return arr.astype(np.float32)


def _centre_fit(arr: np.ndarray, size: int) -> np.ndarray:
    k, c, h, w = arr.shape
    out = np.zeros((k, c, size, size), dtype=arr.dtype)
    hh, ww = min(h, size), min(w, size)
    sy, sx = (h - hh) // 2, (w - ww) // 2
    dy, dx = (size - hh) // 2, (size - ww) // 2
    out[:, :, dy:dy + hh, dx:dx + ww] = arr[:, :, sy:sy + hh, sx:sx + ww]
    return out


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

class SliceEncoder(nn.Module):
    """Compact 2D CNN encoding one 4-sequence axial slice -> `dim` embedding."""

    def __init__(self, in_ch: int = N_SEQUENCES, dim: int = 128):
        super().__init__()

        def block(ci, co, k=3, s=2):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, stride=s, padding=k // 2),
                nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            )

        self.net = nn.Sequential(
            block(in_ch, 32), block(32, 64), block(64, 128),
            block(128, 128), block(128, dim),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dim = dim

    def forward(self, x):                       # x: [B*K, C, H, W]
        h = self.net(x)
        return self.pool(h).flatten(1)          # [B*K, dim]


class ABMIL(nn.Module):
    """Gated attention pooling over the K slices of a subject (Ilse 2018).

    The 2.5D analogue of the tile-level ABMIL used on the whole-slide images:
    each subject is a bag of axial slices and the attention weights are the L1
    'where' read-out (which slices carry the genotype signal).
    """

    def __init__(self, dim: int, att: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, att)
        self.U = nn.Linear(dim, att)
        self.w = nn.Linear(att, 1)

    def forward(self, h):                        # h: [B, K, dim]
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h)))  # [B, K, 1]
        a = torch.softmax(a, dim=1)
        z = (a * h).sum(dim=1)                   # [B, dim]
        return z, a.squeeze(-1)


class MRIEncoder(nn.Module):
    """2.5D encoder: per-slice CNN + ABMIL pooling -> subject embedding."""

    def __init__(self, in_ch: int = N_SEQUENCES, dim: int = 128):
        super().__init__()
        self.slice_enc = SliceEncoder(in_ch, dim)
        self.mil = ABMIL(dim)
        self.dim = dim

    def forward(self, x):                        # x: [B, K, C, H, W]
        B, K, C, H, W = x.shape
        h = self.slice_enc(x.reshape(B * K, C, H, W))   # [B*K, dim]
        h = h.reshape(B, K, self.dim)
        z, _att = self.mil(h)                    # [B, dim]
        return z


class Choquet2Additive(nn.Module):
    """Differentiable 2-additive fuzzy Choquet integral over `n` features.

    g(A) = sum_{k in A} phi_k + sum_{{j,k} subset A} I_jk. Returns the integral
    C_g(P) for each subject; the learned interaction matrix I_jk is the L4
    read-out (positive => synergy, negative => redundancy). Ported verbatim
    from run_ecg_domainC.py so the operator is identical across domains.
    """

    def __init__(self, n: int, l1: float = 1e-3):
        super().__init__()
        self.n = n
        self.psi = nn.Parameter(torch.zeros(n))                    # singletons via softmax
        self.iflat = nn.Parameter(torch.zeros(n * (n - 1) // 2))   # upper-tri pairs
        self.l1 = l1
        iu = torch.triu_indices(n, n, offset=1)
        self.register_buffer("iu", iu)

    def interaction_matrix(self) -> torch.Tensor:
        I = torch.zeros(self.n, self.n, device=self.iflat.device)
        I[self.iu[0], self.iu[1]] = self.iflat
        return I + I.t()

    def forward(self, P):                                          # P: [B, n] in [0,1]
        phi = torch.softmax(self.psi, dim=0)                       # [n]
        I = self.interaction_matrix()                              # [n, n], zero diagonal
        Bsz = P.size(0)
        order = torch.argsort(P, dim=1)                            # ascending
        rank = torch.argsort(order, dim=1)                         # rank[b,i] in 0..n-1
        Psorted = torch.gather(P, 1, order)
        C = torch.zeros(Bsz, device=P.device)
        prev = torch.zeros(Bsz, device=P.device)
        for k in range(self.n):
            mask = (rank >= k).to(P.dtype)                         # [B, n]
            gk = (mask * phi.unsqueeze(0)).sum(1)
            gk = gk + 0.5 * torch.einsum("bi,bj,ij->b", mask, mask, I)
            C = C + (Psorted[:, k] - prev) * gk
            prev = Psorted[:, k]
        return C.unsqueeze(1)                                      # [B, 1]

    def l1_penalty(self) -> torch.Tensor:
        return self.l1 * self.iflat.abs().sum()


class Head(nn.Module):
    """One of the four conditions -> a single logit."""

    def __init__(self, condition: str, edim: int, pdim: int):
        super().__init__()
        self.condition = condition
        self.enc = MRIEncoder(dim=edim) if condition in ("B2", "PI", "FC") else None
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

def train_eval(condition, tr_ds, te_ds, edim, pdim, epochs, bs, lr, device):
    model = Head(condition, edim=edim, pdim=pdim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    ytr = tr_ds.y.numpy()
    pos = float(ytr.sum()); neg = float(len(ytr) - pos)
    pw = torch.tensor([neg / max(pos, 1.0)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)

    tr = DataLoader(tr_ds, batch_size=bs, shuffle=True, drop_last=False, num_workers=2)
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
    scores, ys = [], []
    with torch.no_grad():
        te = DataLoader(te_ds, batch_size=bs, shuffle=False, num_workers=2)
        for xb, pb, yb in te:
            logit = model(xb.to(device), pb.to(device)).squeeze(1)
            scores.append(torch.sigmoid(logit).cpu().numpy())
            ys.append(yb.numpy())
    prob = np.concatenate(scores)
    ytrue = np.concatenate(ys)
    try:
        auroc = roc_auc_score(ytrue, prob)   # NaN if a fold is single-class
    except ValueError:
        auroc = float("nan")
    inter = None
    if condition == "FC":
        inter = model.choquet.interaction_matrix().detach().cpu().numpy()
    return float(auroc), inter


def regime_signature(cond_means: dict) -> str:
    """Assign the model-side regime signature from the condition means.

    Ported verbatim from run_ecg_domainC.py. Expected glioma outcomes:
        IDH    -> 'encoder saturation'         (regime i)
        1p/19q -> 'synergy (fusion beats ...)'  (regime ii)
        MGMT   -> 'pattern-side unique/floor' if B3>B2 (a DATA FLOOR), else
                  'encoder saturation' with all conditions ~chance, which is
                  the OUT-OF-ONTOLOGY prediction (empty VASARI-side atom).
    """
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
    ap.add_argument("--targets", nargs="+", default=TARGETS_DEFAULT,
                    help="subset of: idh onep19q mgmt")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--edim", type=int, default=128)
    ap.add_argument("--choquet", dest="choquet", action="store_true", default=True)
    ap.add_argument("--no-choquet", dest="choquet", action="store_false")
    ap.add_argument("--smoke", action="store_true", help="1 fold, small n, 3 epochs")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    glioma_root = resolve_root(args.data_root)

    manifest = load_manifest(glioma_root)
    subjects = manifest.index
    P_all, feature_cols = load_vasari_channel(glioma_root, subjects)
    Y_all = load_labels(glioma_root, subjects, args.targets)
    folds = manifest["strat_fold"].to_numpy()
    slice_paths = manifest["slices_path"].astype(str).to_numpy()
    pdim = P_all.shape[1]
    print(f"{len(subjects)} subjects x {pdim} VASARI features; targets {args.targets}")

    if args.smoke:
        folds_to_run = [1]
        args.epochs = min(args.epochs, 3)
    else:
        folds_to_run = list(range(1, N_FOLDS + 1))

    conditions = ["B2", "B3", "PI"] + (["FC"] if args.choquet else [])

    results, regimes, interactions = {}, {}, {}
    for target in args.targets:
        y_full = Y_all[target].to_numpy(dtype=np.float32)
        keep = ~np.isnan(y_full)                     # drop subjects lacking this call
        n_kept = int(keep.sum())
        n_pos = int(np.nansum(y_full[keep])) if n_kept else 0
        n_neg = n_kept - n_pos
        print(f"\n=== target: {target}  (n={n_kept}, "
              f"prev {np.nanmean(y_full[keep]) if n_kept else float('nan'):.3f}, "
              f"n_+={n_pos}) ===")
        if n_pos < MIN_POS or n_neg < MIN_POS:
            reason = ("no labels" if n_kept == 0
                      else f"underpowered (n_+={n_pos}, n_-={n_neg}, floor={MIN_POS})")
            print(f"  -> SKIP: {reason}")
            regimes[target] = {
                "best_auroc": None, "verdict": "Inconclusive",
                "signature": "underpowered / unavailable",
                "n": n_kept, "n_pos": n_pos, "n_neg": n_neg,
            }
            results[target] = {"skipped": reason, "n": n_kept,
                               "n_pos": n_pos, "n_neg": n_neg}
            continue
        per_cond = {c: [] for c in conditions}
        fc_inter_acc = None
        for gf in folds_to_run:
            te_mask = (folds == gf) & keep
            tr_mask = (folds != gf) & keep
            for c in conditions:
                tr_ds = GliomaDataset(list(slice_paths[tr_mask]), P_all[tr_mask], y_full[tr_mask])
                te_ds = GliomaDataset(list(slice_paths[te_mask]), P_all[te_mask], y_full[te_mask])
                auroc, inter = train_eval(
                    c, tr_ds, te_ds, args.edim, pdim,
                    args.epochs, args.batch_size, args.lr, device,
                )
                per_cond[c].append(auroc)
                print(f"  fold{gf} {c}: AUROC {auroc:.3f}")
                if c == "FC" and inter is not None:
                    fc_inter_acc = inter if fc_inter_acc is None else fc_inter_acc + inter

        cond_means = {c: float(np.nanmean(v)) for c, v in per_cond.items()}
        cond_sds = {c: float(np.nanstd(v)) for c, v in per_cond.items()}
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
            for a in range(len(feature_cols)):
                for b in range(a + 1, len(feature_cols)):
                    pairs.append((feature_cols[a], feature_cols[b], float(I[a, b])))
            pairs.sort(key=lambda t: abs(t[2]), reverse=True)
            interactions[target] = pairs[:25]

    out_dir = glioma_root / "derived" / "domainB"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "regimes.json").write_text(json.dumps(regimes, indent=2))
    if interactions:
        rows = []
        for target, pairs in interactions.items():
            for j, k, v in pairs:
                rows.append({"target": target, "feat_j": j, "feat_k": k, "I_jk": v})
        pd.DataFrame(rows).to_csv(out_dir / "choquet_interactions.csv", index=False)

    print(f"\nwrote {out_dir}")
    print(json.dumps(regimes, indent=2))


if __name__ == "__main__":
    main()
