#!/usr/bin/env python
"""Re-read the three regimes on CPTAC-LUAD by TRAINING the four conditions there.

WHAT QUESTION THIS ANSWERS
--------------------------
eval_cptac_transfer.py answered: do the frozen TCGA weights survive the move to
CPTAC? (No.) That is a statement about a predictor. It leaves the more
interesting question untouched: if we retrain the four-condition protocol
*inside* CPTAC, does the same per-gene regime structure appear? A regime is read
from the sign of contrasts between B2/B3/PI/FC, so it can only be re-read where
all four conditions have been fitted. Until this script runs, the paper cannot
say whether a regime assignment is a property of the gene or of the cohort, and
it says so explicitly.

Conditions, identical to the LUAD, glioma and ECG runs:
    B2  encoder embedding only (frozen CTransPath + ABMIL)   ceiling baseline
    B3  ANORAK 6-pattern channel only
    PI  concat fusion [E ; P]                                additive
    FC  E + 2-additive fuzzy Choquet over the 6 patterns     non-additive

THE POWER PROBLEM, STATED UP FRONT
----------------------------------
CPTAC-LUAD has 106 genotyped patients with matched H&E. Positive counts are
TP53 55, EGFR 36, KRAS 32, STK11 20, KEAP1 12, RBM10 5, against the n_+ = 37
estimation floor established on TCGA. Only TP53 clears it; EGFR is at the edge.
This run therefore CANNOT adjudicate a regime for four of six genes, and the
script is built to say so rather than to emit a confident label from noise:

  * Per-fold AUROC is not the primary statistic. With ~21 test patients per
    fold, a gene with n_+ = 12 puts ~2 positives in a fold, where AUROC is
    close to meaningless. We pool the out-of-fold predictions and treat the
    pooled OOF AUROC as the point estimate (per-fold values are still reported).
  * The headline statistic is a PAIRED BOOTSTRAP OVER PATIENTS on the pooled
    OOF predictions: resample patients, recompute all four AUROCs, recompute
    the contrasts, and report how often the CPTAC sign agrees with the TCGA
    sign and how often the whole regime label reproduces. A regime that
    reproduces in 52% of resamples is a coin flip and is reported as such.
  * Any gene below the floor is flagged `underpowered` in regimes.json no
    matter how clean its point estimate looks.

WHAT EACH OUTCOME WOULD MEAN
----------------------------
  TP53 (the one powered gene) re-reads as regime (i)  -> the assignment is
      reproducible across cohorts for the case we can actually test; the
      Discussion's "property of the triple, not of the gene" hedge can be
      narrowed.
  TP53 re-reads as something else                     -> the assignment is
      cohort-relative, which is a real finding and the current hedge stands.
  Everything else                                     -> unresolvable at this
      n_+, which is itself a validation of the estimation-floor machinery.

A secondary output matters independently of the regime question: the FC
condition yields the Choquet interaction matrix on a SECOND cohort. The paper
claims the sign of an interaction is more stable than the AUROC gap it
produces, and currently evidences that with a single case-study slide. The
per-fold I_jk distribution written here is the direct test of that claim.

Usage (DGX, after `source env.sh` and extract_cptac_features.py):
    python run_cptac_domainA_v2.py                       # all genes, 5 folds
    python run_cptac_domainA_v2.py --smoke               # 1 fold, 5 epochs
    python run_cptac_domainA_v2.py --genes TP53 EGFR     # the powered ones
    python run_cptac_domainA_v2.py --features "$DATA_ROOT/luad_cptac/features_stain_reinhard"

Outputs under $DATA_ROOT/luad_cptac/derived/domainA_v2/:
    results.json              per-gene, per-condition pooled + per-fold AUROC
    regimes.json              CPTAC regime read, TCGA reference, agreement
    contrast_bootstrap.csv    contrast CIs and sign-agreement probabilities
    choquet_interactions.csv  per-fold I_jk, including lepidic x solid
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.model_selection import StratifiedKFold

SEED = 42
CONCLUSIVE_THRESHOLD = 0.70
ESTIMATION_FLOOR = 37       # n_+ floor carried over from the TCGA-LUAD run
MIN_POS_ATTEMPT = 8         # below this we do not even fit; nothing is learnable
N_FOLDS = 5
EMB_DIM = 512               # CTransPath
GENES = ["TP53", "EGFR", "KRAS", "STK11", "KEAP1", "RBM10"]
PATTERNS = ["lepidic", "acinar", "papillary", "micropapillary", "solid", "cribriform"]

# The TCGA-LUAD reference, transcribed from tab:mutation_luad_resultssec. Used
# only to define the sign each contrast is compared against; override with
# --tcga-reference if the table changes. B3 is recorded where the paper states
# it; None means the contrast against B3 is not scored.
TCGA_REFERENCE = {
    "TP53":  {"regime": "(i)",    "B2": 0.718, "B3": None,  "PI": 0.717, "FC": 0.716},
    "EGFR":  {"regime": "(i)",    "B2": 0.701, "B3": None,  "PI": 0.694, "FC": 0.684},
    "KRAS":  {"regime": "(ii-b)", "B2": 0.607, "B3": None,  "PI": 0.590, "FC": 0.609},
    "STK11": {"regime": "(iii)",  "B2": 0.684, "B3": None,  "PI": 0.695, "FC": 0.658},
    "KEAP1": {"regime": "(iii)",  "B2": 0.597, "B3": 0.588, "PI": 0.610, "FC": 0.589},
    "RBM10": {"regime": "(iii)",  "B2": 0.642, "B3": 0.653, "PI": 0.640, "FC": 0.661},
}


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "luad_cptac"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def load_patient_bags(features_dir: Path, max_tiles: int, rng: np.random.Generator):
    """Collapse per-slide .npz features into one bag per patient.

    A CPTAC patient can carry several slides (498 slides / 106 patients). The
    frozen-transfer run mean-pooled slide probabilities to the patient; the
    bag-level analogue is to pool the patient's tiles into a single bag, which
    is what the MIL head consumes. Bags are capped at --max-tiles by uniform
    subsample (seeded) so a patient with eight slides does not dominate.
    """
    manifest_path = features_dir / "manifest.parquet"
    if not manifest_path.exists():
        sys.exit(f"missing {manifest_path}\nrun extract_cptac_features.py first.")
    man = pd.read_parquet(manifest_path)

    bags: dict[str, dict] = {}
    for pid, grp in man.groupby("patient_id"):
        embs, pats = [], []
        for npz in grp["npz"]:
            p = Path(npz)
            if not p.exists():
                print(f"[warn] missing {p}; skipping slide")
                continue
            with np.load(p) as z:
                embs.append(z["emb"].astype(np.float32))
                pats.append(z["pat"].astype(np.float32))
        if not embs:
            continue
        E = np.concatenate(embs, axis=0)
        P = np.concatenate(pats, axis=0)
        if E.shape[0] > max_tiles:
            idx = rng.choice(E.shape[0], size=max_tiles, replace=False)
            idx.sort()
            E, P = E[idx], P[idx]
        bags[str(pid)] = {"E": E, "P": P, "n_slides": int(len(grp))}
    if not bags:
        sys.exit(f"no usable feature files under {features_dir}")
    print(f"[ok] {len(bags)} patient bags from {len(man)} slides")
    return bags


def load_labels(root: Path, patients: list[str], genes: list[str]) -> pd.DataFrame:
    path = root / "labels.parquet"
    if not path.exists():
        sys.exit(f"missing {path}\nrun: python fetch_cptac_labels.py")
    df = pd.read_parquet(path)
    missing = [g for g in genes if g not in df.columns]
    if missing:
        sys.exit(f"labels missing genes {missing}; have {list(df.columns)}")
    out = df.reindex(patients)[genes]
    n_lab = int(out.notna().all(axis=1).sum())
    print(f"[ok] labels for {n_lab}/{len(patients)} patients with features")
    return out


class BagDataset(Dataset):
    """Returns (tiles [T, 512], pattern vector [6], y). T varies per patient."""

    def __init__(self, bags: list[dict], y: np.ndarray):
        self.bags = bags
        self.y = y.astype(np.float32)

    def __len__(self) -> int:
        return len(self.bags)

    def __getitem__(self, i: int):
        b = self.bags[i]
        E = torch.from_numpy(b["E"])
        # The source channel at bag level is the mean growth-pattern
        # composition over the retained tiles -- the c_s of the levels table.
        P = torch.from_numpy(b["P"].mean(axis=0))
        return E, P, torch.tensor(self.y[i])


def collate(batch):
    """Pad variable-length tile bags and carry a mask so attention ignores pad."""
    Es, Ps, ys = zip(*batch)
    T = max(e.shape[0] for e in Es)
    B = len(Es)
    out = torch.zeros(B, T, EMB_DIM)
    mask = torch.zeros(B, T, dtype=torch.bool)
    for i, e in enumerate(Es):
        out[i, : e.shape[0]] = e
        mask[i, : e.shape[0]] = True
    return out, mask, torch.stack(Ps), torch.stack(ys)


# --------------------------------------------------------------------------
# Models. Ported from run_glioma_domainB.py; only the encoder differs, because
# CTransPath is frozen here (features are precomputed) so the trainable encoder
# is a projection + ABMIL rather than a CNN. That keeps the parameter count low
# enough to fit 106 patients.
# --------------------------------------------------------------------------

class ABMIL(nn.Module):
    """Gated attention pooling over tiles (Ilse 2018), mask-aware."""

    def __init__(self, dim: int, att: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, att)
        self.U = nn.Linear(dim, att)
        self.w = nn.Linear(att, 1)

    def forward(self, h, mask):                         # h [B,T,dim] mask [B,T]
        a = self.w(torch.tanh(self.V(h)) * torch.sigmoid(self.U(h))).squeeze(-1)
        a = a.masked_fill(~mask, float("-inf"))
        a = torch.softmax(a, dim=1)
        z = torch.einsum("bt,btd->bd", a, h)
        return z, a


class TileEncoder(nn.Module):
    """Frozen CTransPath features -> projection -> ABMIL -> bag embedding."""

    def __init__(self, dim: int = 128, dropout: float = 0.25):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(EMB_DIM, dim), nn.ReLU(inplace=True), nn.Dropout(dropout))
        self.mil = ABMIL(dim)
        self.dim = dim

    def forward(self, x, mask):
        z, _a = self.mil(self.proj(x), mask)
        return z


class Choquet2Additive(nn.Module):
    """Differentiable 2-additive fuzzy Choquet integral. Verbatim port from
    run_glioma_domainB.py / run_ecg_domainC.py so the operator is identical
    across all four domains and the I_jk are directly comparable."""

    def __init__(self, n: int, l1: float = 1e-3):
        super().__init__()
        self.n = n
        self.psi = nn.Parameter(torch.zeros(n))
        self.iflat = nn.Parameter(torch.zeros(n * (n - 1) // 2))
        self.l1 = l1
        self.register_buffer("iu", torch.triu_indices(n, n, offset=1))

    def interaction_matrix(self) -> torch.Tensor:
        I = torch.zeros(self.n, self.n, device=self.iflat.device)
        I[self.iu[0], self.iu[1]] = self.iflat
        return I + I.t()

    def forward(self, P):                                # P [B, n] in [0,1]
        phi = torch.softmax(self.psi, dim=0)
        I = self.interaction_matrix()
        Bsz = P.size(0)
        order = torch.argsort(P, dim=1)
        rank = torch.argsort(order, dim=1)
        Psorted = torch.gather(P, 1, order)
        C = torch.zeros(Bsz, device=P.device)
        prev = torch.zeros(Bsz, device=P.device)
        for k in range(self.n):
            mask = (rank >= k).to(P.dtype)
            gk = (mask * phi.unsqueeze(0)).sum(1)
            gk = gk + 0.5 * torch.einsum("bi,bj,ij->b", mask, mask, I)
            C = C + (Psorted[:, k] - prev) * gk
            prev = Psorted[:, k]
        return C.unsqueeze(1)

    def l1_penalty(self) -> torch.Tensor:
        return self.l1 * self.iflat.abs().sum()


class Head(nn.Module):
    def __init__(self, condition: str, edim: int, pdim: int):
        super().__init__()
        self.condition = condition
        self.enc = TileEncoder(edim) if condition in ("B2", "PI", "FC") else None
        self.choquet = Choquet2Additive(pdim) if condition == "FC" else None
        if condition == "B2":
            self.clf = nn.Linear(edim, 1)
        elif condition == "B3":
            self.clf = nn.Sequential(nn.Linear(pdim, 32), nn.ReLU(), nn.Linear(32, 1))
        elif condition == "PI":
            self.clf = nn.Linear(edim + pdim, 1)
        elif condition == "FC":
            self.clf = nn.Linear(edim + 1, 1)
        else:
            raise ValueError(condition)

    def forward(self, x, mask, P):
        if self.condition == "B2":
            return self.clf(self.enc(x, mask))
        if self.condition == "B3":
            return self.clf(P)
        if self.condition == "PI":
            return self.clf(torch.cat([self.enc(x, mask), P], dim=1))
        return self.clf(torch.cat([self.enc(x, mask), self.choquet(P)], dim=1))


# --------------------------------------------------------------------------
# Train / eval
# --------------------------------------------------------------------------

def train_eval(condition, tr_ds, te_ds, edim, pdim, epochs, bs, lr, device):
    """Fit one condition on one fold; return (test probabilities, I_jk|None)."""
    model = Head(condition, edim=edim, pdim=pdim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    pos = float(tr_ds.y.sum()); neg = float(len(tr_ds.y) - pos)
    lossf = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))

    tr = DataLoader(tr_ds, batch_size=bs, shuffle=True, collate_fn=collate)
    for _ in range(epochs):
        model.train()
        for xb, mb, pb, yb in tr:
            xb, mb, pb, yb = xb.to(device), mb.to(device), pb.to(device), yb.to(device)
            opt.zero_grad()
            loss = lossf(model(xb, mb, pb).squeeze(1), yb)
            if condition == "FC":
                loss = loss + model.choquet.l1_penalty()
            loss.backward()
            opt.step()

    model.eval()
    probs = []
    with torch.no_grad():
        te = DataLoader(te_ds, batch_size=bs, shuffle=False, collate_fn=collate)
        for xb, mb, pb, _yb in te:
            logit = model(xb.to(device), mb.to(device), pb.to(device)).squeeze(1)
            probs.append(torch.sigmoid(logit).cpu().numpy())
    inter = (model.choquet.interaction_matrix().detach().cpu().numpy()
             if condition == "FC" else None)
    return np.concatenate(probs), inter


def safe_auroc(y, p) -> float:
    try:
        return float(roc_auc_score(y, p))
    except ValueError:
        return float("nan")


def regime_signature(m: dict) -> str:
    """Model-side regime read. Identical thresholds to run_glioma_domainB.py so
    a CPTAC read and a TCGA read are produced by the same rule."""
    b2, b3, pi, fc = m.get("B2"), m.get("B3"), m.get("PI"), m.get("FC")
    if None in (b2, b3, pi, fc) or any(np.isnan(v) for v in (b2, b3, pi, fc)):
        return "incomplete"
    if b3 > b2 + 0.01:
        return "pattern-side unique/floor (B3 > B2)"
    if fc > pi + 0.01 and fc > b2 + 0.015:
        return "synergy (non-additive Choquet beats concat)"
    if pi > b2 + 0.015:
        return "additive fusion (concat helps, Choquet does not)"
    if max(b2, b3, pi, fc) - b2 <= 0.02:
        return "encoder saturation (fusion ~ encoder)"
    return "mixed"


SIGNATURE_TO_REGIME = {
    "encoder saturation (fusion ~ encoder)": "(i)",
    "additive fusion (concat helps, Choquet does not)": "(ii-a)",
    "synergy (non-additive Choquet beats concat)": "(ii-b)",
    "pattern-side unique/floor (B3 > B2)": "(iii)",
}


def bootstrap_contrasts(oof: dict, y: np.ndarray, ref: dict, n_boot: int,
                        rng: np.random.Generator) -> dict:
    """Paired bootstrap over patients on the pooled out-of-fold predictions.

    Resampling patients (not retraining) is what makes 1000 replicates
    affordable. It quantifies sampling noise in the contrasts, which is the
    quantity the regime read actually depends on; it does not capture
    training-time variance, and we say so in the run plan.
    """
    n = len(y)
    contrasts = {"PI-B2": ("PI", "B2"), "FC-PI": ("FC", "PI"),
                 "B3-B2": ("B3", "B2"), "FC-B2": ("FC", "B2")}
    draws = {k: [] for k in contrasts}
    regimes: list[str] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yb = y[idx]
        if yb.sum() < 2 or (len(yb) - yb.sum()) < 2:
            continue
        aur = {c: safe_auroc(yb, oof[c][idx]) for c in oof}
        if any(np.isnan(v) for v in aur.values()):
            continue
        for name, (a, b) in contrasts.items():
            if a in aur and b in aur:
                draws[name].append(aur[a] - aur[b])
        regimes.append(SIGNATURE_TO_REGIME.get(regime_signature(aur), "other"))

    out = {}
    for name, vals in draws.items():
        if not vals:
            continue
        v = np.asarray(vals)
        a, b = contrasts[name]
        ref_a, ref_b = ref.get(a), ref.get(b)
        entry = {
            "mean": float(v.mean()),
            "ci_lo": float(np.percentile(v, 2.5)),
            "ci_hi": float(np.percentile(v, 97.5)),
            "p_positive": float((v > 0).mean()),
            "n_draws": int(v.size),
        }
        if ref_a is not None and ref_b is not None:
            tcga_delta = ref_a - ref_b
            entry["tcga_delta"] = float(tcga_delta)
            entry["p_sign_agrees_with_tcga"] = float(
                (np.sign(v) == np.sign(tcga_delta)).mean())
        out[name] = entry

    if regimes:
        vals, counts = np.unique(regimes, return_counts=True)
        out["_regime_distribution"] = {
            str(v): int(c) for v, c in zip(vals, counts)}
        out["_p_regime_matches_tcga"] = float(
            np.mean([r == ref.get("regime") for r in regimes]))
    return out


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--features", default=None,
                    help="feature dir (default $DATA_ROOT/luad_cptac/features)")
    ap.add_argument("--genes", nargs="+", default=GENES)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--edim", type=int, default=128,
                    help="projection width for the frozen 512-d CTransPath features")
    ap.add_argument("--max-tiles", type=int, default=600,
                    help="cap on tiles per patient bag (pooled across slides)")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--tcga-reference", default=None,
                    help="JSON overriding the built-in TCGA reference table")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    root = resolve_root(args.data_root)
    features_dir = Path(args.features) if args.features else root / "features"
    reference = dict(TCGA_REFERENCE)
    if args.tcga_reference:
        reference.update(json.loads(Path(args.tcga_reference).read_text()))

    bags = load_patient_bags(features_dir, args.max_tiles, rng)
    patients = sorted(bags)
    labels = load_labels(root, patients, args.genes)

    if args.smoke:
        args.epochs = min(args.epochs, 5)
        args.n_boot = min(args.n_boot, 50)

    conditions = ["B2", "B3", "PI", "FC"]
    pdim = len(PATTERNS)
    results, regimes, boot_rows, inter_rows = {}, {}, [], []

    for gene in args.genes:
        y_all = labels[gene].to_numpy(dtype=np.float32)
        keep = ~np.isnan(y_all)
        pats = [p for p, k in zip(patients, keep) if k]
        y = y_all[keep]
        n_pos, n_neg = int(y.sum()), int(len(y) - y.sum())
        powered = n_pos >= ESTIMATION_FLOOR
        print(f"\n=== {gene}: n={len(y)}, n_+={n_pos}, n_-={n_neg}, "
              f"floor={ESTIMATION_FLOOR} -> {'POWERED' if powered else 'UNDERPOWERED'} ===")

        if n_pos < MIN_POS_ATTEMPT or n_neg < MIN_POS_ATTEMPT:
            print(f"  -> SKIP: nothing is learnable at n_+={n_pos}, n_-={n_neg}")
            regimes[gene] = {"verdict": "Inconclusive", "cptac_regime": None,
                             "tcga_regime": reference.get(gene, {}).get("regime"),
                             "signature": "not attempted (below fitting minimum)",
                             "n": len(y), "n_pos": n_pos, "powered": False}
            results[gene] = {"skipped": "below fitting minimum",
                             "n_pos": n_pos, "n_neg": n_neg}
            continue

        n_splits = 1 if args.smoke else N_FOLDS
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
        splits = list(skf.split(np.zeros(len(y)), y))[:n_splits]

        oof = {c: np.full(len(y), np.nan) for c in conditions}
        per_fold = {c: [] for c in conditions}

        for fi, (tr_i, te_i) in enumerate(splits, 1):
            tr_bags = [bags[pats[i]] for i in tr_i]
            te_bags = [bags[pats[i]] for i in te_i]
            for c in conditions:
                probs, inter = train_eval(
                    c, BagDataset(tr_bags, y[tr_i]), BagDataset(te_bags, y[te_i]),
                    args.edim, pdim, args.epochs, args.batch_size, args.lr, device)
                oof[c][te_i] = probs
                a = safe_auroc(y[te_i], probs)
                per_fold[c].append(a)
                print(f"  fold{fi} {c}: AUROC {a:.3f}")
                if c == "FC" and inter is not None:
                    for a_i in range(pdim):
                        for b_i in range(a_i + 1, pdim):
                            inter_rows.append({
                                "gene": gene, "fold": fi,
                                "pattern_j": PATTERNS[a_i], "pattern_k": PATTERNS[b_i],
                                "I_jk": float(inter[a_i, b_i]),
                            })

        # Pooled out-of-fold AUROC is the point estimate; with ~21 test patients
        # per fold the mean-of-folds is dominated by fold composition.
        evaluated = ~np.isnan(oof["B2"])
        pooled = {c: safe_auroc(y[evaluated], oof[c][evaluated]) for c in conditions}
        fold_mean = {c: float(np.nanmean(v)) for c, v in per_fold.items()}
        fold_sd = {c: float(np.nanstd(v)) for c, v in per_fold.items()}

        ref = reference.get(gene, {})
        sig = regime_signature(pooled)
        cptac_regime = SIGNATURE_TO_REGIME.get(sig, "other")
        best = max(v for v in pooled.values() if not np.isnan(v))

        boot = bootstrap_contrasts(
            {c: oof[c][evaluated] for c in conditions}, y[evaluated],
            ref, args.n_boot, rng)

        results[gene] = {"pooled_oof": pooled, "fold_mean": fold_mean,
                         "fold_sd": fold_sd, "folds": per_fold,
                         "n": int(evaluated.sum()), "n_pos": n_pos}
        regimes[gene] = {
            "best_auroc": best,
            "verdict": "Conclusive" if best >= CONCLUSIVE_THRESHOLD else "Inconclusive",
            "signature": sig,
            "cptac_regime": cptac_regime,
            "tcga_regime": ref.get("regime"),
            "agrees_with_tcga": (cptac_regime == ref.get("regime")),
            "p_regime_matches_tcga": boot.get("_p_regime_matches_tcga"),
            "powered": powered,
            "n_pos": n_pos,
            "caveat": None if powered else
                      f"n_+={n_pos} is below the {ESTIMATION_FLOOR} floor; the "
                      "regime read is reported but not adjudicated",
        }
        for name, entry in boot.items():
            if name.startswith("_"):
                continue
            boot_rows.append({"gene": gene, "contrast": name, **entry})

        print(f"  pooled OOF: " + ", ".join(f"{c}={pooled[c]:.3f}" for c in conditions))
        print(f"  -> {cptac_regime} vs TCGA {ref.get('regime')} "
              f"[{'agree' if cptac_regime == ref.get('regime') else 'DIFFER'}], "
              f"P(regime reproduces)={boot.get('_p_regime_matches_tcga')}")

    # Carry the feature arm into the output path. The native, Reinhard and
    # field-of-view-matched arms are meant to be compared against each other, so
    # a fixed output directory would have each run silently overwrite the last.
    arm = features_dir.name
    arm = arm[len("features"):] if arm.startswith("features") else f"_{arm}"
    out_dir = root / "derived" / f"domainA_v2{arm}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2))
    (out_dir / "regimes.json").write_text(json.dumps(regimes, indent=2))
    if boot_rows:
        pd.DataFrame(boot_rows).to_csv(out_dir / "contrast_bootstrap.csv", index=False)
    if inter_rows:
        pd.DataFrame(inter_rows).to_csv(out_dir / "choquet_interactions.csv", index=False)

    print(f"\nwrote {out_dir}")
    print(json.dumps(regimes, indent=2))
    powered = [g for g, r in regimes.items() if r.get("powered")]
    print(f"\nAdjudicable at the n_+={ESTIMATION_FLOOR} floor: "
          f"{powered or 'NONE'}. All other rows are reported, not adjudicated.")


if __name__ == "__main__":
    main()
