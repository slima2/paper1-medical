#!/usr/bin/env python
"""Domain-C v2 run: ECG -> laboratory state on MIMIC-IV-ECG, with IPW.

The measured version of the pre-registered laboratory targets. Same two
channels and four conditions as the PTB-XL v1 run, but on MIMIC-IV-ECG and with
every metric inverse-probability-weighted for the non-random observation of
labs.

    encoder channel E : learned 1D-CNN of the raw 12-lead signal (500->100 Hz)
    source  channel P : 71 SCP statements PROPAGATED from PTB-XL
                        (scp_propagated.parquet)
    target        Y   : a binarised lab value (ecg_lab_labels.parquet), read
                        only on the observed subset, weighted by ipw_weights

Conditions (mirroring tab:mutation_luad / domain C v1):
    B2  encoder only          B3  statement channel only
    PI  concat fusion         FC  encoder + 2-additive Choquet over statements

Faithful to the manuscript, the decomposition is reported through its
model-side signatures (which condition wins, B3-vs-B2, PI/FC-vs-B2 and the
Choquet interaction matrix), not as numeric PID atoms, and the regime is read
DIRECTIONALLY per tab:atom_operationalization. A verdict is Inconclusive unless
the IPW-weighted best AUROC >= 0.70. Both IPW and unweighted AUROC are printed.

Guardians (cite alongside any regime-(iii) verdict):
    propagation_quality.csv   is the P channel recoverable at all?
    propensity_quality.csv    is the observation model informative (IPW needed)?

Usage (DGX, after `source env.sh`):
    python run_ecg_domainC_v2.py --smoke --targets ntprobnp
    python run_ecg_domainC_v2.py --targets ntprobnp troponin_i

Outputs under $DATA_ROOT/ecg/derived/domainC_v2/:
    results_v2.json  regimes_v2.json  choquet_interactions_v2.csv
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

SEED = 42
CONCLUSIVE_THRESHOLD = 0.70
N_FOLDS = 5


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "ecg"


# --- models (identical primitives to run_ecg_domainC.py) ------------------

class ECGEncoder(nn.Module):
    def __init__(self, in_ch: int = 12, dim: int = 128):
        super().__init__()
        def block(ci, co, k=7, s=2):
            return nn.Sequential(
                nn.Conv1d(ci, co, k, stride=s, padding=k // 2),
                nn.BatchNorm1d(co), nn.ReLU(inplace=True))
        self.net = nn.Sequential(block(in_ch, 32), block(32, 64), block(64, 128),
                                 block(128, 128), block(128, dim))
        self.pool = nn.AdaptiveAvgPool1d(1); self.dim = dim

    def forward(self, x):
        return self.pool(self.net(x)).squeeze(-1)


class Choquet2Additive(nn.Module):
    def __init__(self, n: int, l1: float = 1e-3):
        super().__init__()
        self.n = n
        self.psi = nn.Parameter(torch.zeros(n))
        self.iflat = nn.Parameter(torch.zeros(n * (n - 1) // 2))
        self.l1 = l1
        self.register_buffer("iu", torch.triu_indices(n, n, offset=1))

    def interaction_matrix(self):
        I = torch.zeros(self.n, self.n, device=self.iflat.device)
        I[self.iu[0], self.iu[1]] = self.iflat
        return I + I.t()

    def forward(self, P):
        phi = torch.softmax(self.psi, dim=0)
        I = self.interaction_matrix()
        order = torch.argsort(P, dim=1)
        rank = torch.argsort(order, dim=1)
        Ps = torch.gather(P, 1, order)
        C = torch.zeros(P.size(0), device=P.device)
        prev = torch.zeros(P.size(0), device=P.device)
        for k in range(self.n):
            mask = (rank >= k).to(P.dtype)
            gk = (mask * phi.unsqueeze(0)).sum(1)
            gk = gk + 0.5 * torch.einsum("bi,bj,ij->b", mask, mask, I)
            C = C + (Ps[:, k] - prev) * gk
            prev = Ps[:, k]
        return C.unsqueeze(1)

    def l1_penalty(self):
        return self.l1 * self.iflat.abs().sum()


class Head(nn.Module):
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
        c = self.choquet(P)
        return self.clf(torch.cat([self.enc(x), c], dim=1))


def train_eval(condition, Xtr, Ptr, ytr, wtr, Xte, Pte, yte, wte,
               epochs, bs, lr, device):
    """Train with IPW-weighted BCE; return (IPW AUROC, unweighted AUROC, I_jk)."""
    model = Head(condition, edim=128, pdim=Ptr.shape[1]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    pos = float((ytr * wtr).sum()); neg = float((wtr).sum() - pos)
    pw = torch.tensor([neg / max(pos, 1.0)], device=device)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")

    tr = DataLoader(TensorDataset(Xtr, Ptr, ytr, wtr), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        model.train()
        for xb, pb, yb, wb in tr:
            xb, pb, yb, wb = (t.to(device) for t in (xb, pb, yb, wb))
            opt.zero_grad()
            logit = model(xb, pb).squeeze(1)
            loss = (lossf(logit, yb) * wb).mean()
            if condition == "FC":
                loss = loss + model.choquet.l1_penalty()
            loss.backward(); opt.step()

    model.eval()
    with torch.no_grad():
        te = DataLoader(TensorDataset(Xte, Pte), batch_size=512, shuffle=False)
        probs = [torch.sigmoid(model(xb.to(device), pb.to(device)).squeeze(1)).cpu().numpy()
                 for xb, pb in te]
    prob = np.concatenate(probs)
    y = yte.numpy(); w = wte.numpy()
    au_w = roc_auc_score(y, prob, sample_weight=w) if 0 < y.sum() < len(y) else np.nan
    au_u = roc_auc_score(y, prob) if 0 < y.sum() < len(y) else np.nan
    inter = model.choquet.interaction_matrix().detach().cpu().numpy() if condition == "FC" else None
    return float(au_w), float(au_u), inter


def regime_signature(cm: dict) -> str:
    """Directional read per tab:atom_operationalization (IPW means)."""
    b2, b3 = cm.get("B2"), cm.get("B3")
    pi, fc = cm.get("PI"), cm.get("FC")
    fused = max([v for v in (pi, fc) if v is not None] or [b2 or 0])
    if b3 is not None and b2 is not None and b3 > b2 + 0.01:
        return "pattern-side unique/floor (B3 > B2)"
    if fc is not None and pi is not None and fc > pi + 0.01:
        return "synergy ii-b (FC > PI)"
    if fused > max(b2 or 0, b3 or 0) + 0.015:
        return "additive complementarity ii-a (fusion > best single)"
    if (max(cm.values()) - (b2 or 0)) <= 0.02:
        return "encoder saturation i (fusion ~ encoder)"
    return "mixed"


def load_waveforms(paths, wf_dir, limit=None):
    n = len(paths) if limit is None else min(limit, len(paths))
    sig = np.zeros((n, 12, 1000), dtype=np.float32)
    ok = np.zeros(n, dtype=bool)
    t0 = time.time()
    for i in range(n):
        try:
            data, _ = wfdb.rdsamp(str(wf_dir / paths[i]))
            x = data.T.astype(np.float32)
            if x.shape[0] != 12:
                continue
            T = x.shape[1]
            idx = np.linspace(0, T - 1, 1000).astype(int) if T >= 1000 else None
            x = x[:, idx] if idx is not None else np.pad(x, ((0, 0), (0, 1000 - T)))
            sig[i] = np.nan_to_num(x); ok[i] = True
        except Exception:
            continue
        if (i + 1) % 2000 == 0:
            print(f"  loaded {i + 1}/{n} ({time.time() - t0:.0f}s)", end="\r")
    print()
    mu = sig[ok].mean(axis=(0, 2), keepdims=True) if ok.any() else 0
    sd = sig[ok].std(axis=(0, 2), keepdims=True) + 1e-6 if ok.any() else 1
    sig = (sig - mu) / sd
    return sig, ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--targets", nargs="+", default=["ntprobnp", "troponin_i"])
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--choquet", dest="choquet", action="store_true", default=True)
    ap.add_argument("--no-choquet", dest="choquet", action="store_false")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    ecg_root = resolve_root(args.data_root)
    d = ecg_root / "derived"
    for f in ("ecg_lab_labels.parquet", "scp_propagated.parquet", "ipw_weights.parquet"):
        if not (d / f).exists():
            sys.exit(f"missing {d / f}\nrun build_ecg_lab_labels / propagate_scp_to_mimic "
                     f"/ fit_lab_propensity first")

    labels = pd.read_parquet(d / "ecg_lab_labels.parquet")
    prop = pd.read_parquet(d / "scp_propagated.parquet")
    ipw = pd.read_parquet(d / "ipw_weights.parquet")
    stmt_cols = [c for c in prop.columns if c != "study_id"]

    # join on study_id; keep only records with a propagated P channel
    tab = labels.merge(prop, on="study_id", how="inner").merge(ipw, on="study_id", how="left")
    tab["fold"] = (tab["subject_id"].astype(np.int64) % N_FOLDS) + 1
    if args.smoke:
        tab = tab.groupby("fold", group_keys=False).head(400)
        args.epochs = min(args.epochs, 3)
    print(f"{len(tab):,} records with P channel; {tab['subject_id'].nunique():,} subjects")

    wf_dir = ecg_root / "mimic4" / "waveforms"
    path_map = pd.read_csv(wf_dir / "record_list.csv")[["study_id", "path"]]
    tab = tab.merge(path_map, on="study_id", how="left").dropna(subset=["path"])

    X, ok = load_waveforms(tab["path"].tolist(), wf_dir)
    tab = tab.iloc[: len(ok)].loc[ok].reset_index(drop=True)
    X = X[ok]
    P = tab[stmt_cols].to_numpy(dtype=np.float32)
    Xt = torch.from_numpy(X); Pt = torch.from_numpy(P)

    conditions = ["B2", "B3", "PI"] + (["FC"] if args.choquet else [])
    results, regimes, interactions = {}, {}, {}

    for target in args.targets:
        if target not in tab.columns:
            print(f"[warn] target {target} not in labels; skipping"); continue
        obs = tab[f"{target}_obs"].fillna(0).astype(int).to_numpy() == 1
        y_all = tab[target].to_numpy(dtype=np.float32)
        w_all = tab.get(f"w_{target}", pd.Series(np.ones(len(tab)))).to_numpy(dtype=np.float32)
        n_pos = int(np.nansum(y_all[obs]))
        print(f"\n=== target: {target}  observed={obs.sum():,}  n_+={n_pos} ===")
        if min(n_pos, obs.sum() - n_pos) < 37:
            print("  [warn] below estimation floor; verdict is a power statement")

        per_w = {c: [] for c in conditions}
        per_u = {c: [] for c in conditions}
        fc_acc = None
        for gf in range(1, N_FOLDS + 1):
            te = obs & (tab["fold"].to_numpy() == gf)
            tr = obs & (tab["fold"].to_numpy() != gf)
            if te.sum() < 20 or tr.sum() < 40 or len(np.unique(y_all[te])) < 2:
                continue
            yt = torch.from_numpy(np.nan_to_num(y_all))
            wt = torch.from_numpy(w_all)
            for c in conditions:
                au_w, au_u, inter = train_eval(
                    c, Xt[tr], Pt[tr], yt[tr], wt[tr],
                    Xt[te], Pt[te], yt[te], wt[te],
                    args.epochs, args.batch_size, args.lr, device)
                per_w[c].append(au_w); per_u[c].append(au_u)
                print(f"  fold{gf} {c}: IPW {au_w:.3f} | unw {au_u:.3f}")
                if c == "FC" and inter is not None:
                    fc_acc = inter if fc_acc is None else fc_acc + inter

        mean_w = {c: float(np.nanmean(v)) for c, v in per_w.items() if v}
        mean_u = {c: float(np.nanmean(v)) for c, v in per_u.items() if v}
        sd_w = {c: float(np.nanstd(v)) for c, v in per_w.items() if v}
        if not mean_w:
            print("  [warn] no evaluable fold"); continue
        results[target] = {"ipw_mean": mean_w, "ipw_sd": sd_w, "unweighted_mean": mean_u}
        best = max(mean_w.values())
        regimes[target] = {
            "best_auroc_ipw": best,
            "verdict": "Conclusive" if best >= CONCLUSIVE_THRESHOLD else "Inconclusive",
            "signature": regime_signature(mean_w),
        }
        print(f"  -> best IPW {best:.3f} [{regimes[target]['verdict']}] "
              f"{regimes[target]['signature']}")

        if fc_acc is not None:
            I = fc_acc / max(len([v for v in per_w['FC'] if v == v]), 1)
            pairs = [(stmt_cols[a], stmt_cols[b], float(I[a, b]))
                     for a in range(len(stmt_cols)) for b in range(a + 1, len(stmt_cols))]
            pairs.sort(key=lambda t: abs(t[2]), reverse=True)
            interactions[target] = pairs[:25]

    out_dir = d / "domainC_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results_v2.json").write_text(json.dumps(results, indent=2))
    (out_dir / "regimes_v2.json").write_text(json.dumps(regimes, indent=2))
    if interactions:
        rows = [{"target": t, "stmt_j": j, "stmt_k": k, "I_jk": v}
                for t, ps in interactions.items() for j, k, v in ps]
        pd.DataFrame(rows).to_csv(out_dir / "choquet_interactions_v2.csv", index=False)

    print(f"\nwrote {out_dir}")
    print(json.dumps(regimes, indent=2))


if __name__ == "__main__":
    main()
