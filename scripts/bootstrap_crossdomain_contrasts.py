"""Patient-clustered (and, with several seeds, seed-and-patient hierarchical)
bootstrap of the four-condition contrasts for the ECG and glioma arms
(E9 / E9b of the DGX plan).

Reads the OOF files written by the patched runners. Two layouts are accepted:

    <oof-dir>/oof_<cond>_<target>_fold<k>.csv            one seed (E9)
    <oof-dir>/seed<S>/oof_<cond>_<target>_fold<k>.csv    several seeds (E9b)

For every target the out-of-fold predictions are pooled across folds. Each
draw resamples *patients* with replacement (all records of a sampled patient
travel together, so the ECG arm respects its 21,799-records-from-18,869-
patients structure) and, when several seeds are present, resamples seeds
with replacement as well, concatenating the sampled seeds' predictions for
the sampled patients before every AUROC is recomputed. The percentile
interval of each contrast is what the noise-aware rule reads:

    PI-B2   additive complementarity (ii-a)      -> must exclude 0
    FC-PI   non-additive complementarity (ii-b)  -> must exclude 0
    FC-B2 } equivalence read for regimes (i)/(iii) -> both inside +-margin
    PI-B2 }
    B3-B2   what the vocabulary omits

With several seeds the table also reports, per contrast, on how many seeds
the point contrast is positive, as in the LUAD contrast table.

Usage:
    python bootstrap_crossdomain_contrasts.py --oof-dir $DATA_ROOT/glioma/derived/domainB/oof
    python bootstrap_crossdomain_contrasts.py --oof-dir $DATA_ROOT/ecg/derived/domainC/oof
Writes bootstrap_contrasts.csv, bootstrap_contrasts.md and bootstrap_aurocs.csv
next to the oof/ directory (or under --out-dir).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

CONTRASTS = [("PI", "B2"), ("FC", "B2"), ("FC", "PI"), ("B3", "B2")]
ROLE = {
    ("PI", "B2"): "(ii-a) decider; equivalence gate",
    ("FC", "B2"): "equivalence gate",
    ("FC", "PI"): "(ii-b) decider",
    ("B3", "B2"): "what the vocabulary omits",
}
COND_ORDER = ["B2", "B3", "PI", "FC"]


def load(oof_dir: Path) -> pd.DataFrame:
    seed_dirs = sorted(p for p in oof_dir.glob("seed*") if p.is_dir())
    frames = []
    if seed_dirs:
        for sd in seed_dirs:
            seed = int(sd.name.replace("seed", ""))
            files = sorted(sd.glob("oof_*_fold*.csv"))
            if not files:
                sys.exit(f"no oof_*_fold*.csv under {sd}")
            for f in files:
                df = pd.read_csv(f, dtype={"record_id": str, "patient_id": str})
                df["seed"] = seed
                frames.append(df)
    else:
        files = sorted(oof_dir.glob("oof_*_fold*.csv"))
        if not files:
            sys.exit(f"no oof_*_fold*.csv (or seed*/ directories) under {oof_dir}")
        for f in files:
            df = pd.read_csv(f, dtype={"record_id": str, "patient_id": str})
            df["seed"] = 0
            frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    need = {"record_id", "patient_id", "target", "condition", "fold", "label", "prob"}
    missing = need - set(df.columns)
    if missing:
        sys.exit(f"OOF files lack columns {sorted(missing)}")
    df = df.dropna(subset=["prob"])
    dup = df.duplicated(["seed", "target", "condition", "record_id"], keep=False)
    if dup.any():
        sys.exit("duplicate (seed, target, condition, record) rows: a fold was written twice?")
    return df


def wide(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str], list[int]]:
    """One row per (seed, record); one column per condition."""
    sub = df[df["target"] == target]
    conds = sorted(sub["condition"].unique(), key=COND_ORDER.index)
    seeds = sorted(sub["seed"].unique())
    probs = sub.pivot_table(index=["seed", "record_id"], columns="condition",
                            values="prob", aggfunc="first")
    labels = sub.groupby("record_id")["label"].agg(["min", "max", "first"])
    if (labels["min"] != labels["max"]).any():
        sys.exit(f"{target}: a record carries different labels across conditions/seeds")
    patients = sub.groupby("record_id")["patient_id"].first()
    complete = probs.dropna(subset=conds)
    dropped = len(probs) - len(complete)
    if dropped:
        print(f"[{target}] {dropped} (seed, record) rows lack a condition and are dropped")
    out = complete.reset_index()
    out["label"] = labels.loc[out["record_id"], "first"].to_numpy().astype(int)
    out["patient_id"] = patients.loc[out["record_id"]].to_numpy().astype(str)
    return out, conds, seeds


def aurocs(y: np.ndarray, P: dict[str, np.ndarray], rows: np.ndarray) -> dict[str, float]:
    yy = y[rows]
    if yy.min() == yy.max():
        return {c: np.nan for c in P}
    return {c: roc_auc_score(yy, P[c][rows]) for c in P}


def bootstrap(block: pd.DataFrame, conds: list[str], seeds: list[int],
              n_boot: int, rng_seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(rng_seed)
    y = block["label"].to_numpy()
    P = {c: block[c].to_numpy() for c in conds}
    seed_arr = block["seed"].to_numpy()
    # patient -> row positions, per seed (each seed is a full replicate of the cohort)
    by_seed = {}
    for s in seeds:
        sub = block[seed_arr == s]
        groups = sub.groupby("patient_id").indices
        pids = np.array(sorted(groups))
        by_seed[s] = (pids, [sub.index.to_numpy()[groups[p]] for p in pids])
    pids_ref = by_seed[seeds[0]][0]
    draws = []
    for _ in range(n_boot):
        pick_pat = rng.integers(0, len(pids_ref), len(pids_ref))
        pick_seeds = rng.choice(seeds, size=len(seeds), replace=True) if len(seeds) > 1 else seeds
        rows = []
        for s in pick_seeds:
            pids, idx = by_seed[s]
            if len(pids) != len(pids_ref):
                sys.exit("seeds cover different patient sets; cannot resample jointly")
            rows.append(np.concatenate([idx[i] for i in pick_pat]))
        draws.append(aurocs(y, P, np.concatenate(rows)))
    return pd.DataFrame(draws)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--oof-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42, help="bootstrap RNG seed")
    ap.add_argument("--margin", type=float, default=0.02)
    args = ap.parse_args()

    out_dir = args.out_dir or args.oof_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    df = load(args.oof_dir)

    auroc_rows, contrast_rows, md = [], [], []
    for target in sorted(df["target"].unique()):
        block, conds, seeds = wide(df, target)
        n_seeds = len(seeds)
        n_rec = block["record_id"].nunique()
        n_pat = block["patient_id"].nunique()
        n_pos = int(block.drop_duplicates("record_id")["label"].sum())
        all_rows = np.arange(len(block))
        y = block["label"].to_numpy(); P = {c: block[c].to_numpy() for c in conds}
        point = aurocs(y, P, all_rows)                      # pooled over seeds
        per_seed = {s: aurocs(y, P, np.flatnonzero(block["seed"].to_numpy() == s))
                    for s in seeds}
        boot = bootstrap(block, conds, seeds, args.n_boot, args.seed)
        mode = (f"{n_seeds} seeds x patients hierarchical" if n_seeds > 1
                else "patient-cluster")
        print(f"\n=== {target}: {n_rec} records, {n_pat} patients, n+={n_pos}, "
              f"{args.n_boot} {mode} draws ===")
        for c in conds:
            lo, hi = np.nanpercentile(boot[c], [2.5, 97.5])
            seed_vals = [per_seed[s][c] for s in seeds]
            auroc_rows.append(dict(target=target, condition=c, auroc=point[c],
                                   ci_lo=lo, ci_hi=hi, n_records=n_rec,
                                   n_patients=n_pat, n_pos=n_pos, n_seeds=n_seeds,
                                   seed_mean=float(np.nanmean(seed_vals)),
                                   seed_sd=float(np.nanstd(seed_vals)) if n_seeds > 1 else np.nan))
            print(f"  {c}: {point[c]:.3f} [{lo:.3f}, {hi:.3f}]"
                  + (f"  per-seed {np.nanmean(seed_vals):.3f}+-{np.nanstd(seed_vals):.3f}"
                     if n_seeds > 1 else ""))
        md.append(f"\n**{target}** ({n_rec} records, {n_pat} patients, n+={n_pos}, "
                  f"{n_seeds} seed{'s' if n_seeds > 1 else ''})\n\n"
                  f"| contrast | point | 95% CI | excludes 0 | inside ±{args.margin:.2f} | "
                  f"seeds >0 | role |\n|---|---|---|---|---|---|---|")
        for a, b in CONTRASTS:
            if a not in conds or b not in conds:
                continue
            d = boot[a] - boot[b]
            lo, hi = np.nanpercentile(d, [2.5, 97.5])
            pt = point[a] - point[b]
            excl = bool(lo > 0 or hi < 0)
            equiv = bool(lo > -args.margin and hi < args.margin)
            n_pos_seeds = int(sum((per_seed[s][a] - per_seed[s][b]) > 0 for s in seeds))
            contrast_rows.append(dict(target=target, contrast=f"{a}-{b}", point=pt,
                                      ci_lo=lo, ci_hi=hi, excludes_zero=excl,
                                      equivalent=equiv, margin=args.margin,
                                      n_seeds=n_seeds, seeds_positive=n_pos_seeds,
                                      role=ROLE[(a, b)]))
            print(f"  {a}-{b}: {pt:+.3f} [{lo:+.3f}, {hi:+.3f}]"
                  f"  excludes0={excl}  equiv±{args.margin}={equiv}"
                  f"  seeds>0={n_pos_seeds}/{n_seeds}")
            md.append(f"| {a}−{b} | {pt:+.3f} | [{lo:+.3f}, {hi:+.3f}] | "
                      f"{'yes' if excl else 'no'} | {'yes' if equiv else 'no'} | "
                      f"{n_pos_seeds}/{n_seeds} | {ROLE[(a, b)]} |")
        cr = {r["contrast"]: r for r in contrast_rows if r["target"] == target}
        read = "no regime from contrasts"
        if cr.get("PI-B2", {}).get("excludes_zero") and cr["PI-B2"]["point"] > 0 \
                and not cr["PI-B2"]["equivalent"]:
            read = "(ii-a) additive complementarity"
        if cr.get("FC-PI", {}).get("excludes_zero") and cr["FC-PI"]["point"] > 0 \
                and not cr["FC-PI"]["equivalent"]:
            read = "(ii-b) candidate (needs the conditional test)"
        if cr.get("PI-B2", {}).get("equivalent") and cr.get("FC-B2", {}).get("equivalent"):
            read = "(i)/(iii) equivalence established; (i) vs (iii) from B3 and the floor"
        best_hi = max(r["ci_hi"] for r in auroc_rows if r["target"] == target)
        if best_hi < 0.70:
            read += f"; level read: every upper bound < 0.70 (best {best_hi:.3f}) -> modality ceiling (iii)"
        md.append(f"\nNoise-aware read: **{read}**")
        print(f"  -> {read}")

    pd.DataFrame(auroc_rows).to_csv(out_dir / "bootstrap_aurocs.csv", index=False)
    pd.DataFrame(contrast_rows).to_csv(out_dir / "bootstrap_contrasts.csv", index=False)
    (out_dir / "bootstrap_contrasts.md").write_text(
        f"# Bootstrap, {args.n_boot} draws, RNG seed {args.seed}\n"
        + "\n".join(md) + "\n", encoding="utf-8")
    print(f"\nwrote {out_dir / 'bootstrap_contrasts.csv'}")


if __name__ == "__main__":
    main()
