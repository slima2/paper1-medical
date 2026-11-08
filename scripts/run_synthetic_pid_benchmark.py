#!/usr/bin/env python3
"""Calibrate the regime read-out against cohorts whose atoms are known.

Generates the synthetic cohorts of synthetic_pid_cohort.py, runs the unmodified
benchmark over each, applies the same adjudication rule used on LUAD, and
reports the confusion matrix of true regime against read regime. The question
it answers is the one the manuscript currently assumes: at this cohort size and
with this instrument, how often does the four-condition signature recover the
atom configuration that generated the data, and which configurations does it
confuse?

Run on the DGX, where the benchmark and its GPU live:

    python scripts/run_synthetic_pid_benchmark.py \
        --benchmark "/path/to/pattern_informed_abmil_benchmark_v2 (3).py" \
        --work /home/rapids/notebooks/slima/synthetic \
        --n-patients 444 --seeds 0 1 2 3 4

Both the generation and the benchmark stage are resumable: an existing cohort
or an existing fold output is left alone, so the job can be killed and resumed.
Use --analyse-only once the runs are done.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic_pid_cohort import CONFIGS  # noqa: E402

GENE = "SYNTH"
CONDITIONS = {
    "B2": "baseline2_abmil_embeddings",
    "B3": "baseline3_abmil_patterns",
    "PI": "proposed_abmil_concat",
    "FC": "proposed_fuzzy_choquet",
}
DEFAULT_CONFIGS = ["EMPTY", "R", "UE", "UP", "R_UE", "UP_UE", "S"]
MARGIN = 0.02
N_BOOT = 2000
CHANCE_BAND = 0.02  # |AUROC - 0.5| below this counts as chance


def auroc(y: np.ndarray, score: np.ndarray) -> float:
    """Rank-based AUROC, ties averaged. Avoids a sklearn dependency here."""
    pos, neg = y == 1, y == 0
    n_pos, n_neg = int(pos.sum()), int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score), dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    # average ranks within ties
    srt = score[order]
    i = 0
    while i < len(srt):
        j = i
        while j + 1 < len(srt) and srt[j + 1] == srt[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


# ───────────────────────────── generation + run ──────────────────────────────

def cohort_dir(work: Path, config: str, seed: int) -> Path:
    return work / "cohorts" / config / f"seed_{seed}"


def results_dir(work: Path, config: str, seed: int) -> Path:
    return work / "results" / config / f"seed_{seed}"


def generate(work: Path, config: str, seed: int, args: argparse.Namespace) -> None:
    out = cohort_dir(work, config, seed)
    if (out / "ground_truth.json").exists():
        print(f"  [skip] cohort {config}/seed_{seed} exists")
        return
    cmd = [
        sys.executable, str(Path(__file__).parent / "synthetic_pid_cohort.py"),
        "--config", config, "--seed", str(seed), "--out", str(out),
        "--n-patients", str(args.n_patients),
        "--embed-dim", str(args.embed_dim),
        "--effect", str(args.effect),
        "--witness-rate", str(args.witness_rate),
        "--label-noise", str(args.label_noise),
        "--min-tiles", str(args.min_tiles),
        "--max-tiles", str(args.max_tiles),
    ]
    subprocess.run(cmd, check=True)


def run_benchmark(work: Path, config: str, seed: int, args: argparse.Namespace) -> None:
    res = results_dir(work, config, seed)
    expected = [
        res / "oof" / f"oof_{cond}_{GENE}_fold{f}.csv"
        for cond in CONDITIONS.values()
        for f in range(args.n_folds)
    ]
    if all(p.exists() for p in expected):
        print(f"  [skip] results {config}/seed_{seed} complete")
        return
    res.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable, str(args.benchmark),
        "--data_dir", str(cohort_dir(work, config, seed)),
        "--results_dir", str(res),
        "--genes", GENE,
        "--seed", str(seed),
        "--n_folds", str(args.n_folds),
        "--split_mode", args.protocol,
        "--conditions", *CONDITIONS.values(),
    ]
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


# ──────────────────────────────── analysis ───────────────────────────────────

def pooled_oof(work: Path, config: str, seed: int, args: argparse.Namespace):
    """Pool out-of-fold predictions across folds, one frame per condition."""
    out = {}
    for short, long in CONDITIONS.items():
        frames = []
        for f in range(args.n_folds):
            path = results_dir(work, config, seed) / "oof" / f"oof_{long}_{GENE}_fold{f}.csv"
            if path.exists():
                frames.append(pd.read_csv(path))
        if not frames:
            return None
        out[short] = pd.concat(frames, ignore_index=True)
    return out


def bootstrap_contrasts(oof: dict, rng: np.random.Generator) -> dict:
    """Hierarchical patient bootstrap of the four conditions and contrasts."""
    key = next(iter(oof))
    ids = oof[key]["slide_id"].to_numpy()
    unique = np.unique(ids)
    # align every condition on the same slide order
    aligned = {}
    for cond, frame in oof.items():
        f = frame.drop_duplicates("slide_id").set_index("slide_id").loc[unique]
        aligned[cond] = (f["label"].to_numpy(), f["prob"].to_numpy())

    point = {c: auroc(*aligned[c]) for c in aligned}
    draws = {c: [] for c in aligned}
    n = len(unique)
    for _ in range(N_BOOT):
        idx = rng.integers(0, n, size=n)
        for c, (y, p) in aligned.items():
            draws[c].append(auroc(y[idx], p[idx]))

    res = {"point": point, "ci": {}}
    for name, (a, b) in {"PI-B2": ("PI", "B2"), "FC-B2": ("FC", "B2"),
                         "FC-PI": ("FC", "PI"), "B3-B2": ("B3", "B2")}.items():
        d = np.asarray(draws[a]) - np.asarray(draws[b])
        d = d[~np.isnan(d)]
        res["ci"][name] = (point[a] - point[b],
                           float(np.percentile(d, 2.5)),
                           float(np.percentile(d, 97.5)))
    return res


def read_regime(res: dict, use_b3_level: bool) -> str:
    """The adjudication rule, applied exactly as it is on LUAD.

    With use_b3_level=False this is the fusion-contrast signature alone, which
    is what the manuscript says collides between (i) and (iii). With
    use_b3_level=True the level of the vocabulary-only arm is allowed to
    separate them, which is what the synthetic run is here to test.
    """
    ci = res["ci"]
    pt = res["point"]

    lo, hi = ci["FC-PI"][1], ci["FC-PI"][2]
    if lo > 0:
        return "(ii-b) synergy"
    lo, hi = ci["PI-B2"][1], ci["PI-B2"][2]
    if lo > 0:
        return "(ii-a) complementarity"

    inside = all(
        abs(ci[c][1]) <= MARGIN and abs(ci[c][2]) <= MARGIN
        for c in ("PI-B2", "FC-B2")
    )
    if not inside:
        return "unresolved"

    b2_up = pt["B2"] - 0.5 > CHANCE_BAND
    b3_up = pt["B3"] - 0.5 > CHANCE_BAND
    if not b2_up and not b3_up:
        return "empty / ceiling"
    if not use_b3_level:
        return "(i) or (iii) [collided]"
    return "(i) redundancy" if b3_up else "(iii) out-of-ontology"


TRUTH = {
    "EMPTY": "empty / ceiling",
    "R": "(i) redundancy",
    "UE": "(iii) out-of-ontology",
    "UP": "(ii-a) complementarity",
    "R_UE": "(i) redundancy",
    "UP_UE": "(ii-a) complementarity",
    "S": "(ii-b) synergy",
}


def analyse(work: Path, args: argparse.Namespace) -> None:
    rows = []
    rng = np.random.default_rng(12345)
    for config in args.configs:
        for seed in args.seeds:
            oof = pooled_oof(work, config, seed, args)
            if oof is None:
                print(f"  [missing] {config}/seed_{seed}")
                continue
            res = bootstrap_contrasts(oof, rng)
            row = {
                "config": config,
                "seed": seed,
                "truth": TRUTH[config],
                "read_collided": read_regime(res, use_b3_level=False),
                "read_with_b3": read_regime(res, use_b3_level=True),
                **{f"auroc_{c}": round(v, 4) for c, v in res["point"].items()},
            }
            for name, (p, lo, hi) in res["ci"].items():
                row[f"{name}"] = round(p, 4)
                row[f"{name}_lo"] = round(lo, 4)
                row[f"{name}_hi"] = round(hi, 4)
            rows.append(row)

    if not rows:
        print("nothing to analyse")
        return
    df = pd.DataFrame(rows)
    out = work / "synthetic_regime_reads.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out}\n")

    print("Per-condition AUROC by configuration (mean over seeds)")
    print(df.groupby("config")[[f"auroc_{c}" for c in CONDITIONS]].mean().round(3))

    for col, title in (("read_with_b3", "using the level of the vocabulary arm"),
                       ("read_collided", "fusion contrasts alone")):
        print(f"\nConfusion matrix, {title}")
        cm = pd.crosstab(df["truth"], df[col])
        print(cm)
        correct = sum(
            int(r["truth"] == r[col]) for _, r in df.iterrows()
        )
        print(f"  recovered {correct}/{len(df)} = {correct / len(df):.0%}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--work", required=True, type=Path)
    p.add_argument("--benchmark", type=Path,
                   help="path to pattern_informed_abmil_benchmark_v2.py")
    p.add_argument("--configs", nargs="+", default=DEFAULT_CONFIGS,
                   choices=sorted(CONFIGS))
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--protocol", default="patient", choices=("patient", "site"))
    p.add_argument("--n-patients", type=int, default=444)
    p.add_argument("--n-folds", type=int, default=5)
    p.add_argument("--embed-dim", type=int, default=512)
    p.add_argument("--effect", type=float, default=0.6)
    p.add_argument("--witness-rate", type=float, default=0.25)
    p.add_argument("--label-noise", type=float, default=0.15)
    p.add_argument("--min-tiles", type=int, default=200)
    p.add_argument("--max-tiles", type=int, default=800)
    p.add_argument("--analyse-only", action="store_true")
    args = p.parse_args()

    args.work.mkdir(parents=True, exist_ok=True)

    if not args.analyse_only:
        if not args.benchmark:
            p.error("--benchmark is required unless --analyse-only is given")
        for config in args.configs:
            for seed in args.seeds:
                print(f"[{config} seed {seed}]")
                generate(args.work, config, seed, args)
                run_benchmark(args.work, config, seed, args)

    analyse(args.work, args)


if __name__ == "__main__":
    main()
