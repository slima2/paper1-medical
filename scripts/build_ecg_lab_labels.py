#!/usr/bin/env python
"""Pair each MIMIC-IV-ECG record to a laboratory value -> binary targets.

Domain-C v2 step 1. For every ECG study in MIMIC-IV-ECG we attach the lab
measurement closest in time (within a window) for each requested target, then
binarise it by a clinical cutoff. Crucially we also record whether a lab was
observed at all, because labs are ordered on a NON-RANDOM subset of encounters
-- that missingness is what fit_lab_propensity.py later corrects by IPW.

Inputs (credentialed PhysioNet, under $DATA_ROOT/ecg/mimic4/):
    waveforms/record_list.csv           subject_id, study_id, ecg_time, path
    hosp/labevents.csv.gz               subject_id, itemid, charttime, valuenum
    hosp/d_labitems.csv.gz              itemid, label, fluid, category
    hosp/patients.csv.gz                subject_id, gender, anchor_age  (optional)

Output, under $DATA_ROOT/ecg/derived/:
    ecg_lab_labels.parquet   one row per (study_id) with, per target:
        <t>        binary label in {0,1} or NaN if not observed in window
        <t>_val    the raw valuenum used
        <t>_obs    1 if a lab existed in the window, else 0  (the IPW mask)
    plus keys: subject_id, study_id, ecg_time, age, sex

Targets are declared below with (label regex, cutoff, unit hint). Cutoffs are
pre-registered decisions; resolve itemids by label match against d_labitems
rather than hard-coding, because ids drift between MIMIC releases.

Usage (DGX, after `source env.sh`):
    python build_ecg_lab_labels.py --targets ntprobnp troponin_i --window-h 24
    python build_ecg_lab_labels.py --list-itemids ntprobnp   # inspect matches
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Pre-registered target definitions: name -> (label regex, cutoff, positive-if)
# The regex is matched (case-insensitive) against d_labitems.label.
TARGETS = {
    "ntprobnp":   {"regex": r"nt.?probnp|pro.?bnp",        "cutoff": 300.0,  "op": ">"},
    "bnp":        {"regex": r"\bbnp\b",                     "cutoff": 100.0,  "op": ">"},
    "troponin_i": {"regex": r"troponin\s*i\b",             "cutoff": 0.04,   "op": ">"},
    "troponin_t": {"regex": r"troponin\s*t\b",             "cutoff": 0.01,   "op": ">"},
    "potassium":  {"regex": r"^potassium",                  "cutoff": 5.0,    "op": ">"},
    "creatinine": {"regex": r"^creatinine",                 "cutoff": 1.3,    "op": ">"},
}


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "ecg"


def match_itemids(d_labitems: pd.DataFrame, regex: str) -> list[int]:
    m = d_labitems["label"].fillna("").str.contains(regex, case=False, regex=True)
    return d_labitems.loc[m, "itemid"].astype(int).tolist()


def load_target_labs(hosp: Path, itemids: set[int]) -> pd.DataFrame:
    """Stream labevents.csv.gz keeping only the requested itemids."""
    path = hosp / "labevents.csv.gz"
    if not path.exists():
        sys.exit(f"missing {path} (MIMIC-IV hosp module, credentialed)")
    usecols = ["subject_id", "itemid", "charttime", "valuenum"]
    keep = []
    reader = pd.read_csv(path, usecols=usecols, chunksize=2_000_000,
                         parse_dates=["charttime"])
    for i, chunk in enumerate(reader):
        sub = chunk[chunk["itemid"].isin(itemids)]
        if len(sub):
            keep.append(sub)
        print(f"  labevents chunk {i + 1}: kept {sum(len(k) for k in keep):,} rows",
              end="\r")
    print()
    if not keep:
        sys.exit("no labevents rows matched the requested itemids")
    labs = pd.concat(keep, ignore_index=True)
    labs = labs.dropna(subset=["valuenum", "charttime"])
    return labs


def pair_nearest(ecg: pd.DataFrame, labs: pd.DataFrame, window_h: int) -> pd.Series:
    """merge_asof nearest lab per ECG within +/- window, returns valuenum Series."""
    tol = pd.Timedelta(hours=window_h)
    left = ecg[["subject_id", "ecg_time"]].sort_values("ecg_time").reset_index()
    right = labs[["subject_id", "charttime", "valuenum"]].sort_values("charttime")
    merged = pd.merge_asof(
        left, right, left_on="ecg_time", right_on="charttime",
        by="subject_id", direction="nearest", tolerance=tol,
    )
    return merged.set_index("index")["valuenum"].reindex(ecg.index)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--targets", nargs="+", default=["ntprobnp", "troponin_i"])
    ap.add_argument("--window-h", type=int, default=24,
                    help="pairing half-window in hours (pre-registered 24; 72 sens.)")
    ap.add_argument("--list-itemids", metavar="TARGET",
                    help="print d_labitems rows matching a target and exit")
    args = ap.parse_args()

    ecg_root = resolve_root(args.data_root)
    mimic = ecg_root / "mimic4"
    hosp = mimic / "hosp"

    d_labitems = pd.read_csv(hosp / "d_labitems.csv.gz")

    if args.list_itemids:
        spec = TARGETS.get(args.list_itemids)
        if not spec:
            sys.exit(f"unknown target {args.list_itemids}; known: {list(TARGETS)}")
        ids = match_itemids(d_labitems, spec["regex"])
        print(d_labitems[d_labitems["itemid"].isin(ids)].to_string())
        return

    for t in args.targets:
        if t not in TARGETS:
            sys.exit(f"unknown target '{t}'. Known: {list(TARGETS)}")

    # ECG record list + demographics ---------------------------------------
    rl = pd.read_csv(mimic / "waveforms" / "record_list.csv",
                     parse_dates=["ecg_time"])
    rl = rl[["subject_id", "study_id", "ecg_time", "path"]].dropna(subset=["ecg_time"])
    rl = rl.reset_index(drop=True)
    print(f"{len(rl):,} ECG studies, {rl['subject_id'].nunique():,} subjects")

    pat_path = hosp / "patients.csv.gz"
    if pat_path.exists():
        pat = pd.read_csv(pat_path, usecols=["subject_id", "gender", "anchor_age"])
        rl = rl.merge(pat, on="subject_id", how="left")
        rl["sex"] = (rl["gender"] == "F").astype(float)
        rl["age"] = rl["anchor_age"].astype(float)
    else:
        rl["sex"] = np.nan
        rl["age"] = np.nan

    out = rl[["subject_id", "study_id", "ecg_time", "age", "sex"]].copy()

    # Per-target pairing ----------------------------------------------------
    for t in args.targets:
        spec = TARGETS[t]
        ids = set(match_itemids(d_labitems, spec["regex"]))
        if not ids:
            print(f"[warn] {t}: no itemid matched /{spec['regex']}/ -- skipping")
            continue
        print(f"\n=== {t}: itemids {sorted(ids)} (cutoff {spec['op']} {spec['cutoff']}) ===")
        labs = load_target_labs(hosp, ids)
        val = pair_nearest(rl, labs, args.window_h)
        obs = val.notna().astype(int)
        if spec["op"] == ">":
            lab = (val > spec["cutoff"]).astype(float)
        else:
            lab = (val < spec["cutoff"]).astype(float)
        lab[val.isna()] = np.nan
        out[t] = lab.values
        out[f"{t}_val"] = val.values
        out[f"{t}_obs"] = obs.values

        n_obs = int(obs.sum())
        n_pos = int(np.nansum(lab.values))
        floor = 37
        flag = "OK" if min(n_pos, n_obs - n_pos) >= floor else "UNDERPOWERED"
        print(f"  observed={n_obs:,}  n_+={n_pos:,}  n_-={n_obs - n_pos:,}  "
              f"prev={n_pos / max(n_obs,1):.3f}  floor~{floor} {flag}")

    out_dir = ecg_root / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_dir / "ecg_lab_labels.parquet")
    print(f"\nwrote {out_dir / 'ecg_lab_labels.parquet'}  ({len(out):,} studies)")


if __name__ == "__main__":
    main()
