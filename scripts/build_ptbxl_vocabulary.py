#!/usr/bin/env python
"""Build the graded SCP-ECG source-vocabulary channel from PTB-XL.

This is the domain-C counterpart of the 6-d ANORAK pattern vector: a per-record
membership vector over the expert vocabulary, which the aggregator receives
alongside an unconstrained learned encoding of the same signal. The partial
information decomposition of the manuscript is defined between exactly these
two channels.

PTB-XL is unusual and favourable here: each cardiologist statement already
carries a 0-100 likelihood, so the fuzzy membership that has to be learned in
LUAD is supplied as data. The one convention that matters is that a likelihood
of 0 does not mean "absent" -- it means the annotator asserted the statement
without grading it, and it is read as full membership.

Outputs, under $DATA_ROOT/ecg/derived/:
    scp_membership.parquet     ecg_id x 71 statements, values in [0, 1]
    scp_superclass.parquet     ecg_id x 5 diagnostic superclasses
    scp_subclass.parquet       ecg_id x diagnostic subclasses
    scp_vocabulary.csv         the statement dictionary actually used
"""

from __future__ import annotations

import argparse
import ast
import os
import sys
from pathlib import Path

import pandas as pd

SUPERCLASS_COL = "diagnostic_class"
SUBCLASS_COL = "diagnostic_subclass"


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "ecg"


def load_ptbxl(ptbxl_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    db_path = ptbxl_dir / "ptbxl_database.csv"
    scp_path = ptbxl_dir / "scp_statements.csv"
    for p in (db_path, scp_path):
        if not p.exists():
            sys.exit(f"missing {p}\nrun: bash download_ecg.sh --core")

    db = pd.read_csv(db_path, index_col="ecg_id")
    db["scp_codes"] = db["scp_codes"].apply(ast.literal_eval)
    scp = pd.read_csv(scp_path, index_col=0)
    return db, scp


def membership(codes: dict[str, float], vocabulary: pd.Index) -> dict[str, float]:
    """Map one record's scp_codes dict onto a dense membership vector.

    Likelihood 0 is 'asserted, ungraded' rather than 'absent', so it maps to 1.0.
    """
    row = dict.fromkeys(vocabulary, 0.0)
    for code, likelihood in codes.items():
        if code not in row:
            continue
        row[code] = 1.0 if likelihood == 0 else float(likelihood) / 100.0
    return row


def aggregate(memb: pd.DataFrame, scp: pd.DataFrame, level: str) -> pd.DataFrame:
    """Max-pool memberships up the SCP hierarchy.

    Max rather than sum because the aggregation is over a fuzzy set: a record
    belongs to a superclass at least as strongly as to its strongest member
    statement, and summing would let three weak statements outrank one certain
    one.
    """
    groups = scp[level].dropna()
    out = {}
    for name, codes in groups.groupby(groups).groups.items():
        cols = [c for c in codes if c in memb.columns]
        if cols:
            out[name] = memb[cols].max(axis=1)
    return pd.DataFrame(out, index=memb.index).sort_index(axis=1)


def attach_snomed(scp: pd.DataFrame, plus_dir: Path) -> pd.DataFrame:
    """Best-effort join of the PTB-XL+ SNOMED-CT mapping onto the vocabulary."""
    if not plus_dir.exists():
        print(f"[warn] {plus_dir} not found; skipping SNOMED-CT mapping")
        return scp

    for candidate in sorted(plus_dir.rglob("*.csv")):
        try:
            head = pd.read_csv(candidate, nrows=5)
        except Exception:
            continue
        snomed_cols = [c for c in head.columns if "snomed" in c.lower()]
        code_cols = [c for c in head.columns if c.lower() in {"scp_code", "statement", "code"}]
        if snomed_cols and code_cols:
            mapping = pd.read_csv(candidate)
            mapping = mapping.set_index(code_cols[0])[snomed_cols]
            merged = scp.join(mapping, how="left")
            print(f"[ok] SNOMED-CT mapping joined from {candidate.name}")
            return merged

    print("[warn] no SNOMED-CT mapping found in PTB-XL+; skipping")
    return scp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    ap.add_argument(
        "--diagnostic-only",
        action="store_true",
        help="restrict the vocabulary to the 44 diagnostic statements, "
        "dropping the 19 form and 12 rhythm ones",
    )
    args = ap.parse_args()

    ecg_root = resolve_root(args.data_root)
    db, scp = load_ptbxl(ecg_root / "ptb-xl")

    if args.diagnostic_only:
        scp = scp[scp["diagnostic"] == 1]

    vocabulary = scp.index
    print(f"vocabulary: {len(vocabulary)} SCP-ECG statements")
    print(f"records:    {len(db)} ECGs from {db['patient_id'].nunique()} patients")

    memb = pd.DataFrame(
        [membership(c, vocabulary) for c in db["scp_codes"]],
        index=db.index,
    )

    # Records with no statement in the retained vocabulary carry no source-side
    # signal at all and would silently dilute every regime estimate.
    empty = (memb.sum(axis=1) == 0).sum()
    if empty:
        print(f"[warn] {empty} records have an empty membership vector")

    graded = ((memb > 0) & (memb < 1)).any(axis=1).sum()
    print(f"graded:     {graded} records carry at least one likelihood < 100")

    out_dir = ecg_root / "derived"
    out_dir.mkdir(parents=True, exist_ok=True)

    keys = db[["patient_id", "strat_fold", "age", "sex", "filename_lr", "filename_hr"]]
    memb.join(keys).to_parquet(out_dir / "scp_membership.parquet")

    for level, fname in ((SUPERCLASS_COL, "scp_superclass"), (SUBCLASS_COL, "scp_subclass")):
        if level in scp.columns:
            agg = aggregate(memb, scp, level)
            agg.join(keys).to_parquet(out_dir / f"{fname}.parquet")
            print(f"{level}: {agg.shape[1]} classes -> {fname}.parquet")

    attach_snomed(scp, ecg_root / "ptb-xl-plus").to_csv(out_dir / "scp_vocabulary.csv")

    print(f"\nwrote {out_dir}")
    print("strat_fold 1-8 = train, 9 = validation, 10 = test (PTB-XL convention);")
    print("all splits are patient-disjoint, so use them rather than resplitting.")


if __name__ == "__main__":
    main()
