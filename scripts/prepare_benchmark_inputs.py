#!/usr/bin/env python3
"""
prepare_benchmark_inputs.py
============================
Converts your existing SLIMA outputs + TCGA MAF files into the exact
directory structure expected by pattern_informed_abmil_benchmark.py.

WHAT YOU ALREADY HAVE
─────────────────────
  /home/rapids/notebooks/slima/outputs/inference_results_v2_ctranspath_fast/
    TCGA-XX-XXXX-01Z-00-DX1_tiles_224_v2.csv    ← per-tile predictions
    ...

  /path/to/TCGA.LUAD.mutect2.maf  (or .maf.gz)
  /path/to/TCGA.LUSC.mutect2.maf

WHAT THIS SCRIPT PRODUCES
─────────────────────────
  data/
    labels.csv                           ← slide_id, TP53, EGFR, KRAS, STK11, KEAP1, RBM10
    slides/
      TCGA-XX-XXXX-01Z-00-DX1/
        pattern_probs.npy                ← (N_tiles, 6)  float32
        pattern_labels.npy               ← (N_tiles,)    int64
        # embeddings.npy is written by extract_embeddings.py (Step 2, optional but needed
        #   for Baseline 2, Proposed, and Ablation conditions)

USAGE
─────
  # Step 1: build labels.csv + pattern_probs.npy from what you already have
  python prepare_benchmark_inputs.py \\
      --inference_dir /home/rapids/notebooks/slima/outputs/inference_results_v2_ctranspath_fast \\
      --maf_files /data/TCGA.LUAD.mutect2.maf /data/TCGA.LUSC.mutect2.maf \\
      --out_dir data

  # Step 2 (only if you need embeddings — required for ABMIL conditions):
  python extract_embeddings.py \\
      --svs_dir "/home/rapids/notebooks/slima/TGCA LUAD LUSC/TCGA LUAD LUSC" \\
      --data_dir data \\
      --ctranspath_ckpt /home/rapids/notebooks/slima/models/ctranspath.pth
"""

import os, sys, re, gzip, argparse, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict

warnings.filterwarnings("ignore")

# ── target genes ──────────────────────────────────────────────────────────────
GENES = ["TP53", "EGFR", "KRAS", "STK11", "KEAP1", "RBM10"]

# pattern column order must match your FuzzyArcLoss V2 label2id
# adjust if your model uses a different order
PATTERN_ORDER = ["acinar", "lepidic", "micropapillary", "mucinous", "papillary", "solid"]


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1a — MAF → labels.csv
# ══════════════════════════════════════════════════════════════════════════════

def _open_maf(path: str):
    """Open .maf or .maf.gz transparently."""
    if path.endswith(".gz"):
        return gzip.open(path, "rt")
    return open(path, "r")


def _case_from_barcode(barcode: str) -> str:
    """
    TCGA-XX-XXXX-01A-... → TCGA-XX-XXXX
    Keeps only the 12-character case ID (first 3 fields).
    """
    parts = barcode.split("-")
    return "-".join(parts[:3]) if len(parts) >= 3 else barcode


def _slide_prefix_from_barcode(barcode: str) -> str:
    """
    TCGA-XX-XXXX-01A-... → TCGA-XX-XXXX-01
    Slides start with the 15-char sample barcode (first 4 fields).
    """
    parts = barcode.split("-")
    return "-".join(parts[:4]) if len(parts) >= 4 else barcode


def parse_maf(maf_paths: List[str], genes: List[str]) -> pd.DataFrame:
    """
    Returns a DataFrame with columns [case_id, TP53, EGFR, ...].
    One row per TCGA case (TCGA-XX-XXXX); gene columns are binary (1=mutated).

    Key design decisions confirmed from inspecting your actual MAF files:
    ─────────────────────────────────────────────────────────────────────
    1. GDC MAFs include a pre-computed `case_id` column (UUID like
       a5bd7d50-... for LUAD, 0366c911-... for LUSC). We do NOT use this
       — it's a UUID, not a TCGA barcode. We derive the 12-char TCGA case
       ID (TCGA-XX-XXXX) from Tumor_Sample_Barcode instead.

    2. LUAD and LUSC MAFs have identical column names but DIFFERENT column
       ORDER for case_id / tumor_bam_uuid / normal_bam_uuid. This is fine
       because we use column-name lookup, not positional indexing.

    3. Variant_Classification values seen in your files:
         Silent, Missense_Mutation, RNA, Nonsense_Mutation, ...
       We exclude: Silent, 3'UTR, 5'UTR, 3'Flank, 5'Flank, IGR, Intron,
                   RNA (non-coding), lincRNA, Targeted_Region
       We KEEP: Missense_Mutation, Nonsense_Mutation, Frame_Shift_Del,
                Frame_Shift_Ins, Splice_Site, In_Frame_Del, In_Frame_Ins,
                Translation_Start_Site, Nonstop_Mutation
    """
    # Variant classes that do NOT affect protein function → exclude
    NON_CODING = {
        "Silent",
        "3'UTR", "5'UTR",
        "3'Flank", "5'Flank",
        "IGR",
        "Intron",
        "RNA",           # non-coding RNA variants (seen in your LUAD MAF as 'RNA' class)
        "lincRNA",
        "Targeted_Region",
        "De_novo_Start_InFrame",
        "De_novo_Start_OutOfFrame",
    }

    case_gene_hits: Dict[str, Dict[str, int]] = {}   # TCGA-XX-XXXX → {gene: 1}
    all_sequenced_cases: set = set()                  # every case in the MAF = sequenced

    for maf_path in maf_paths:
        print(f"  Parsing MAF: {maf_path}")
        try:
            fh = _open_maf(maf_path)
        except FileNotFoundError:
            print(f"  [WARNING] File not found: {maf_path} — skipping.")
            continue

        header = None
        col    = {}
        n_rows = 0
        n_kept = 0

        for line in fh:
            if line.startswith("#"):
                continue
            fields = line.rstrip("\n").split("\t")

            # ── header row ──────────────────────────────────────────────
            if header is None:
                header = fields
                col    = {h: i for i, h in enumerate(header)}
                # Verify the three columns we definitely need
                missing = [c for c in ("Hugo_Symbol", "Variant_Classification",
                                       "Tumor_Sample_Barcode") if c not in col]
                if missing:
                    print(f"  [ERROR] MAF missing required columns: {missing}")
                    print(f"  First 20 columns found: {list(col.keys())[:20]}")
                    sys.exit(1)
                has_case_uuid = "case_id" in col   # GDC UUID column (not the TCGA barcode)
                print(f"  Columns: {len(header)}  |  GDC case_id UUID column present: {has_case_uuid}")
                continue

            # ── data row ─────────────────────────────────────────────────
            if len(fields) < len(header):
                continue   # malformed row

            n_rows += 1

            # Derive 12-char TCGA case ID from the tumour barcode
            # e.g. TCGA-69-7979-01A-11D-2184-08  →  TCGA-69-7979
            barcode = fields[col["Tumor_Sample_Barcode"]]
            case_id = _case_from_barcode(barcode)
            all_sequenced_cases.add(case_id)

            # Filter to target genes only
            gene = fields[col["Hugo_Symbol"]]
            if gene not in genes:
                continue

            # Filter out non-coding variants
            vclass = fields[col["Variant_Classification"]]
            if vclass in NON_CODING:
                continue

            n_kept += 1
            if case_id not in case_gene_hits:
                case_gene_hits[case_id] = {}
            case_gene_hits[case_id][gene] = 1

        fh.close()
        print(f"  Rows: {n_rows} total  |  {n_kept} functional variants in target genes")
        print(f"  Sequenced cases in this MAF: {len(all_sequenced_cases)}")

    if not all_sequenced_cases:
        print("  [WARNING] No cases found. Check --maf_files paths.")
        return pd.DataFrame(columns=["case_id"] + genes)

    # Build one row per sequenced case
    # Cases absent from case_gene_hits have 0 mutations → wildtype
    all_cases = sorted(all_sequenced_cases)
    rows = []
    for case in all_cases:
        row = {"case_id": case}
        hits = case_gene_hits.get(case, {})
        for g in genes:
            row[g] = hits.get(g, 0)
        rows.append(row)

    df = pd.DataFrame(rows)
    n_cases = len(df)

    print(f"\n  Total sequenced cases across all MAFs: {n_cases}")
    print("  Mutation rates (functional, non-silent variants only):")
    for g in genes:
        n_mut = int(df[g].sum())
        pct   = n_mut / n_cases if n_cases > 0 else 0
        print(f"    {g:8s}: {n_mut:4d}/{n_cases} ({pct:.1%})")

    return df


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1b — Tile CSVs → pattern_probs.npy + pattern_labels.npy
# ══════════════════════════════════════════════════════════════════════════════

def _slide_id_from_csv(csv_name: str) -> str:
    """
    TCGA-69-7978-01Z-00-DX1_tiles_224_v2.csv  →  TCGA-69-7978-01Z-00-DX1
    Strips the _tiles_*_v*.csv suffix.
    """
    name = Path(csv_name).stem                          # drop .csv
    # remove trailing _tiles_<size>_v<n>
    name = re.sub(r"_tiles_\d+_v\d+$", "", name)
    return name


def _detect_prob_columns(columns: List[str]) -> List[str]:
    """
    Find prob_* columns and return them in PATTERN_ORDER.
    Raises if any expected pattern is missing.
    """
    prob_cols_found = {c.replace("prob_", ""): c for c in columns if c.startswith("prob_")}
    ordered = []
    for pat in PATTERN_ORDER:
        if pat in prob_cols_found:
            ordered.append(prob_cols_found[pat])
        else:
            # try case-insensitive
            match = next((v for k, v in prob_cols_found.items()
                          if k.lower() == pat.lower()), None)
            if match:
                ordered.append(match)
            else:
                raise ValueError(
                    f"Pattern '{pat}' not found in CSV columns.\n"
                    f"Found prob columns: {list(prob_cols_found.keys())}\n"
                    f"Expected: {PATTERN_ORDER}\n"
                    f"Set PATTERN_ORDER at the top of this script to match your model's label2id."
                )
    return ordered


def convert_tile_csvs(
    inference_dir: str,
    out_dir: str,
    case_df: Optional[pd.DataFrame] = None,
) -> List[str]:
    """
    For each *_tiles_*_v*.csv in inference_dir:
      - Read tile predictions
      - Save pattern_probs.npy  (N_tiles, 6)
      - Save pattern_labels.npy (N_tiles,)
      - Return list of slide_ids processed

    case_df is the MAF-derived DataFrame; if provided, only slides whose
    case_id appears in case_df are processed (speeds things up).
    """
    inf_path = Path(inference_dir)
    out_path = Path(out_dir) / "slides"
    out_path.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(inf_path.glob("*_tiles_*_v*.csv"))
    if not csv_files:
        # try any csv
        csv_files = sorted(inf_path.glob("*.csv"))
    print(f"\nFound {len(csv_files)} tile CSV files in {inference_dir}")

    if not csv_files:
        print("[ERROR] No CSV files found. Run inference script first.")
        sys.exit(1)

    # build case_id lookup if MAF provided
    case_ids_with_mutations = None
    if case_df is not None and not case_df.empty:
        case_ids_with_mutations = set(case_df["case_id"].tolist())

    processed_slide_ids = []

    for ci, csv_file in enumerate(csv_files):
        slide_id = _slide_id_from_csv(csv_file.name)

        # check if this slide's case has mutation labels
        if case_ids_with_mutations is not None:
            case_id = _case_from_barcode(slide_id)
            if case_id not in case_ids_with_mutations:
                # Still process it — wildtype cases are label 0, not excluded
                pass  # we include all slides; MAF join happens later

        slide_out = out_path / slide_id
        probs_out = slide_out / "pattern_probs.npy"
        label_out = slide_out / "pattern_labels.npy"

        if probs_out.exists() and label_out.exists():
            processed_slide_ids.append(slide_id)
            if (ci + 1) % 50 == 0:
                print(f"  [{ci+1}/{len(csv_files)}] Skipping {slide_id} (already done)")
            continue

        try:
            df = pd.read_csv(csv_file)
        except Exception as e:
            print(f"  [WARN] Cannot read {csv_file.name}: {e}")
            continue

        if df.empty:
            print(f"  [WARN] Empty CSV: {csv_file.name}")
            continue

        # detect probability columns
        try:
            prob_cols = _detect_prob_columns(df.columns.tolist())
        except ValueError as e:
            print(f"  [WARN] {csv_file.name}: {e}")
            continue

        probs  = df[prob_cols].values.astype(np.float32)   # (N, 6)
        labels = np.argmax(probs, axis=1).astype(np.int64) # (N,)

        slide_out.mkdir(parents=True, exist_ok=True)
        np.save(probs_out, probs)
        np.save(label_out, labels)

        processed_slide_ids.append(slide_id)

        if (ci + 1) % 50 == 0 or (ci + 1) == len(csv_files):
            print(f"  [{ci+1}/{len(csv_files)}] {slide_id}: {len(df)} tiles")

    print(f"\n  {len(processed_slide_ids)} slides converted to .npy")
    return processed_slide_ids


# ══════════════════════════════════════════════════════════════════════════════
# STEP 1c — Join slides ↔ mutation labels → labels.csv
# ══════════════════════════════════════════════════════════════════════════════

def build_labels_csv(
    slide_ids: List[str],
    case_df: pd.DataFrame,
    genes: List[str],
    out_dir: str,
) -> pd.DataFrame:
    """
    Match each slide_id to a case in case_df (via TCGA-XX-XXXX prefix).
    Slides from cases NOT in the MAF are assumed wildtype (0) for all genes,
    unless the case simply wasn't sequenced — those get NaN (excluded from
    per-gene analysis).

    In TCGA, all cases in the MAF were sequenced; absence = wildtype.
    """
    # build case_id → mutation dict
    case_to_muts = {}
    if not case_df.empty:
        for _, row in case_df.iterrows():
            case_to_muts[row["case_id"]] = {g: row[g] for g in genes}

    # the full set of sequenced cases (anyone in the MAF cohort = wildtype if absent)
    sequenced_cases = set(case_to_muts.keys())

    rows = []
    unmatched = 0
    for slide_id in slide_ids:
        case_id = _case_from_barcode(slide_id)

        if case_id in case_to_muts:
            # case has mutations in target genes
            row = {"slide_id": slide_id}
            row.update(case_to_muts[case_id])
            rows.append(row)
        elif sequenced_cases:
            # case was sequenced but no mutation in ANY target gene → all wildtype
            row = {"slide_id": slide_id}
            row.update({g: 0 for g in genes})
            rows.append(row)
        else:
            # no MAF at all — still include with NaN
            row = {"slide_id": slide_id}
            row.update({g: float("nan") for g in genes})
            rows.append(row)
            unmatched += 1

    labels_df = pd.DataFrame(rows, columns=["slide_id"] + genes)

    out_path = Path(out_dir) / "labels.csv"
    labels_df.to_csv(out_path, index=False)

    print(f"\n  labels.csv: {len(labels_df)} slides → {out_path}")
    if unmatched:
        print(f"  [WARN] {unmatched} slides had no case match in MAF (set to NaN).")

    print("\n  Gene summary in labels.csv:")
    for g in genes:
        col = labels_df[g]
        n_mut = int(col.sum(skipna=True))
        n_wt  = int((col == 0).sum())
        n_nan = int(col.isna().sum())
        total = n_mut + n_wt
        pct   = n_mut / total if total > 0 else 0
        print(f"    {g}: {n_mut} mutated / {n_wt} WT / {n_nan} unknown  ({pct:.1%} positive)")

    return labels_df


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Prepare benchmark inputs from SLIMA inference CSVs + TCGA MAF files"
    )
    ap.add_argument(
        "--inference_dir", required=True,
        default="/home/rapids/notebooks/slima/outputs/inference_results_v2_ctranspath_fast",
        help="Directory containing *_tiles_224_v2.csv files from the inference script",
    )
    ap.add_argument(
        "--maf_files", nargs="+", required=False, default=[],
        help="One or more TCGA MAF files (.maf or .maf.gz). "
             "Example: --maf_files TCGA.LUAD.mutect2.maf TCGA.LUSC.mutect2.maf",
    )
    ap.add_argument(
        "--out_dir", default="data",
        help="Root output directory (default: data/)",
    )
    ap.add_argument(
        "--genes", nargs="+", default=GENES,
        help=f"Genes to extract from MAF (default: {GENES})",
    )
    ap.add_argument(
        "--pattern_order", nargs="+", default=None,
        help="Order of prob_* columns matching your model's label2id "
             "(default: acinar lepidic micropapillary mucinous papillary solid)",
    )
    args = ap.parse_args()

    # allow overriding pattern order via CLI (in-place mutation avoids global scoping issue)
    if args.pattern_order is not None:
        PATTERN_ORDER[:] = args.pattern_order

    out_dir = args.out_dir
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("  SLIMA Benchmark Input Preparation")
    print("=" * 65)

    # ── Step 1: parse MAF files ──────────────────────────────────────────────
    if args.maf_files:
        print("\n[1/3] Parsing MAF files …")
        case_df = parse_maf(args.maf_files, args.genes)
    else:
        print("\n[1/3] No MAF files provided — will assign NaN for all genes.")
        print("      Run again with --maf_files to add mutation labels.")
        case_df = pd.DataFrame(columns=["case_id"] + args.genes)

    # ── Step 2: convert tile CSVs → .npy ─────────────────────────────────────
    print("\n[2/3] Converting tile CSVs to .npy …")
    slide_ids = convert_tile_csvs(args.inference_dir, out_dir, case_df)

    if not slide_ids:
        print("[ERROR] No slides were processed. Check --inference_dir path.")
        sys.exit(1)

    # ── Step 3: build labels.csv ──────────────────────────────────────────────
    print("\n[3/3] Building labels.csv …")
    build_labels_csv(slide_ids, case_df, args.genes, out_dir)

    print("\n" + "=" * 65)
    print("  DONE.  Next steps:")
    print("=" * 65)
    print(f"""
  Your data/ directory now contains:
    data/labels.csv
    data/slides/<slide_id>/pattern_probs.npy
    data/slides/<slide_id>/pattern_labels.npy

  MISSING: embeddings.npy (needed for Baseline 2, Proposed, Ablation)
  Run the companion script to extract CTransPath embeddings:

    python extract_embeddings.py \\
        --svs_dir "/home/rapids/notebooks/slima/TGCA LUAD LUSC/TCGA LUAD LUSC" \\
        --data_dir {out_dir} \\
        --ctranspath_ckpt /home/rapids/notebooks/slima/models/ctranspath.pth \\
        --model_ckpt /home/rapids/notebooks/slima/outputs/ablation_study_v16_optuna/best_fuzzyarcloss_v2.pth

  Then run the benchmark:

    python pattern_informed_abmil_benchmark.py \\
        --data_dir {out_dir} \\
        --results_dir results \\
        --n_folds 5

  Or test with synthetic data first:
    python pattern_informed_abmil_benchmark.py --generate_synthetic
    python pattern_informed_abmil_benchmark.py --data_dir data --abmil_epochs 5
""")


if __name__ == "__main__":
    main()
