#!/usr/bin/env python
"""Fetch CPTAC-LUAD mutation labels for the cohort-transfer test (domain A').

This is the label side of the within-domain cohort-transfer pre-registered in
the manuscript: the six target genes evaluated on CPTAC-LUAD rather than
TCGA-LUAD. Whole-slide images come from The Cancer Imaging Archive
(download_cptac_wsi.sh); the mutation calls come from cBioPortal study
`luad_cptac_2020` (Gillette et al., Cell 2020), which mirrors the Genomic Data
Commons calls and requires no credentials.

Label definition matches the LUAD methods exactly: a gene is positive for a
patient if it carries at least one NON-SILENT somatic variant (missense,
nonsense, frameshift ins/del, in-frame del, splice-site). Silent/synonymous,
UTR, intronic and flanking variants are dropped.

Outputs, under $DATA_ROOT/luad_cptac/:
    labels.parquet     patient_id x {sample_id, 6 gene 0/1 columns}
    labels.csv         same, for inspection
    prevalence.csv     per-gene positive count, prevalence, and n_+ vs floor 37
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import requests

CBIO = "https://www.cbioportal.org/api"
STUDY = "luad_cptac_2020"
MUT_PROFILE = f"{STUDY}_mutations"
SAMPLE_LIST = f"{STUDY}_sequenced"          # 110 samples with mutation data
COHORT_N = 110                               # sequenced tumours
ESTIMATION_FLOOR = 37                        # n_+ floor from the TCGA-LUAD run

# Entrez IDs for the six target genes (verified against cBioPortal).
GENES = {
    "TP53": 7157, "KRAS": 3845, "EGFR": 1956,
    "STK11": 6794, "KEAP1": 9817, "RBM10": 8241,
}

# Non-silent set kept, per the manuscript's label definition. The MAF profile
# already excludes most silent calls; this guards against any that slip in.
NONSILENT = {
    "Missense_Mutation", "Nonsense_Mutation", "Nonstop_Mutation",
    "Frame_Shift_Ins", "Frame_Shift_Del", "In_Frame_Ins", "In_Frame_Del",
    "Splice_Site", "Splice_Region", "Translation_Start_Site",
}


def resolve_root(cli_root: str | None) -> Path:
    root = cli_root or os.environ.get("DATA_ROOT")
    if not root:
        sys.exit("set DATA_ROOT (source env.sh) or pass --data-root")
    return Path(root) / "luad_cptac"


def fetch_mutations() -> pd.DataFrame:
    url = f"{CBIO}/molecular-profiles/{MUT_PROFILE}/mutations/fetch"
    body = {"sampleListId": SAMPLE_LIST, "entrezGeneIds": list(GENES.values())}
    r = requests.post(url, params={"projection": "DETAILED"}, json=body, timeout=120)
    r.raise_for_status()
    muts = pd.DataFrame(r.json())
    if muts.empty:
        sys.exit("cBioPortal returned no mutations; check study/profile ids")
    return muts


def fetch_all_samples() -> pd.DataFrame:
    """Every sequenced sample in the study, so negatives are real zeros.

    A mutation-only fetch omits patients carrying none of the six genes, which
    would silently shrink the AUROC denominator; we anchor on the full cohort.
    """
    url = f"{CBIO}/studies/{STUDY}/samples"
    r = requests.get(url, params={"projection": "SUMMARY"}, timeout=60)
    r.raise_for_status()
    df = pd.DataFrame(r.json())
    return df[["patientId", "sampleId"]].drop_duplicates("patientId")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None)
    args = ap.parse_args()

    out_dir = resolve_root(args.data_root)
    out_dir.mkdir(parents=True, exist_ok=True)

    muts = fetch_mutations()
    print(f"fetched {len(muts)} mutation records for {len(GENES)} genes")

    kept = muts[muts["mutationType"].isin(NONSILENT)].copy()
    dropped = len(muts) - len(kept)
    if dropped:
        print(f"[info] dropped {dropped} silent/non-coding records")

    ez_to_gene = {v: k for k, v in GENES.items()}
    kept["gene"] = kept["entrezGeneId"].map(ez_to_gene)

    # Anchor on the FULL cohort so patients negative for all six genes are kept
    # as real zeros rather than dropped (they are absent from a 6-gene fetch).
    samples = fetch_all_samples()
    all_patients = sorted(samples["patientId"].tolist())
    labels = samples.set_index("patientId").reindex(all_patients)
    for gene in GENES:
        pos = set(kept.loc[kept["gene"] == gene, "patientId"])
        labels[gene] = [1 if p in pos else 0 for p in labels.index]

    labels.index.name = "patient_id"
    labels = labels.rename(columns={"sampleId": "sample_id"})
    labels.to_parquet(out_dir / "labels.parquet")
    labels.to_csv(out_dir / "labels.csv")

    rows = []
    for gene in GENES:
        n_pos = int(labels[gene].sum())
        rows.append({
            "gene": gene,
            "n_pos": n_pos,
            "prevalence_pct": round(100 * n_pos / COHORT_N, 1),
            "vs_floor_37": "above" if n_pos > ESTIMATION_FLOOR
                           else ("at" if n_pos == ESTIMATION_FLOOR else "below"),
        })
    prev = pd.DataFrame(rows).sort_values("n_pos", ascending=False)
    prev.to_csv(out_dir / "prevalence.csv", index=False)

    print(f"\ncohort: {len(all_patients)} patients (target {COHORT_N})")
    print(prev.to_string(index=False))
    print(f"\nwrote {out_dir}")
    print("next: download_cptac_wsi.sh to fetch the matched H&E WSIs, then run "
          "lchai inference and eval_cptac_transfer.py")


if __name__ == "__main__":
    main()
